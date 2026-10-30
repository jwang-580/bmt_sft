"""Apply a versioned, source-supported label revision from the manual error audit.

This creates a CSV input for the evaluation pipeline, not independent ground truth.
It does not change prediction values or repair first-pass extraction defects.
"""
from __future__ import annotations
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_eval.research_common import (ROOT, DEFAULT_DATA, read_csv, write_csv,
    write_json, file_hash, encounter_key, record_id, gold_label, normalize_value)

# Source chronology/cause is ambiguous in these three daily/discharge conflicts.
AMBIGUOUS_SOURCE_CONFLICTS = {'E218', 'E235', 'E292'}
# The observed event is explicitly excluded by the target's CRS-causality rule.
# These are explicit scope recodings, never negatives inferred from silence.
EXCLUDED_CAUSE_VALUES = {
    'E019':'N', 'E020':'0', 'E021':'N', 'E022':'N', 'E023':'N',
    'E081':'N', 'E105':'N', 'E106':'N', 'E118':'N', 'E119':'N',
    'E206':'N', 'E271':'N', 'E272':'N', 'E273':'N', 'E309':'N',
    'E412':'N', 'E452':'N',
}

def revise():
    run=ROOT/'research_eval/gemma_finetuned'
    audit_path=ROOT/'outputs/gemma_remaining_errors_20261005/error_audit.json'
    audit=json.loads(audit_path.read_text(encoding='utf-8'))
    source=DEFAULT_DATA/'toci_ptcy_crs.csv'
    output=DEFAULT_DATA/'toci_ptcy_crs_adjudicated.csv'
    out=ROOT/'outputs/gemma_label_revision_20261005'
    out.mkdir(exist_ok=True)
    source_hash=file_hash(source)
    prediction_hash=file_hash(run/'research_predictions.json')
    # After promotion the original label CSV is removed. Verify the applied
    # revision rather than recreating an alternate label file on a repeat run.
    revision_path=out/'label_revision.json'
    if revision_path.is_file():
        applied=json.loads(revision_path.read_text(encoding='utf-8'))
        if source_hash==applied['revised_sha256']:
            assert prediction_hash==applied['prediction_sha256']
            current={record_id(encounter_key(r)):r for r in read_csv(source)}
            for change in applied['changes']:
                assert current[change['record_id']][change['target_column']]==change['revised_label']
            print(json.dumps(dict(status='already applied',ground_truth=source.relative_to(ROOT).as_posix(),sha256=source_hash)))
            return
    assert source_hash==audit['source_hashes'][str(source)]
    assert prediction_hash==audit['source_hashes'][str(run/'research_predictions.json')]
    rows=read_csv(source)
    original={record_id(encounter_key(r)):dict(r) for r in rows}
    index={record_id(encounter_key(r)):r for r in rows}
    ledger=[]
    seen=set()
    for e in audit['cases']:
        if e['category']!=4:continue
        eid=e['error_id']; subtype=e['subtype']
        if subtype=='rounding':
            action='retain';new=e['gold_answer']
            rationale='Within the existing temperature tolerance; insufficient evidence to call the annotation itself wrong.'
        elif (subtype=='source_conflict' and eid not in AMBIGUOUS_SOURCE_CONFLICTS) or (
                subtype=='episode_scope' and e['prediction_status']=='valid'):
            action='replace';new=e['predicted_answer']
            assert normalize_value(e['target_column'],new)[1]=='valid',eid
            rationale='Adopt the source-supported extracted value under the current question scope. '+e['assessment']
        elif eid in EXCLUDED_CAUSE_VALUES:
            action='replace';new=EXCLUDED_CAUSE_VALUES[eid]
            rationale='Recode the explicitly documented non-CRS event as excluded by this CRS-specific target. '+e['assessment']
        else:
            action='unlabel';new=''
            rationale='Available evidence cannot establish a reliable encounter-level label; exclude rather than fabricate a negative or copy an unknown prediction. '+e['assessment']
        rid=e['record_id'];field=e['target_column'];key=(rid,field)
        assert key not in seen,key
        seen.add(key)
        assert index[rid][field]==e['gold_answer'],eid
        index[rid][field]=new
        ledger.append(dict(error_id=eid,case=e['case'],record_id=rid,field_id=e['field_id'],target_column=field,
            original_label=e['gold_answer'],revised_label=new,action=action,audit_subtype=subtype,
            source_assessment=e['assessment'],rationale=rationale,
            source_passages=e['source_passages'],extracted_answer=e['predicted_answer'],
            prediction_status=e['prediction_status'],original_annotation=e['gold_annotation']))

    # The entire encounter has only an unrelated historical discharge document.
    # Exclude its matching labels too, to avoid retaining accidental agreements.
    bad_source_rids={e['record_id'] for e in audit['cases'] if e['subtype']=='source_mismatch'}
    predictions=json.loads((run/'research_predictions.json').read_text(encoding='utf-8'))
    for p in predictions:
        rid=p['record_id'];field=p['target_column'];key=(rid,field)
        if rid not in bad_source_rids or key in seen:continue
        seen.add(key)
        old=index[rid][field];index[rid][field]=''
        ledger.append(dict(error_id=None,case=p['manual_review_case'],record_id=rid,field_id=p['field_id'],
            target_column=field,original_label=old,revised_label='',action='unlabel',audit_subtype='source_mismatch',
            source_assessment='Only supplied discharge source describes May 2023 Legionella/sepsis, not this August 2024 transplant admission.',
            rationale='Exclude every field of this unusable-source encounter, including originally matching predictions.',
            source_passages=[],extracted_answer=p['answers'],prediction_status=p['extraction_status'],
            original_annotation=original[rid][next(reversed(original[rid]))]))

    for change in ledger:
        if change['action']=='unlabel':
            # This also guards against an implicit N/A from a negative toci parent.
            assert gold_label(change['target_column'],index[change['record_id']])[1]=='unlabeled',change
    changed={(c['record_id'],c['target_column']) for c in ledger if c['original_label']!=c['revised_label']}
    actual={(rid,f) for rid,r in index.items() for f in r if r[f]!=original[rid][f]}
    assert changed==actual
    assert all(e['category']==4 or (e['record_id'],e['target_column']) not in changed for e in audit['cases'])
    write_csv(output,rows,list(rows[0]))
    assert read_csv(output)==rows
    assert file_hash(source)==source_hash and file_hash(run/'research_predictions.json')==prediction_hash
    counts=dict(Counter(c['action'] for c in ledger))
    manifest=dict(revision='source_reviewed_extraction_informed_labels_v1',completed_utc=datetime.now(timezone.utc).isoformat(),
        original_ground_truth=source.relative_to(ROOT).as_posix(),original_sha256=source_hash,
        revised_ground_truth=output.relative_to(ROOT).as_posix(),revised_sha256=file_hash(output),
        audit_source=audit_path.relative_to(ROOT).as_posix(),audit_sha256=file_hash(audit_path),
        prediction_source=(run/'research_predictions.json').relative_to(ROOT).as_posix(),prediction_sha256=prediction_hash,
        selection='Audited category 4 fields only, plus all fields in the source-mismatched encounter.',
        policy='Replace source-supported values; mask unsupported/conflicting labels; retain already-tolerated rounding discrepancies. Category 1/2 extraction failures are not relabeled.',
        evaluation_type='extraction-informed label adjudication; not an independent model benchmark',
        source_encounters=len(rows),reviewed_fields=len(ledger),changed_cells=len(changed),
        changed_encounters=len({rid for rid,f in changed}),actions=counts,
        missing_source_encounters=sorted(bad_source_rids),changes=ledger)
    write_json(out/'label_revision.json',manifest)
    write_json(DEFAULT_DATA/'toci_ptcy_crs_adjudicated.manifest.json',{k:v for k,v in manifest.items() if k!='changes'})
    lines=['# Label correction ledger','',
        '| Error | Case | Field | Original | Revised | Action | Reason |',
        '| --- | ---: | --- | --- | --- | --- | --- |']
    for c in ledger:
        cells=[c['error_id'] or 'whole encounter',c['case'],c['field_id'],c['original_label'],c['revised_label'] or '[unlabeled]',c['action'],c['rationale']]
        lines.append('| '+' | '.join(str(v).replace('|','/').replace('\n',' ') for v in cells)+' |')
    (out/'label_changes.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k not in {'changes','missing_source_encounters'}},indent=2))

if __name__=='__main__':revise()
