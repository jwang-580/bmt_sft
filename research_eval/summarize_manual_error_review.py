"""Summarize explicit manual error adjudications; never choose clinical categories."""
from __future__ import annotations
import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_eval.research_common import ROOT, load_questions, load_ground_truth, candidate_spans, file_hash, write_json

CATEGORIES = {
    1:'First-pass omission or incorrect extraction',
    2:'Missing validated first-pass results',
    3:'Standalone synthesis decision error',
    4:'Ground-truth / scope ambiguity or source disagreement',
}

def read_decisions(path):
    decisions={}
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line or line.startswith('#'):continue
        ids,category,subtype,assessment=line.split('|',3)
        if '-' in ids:
            start,end=ids.split('-')
            ids=[f'E{i:03d}' for i in range(int(start[1:]),int(end[1:])+1)]
        else:ids=[ids]
        for eid in ids:
            if eid in decisions:raise ValueError(f'Duplicate review decision: {eid}')
            category=int(category)
            if category not in CATEGORIES:raise ValueError(category)
            decisions[eid]=dict(category=category,category_name=CATEGORIES[category],
                                subtype=subtype,assessment=assessment)
    return decisions

def passages(encounter, target):
    results=[]
    for ni,note in enumerate(encounter['notes']):
        if note['source']=='history':continue
        spans=candidate_spans(note['text'],[target])
        if target=='neurotox_y_n':
            spans += [(m.start(),m.end()) for m in re.finditer(r'\b(?:AMS|hallucinat\w*|disorient\w*|altered\s+mentation)\b',note['text'],re.I)]
        windows=[]
        for start,end in sorted(spans):
            start=max(0,start-160);end=min(len(note['text']),end+420)
            if windows and start<=windows[-1][1]:windows[-1][1]=max(windows[-1][1],end)
            else:windows.append([start,end])
        results.extend(dict(note_index=ni,source=note['source'],date=note['date'],
            start=start,end=end,text=note['text'][start:end]) for start,end in windows)
    return results

def summarize(out):
    run=ROOT/'research_eval/gemma_finetuned'
    errors=json.loads((out/'error_review_inputs.json').read_text(encoding='utf-8'))
    notes={e['record_id']:e for e in json.loads((out/'source_notes.json').read_text(encoding='utf-8'))}
    decisions=read_decisions(out/'manual_error_classifications.txt')
    gold=load_ground_truth(ROOT/'data/research/toci_ptcy_crs.csv')
    questions=load_questions()
    assert len({r['error_id'] for r in errors})==len(errors)==459
    assert set(decisions)<={r['error_id'] for r in errors}
    reviewed=[]
    for r in errors:
        if r['prediction_status']=='failed':
            assert r['missing_reviews'] and r['error_id'] not in decisions
            d=dict(category=2,category_name=CATEGORIES[2],subtype='missing_validated_review',
                assessment=f"{len(r['missing_reviews'])} required note reviews are absent from the validated cache. No usable affirmative value permits a completed final answer.")
        else:
            if r['error_id'] not in decisions:raise ValueError(f'Unreviewed error: {r["error_id"]}')
            d=decisions[r['error_id']]
        if d['category']==2:assert r['missing_reviews'],r['error_id']
        g=gold[(r['MRN'],r['bmt_date'])]
        annotation=g[next(reversed(g))]
        confidence=('unresolved_source_gap' if d['subtype']=='source_insufficient' else
                    'scope_or_causality_requires_adjudication' if d['subtype'] in {'attribution_scope','episode_scope','threshold_scope'} else
                    'observable_source_label_disagreement' if d['category']==4 else
                    'documented_pipeline_failure' if d['category']==2 else 'source_supported_extraction_defect')
        reviewed.append(dict(**r,**d,assessment_status=confidence,gold_annotation=annotation,
            source_note_count=len(notes[r['record_id']]['notes']),
            source_passages=[] if r['prediction_status']=='failed' else passages(notes[r['record_id']],r['target_column'])))
    assert len(reviewed)==459
    summaries=[]
    substantive=[r for r in reviewed if r['tolerance_match']!='True']
    for cat,name in CATEGORIES.items():
        rows=[r for r in reviewed if r['category']==cat]
        serious=[r for r in substantive if r['category']==cat]
        summaries.append(dict(category=cat,name=name,exact_mismatches=len(rows),
            percent_exact_mismatches=100*len(rows)/459,
            beyond_tolerance=len(serious),percent_beyond_tolerance=100*len(serious)/424,
            percent_all_scorable_fields=100*len(rows)/2408,
            affected_encounters=len({r['record_id'] for r in rows}),
            percent_all_encounters=100*len({r['record_id'] for r in rows})/191,
            subtypes=dict(Counter(r['subtype'] for r in rows))))
    assert sum(r['exact_mismatches'] for r in summaries)==459
    assert sum(r['beyond_tolerance'] for r in summaries)==424
    source_hashes={str(run/name):file_hash(run/name) for name in ['research_predictions.json','note_evidence.json','run_manifest.json','research_evaluation_details.csv']}
    source_hashes.update({str(ROOT/'data/research'/name):file_hash(ROOT/'data/research'/name) for name in ['progress_notes.csv','dc_summary_notes.csv','h_p_notes.csv','toci_ptcy_crs.csv','target_filtered.csv']})
    source_hashes[str(ROOT/'research_eval/research_questions.csv')]=file_hash(ROOT/'research_eval/research_questions.csv')
    expected=json.loads((out/'initial_counts.json').read_text())['source_hashes']
    assert all(file_hash(run/name)==digest for name,digest in expected.items()),'Final predictions/scoring changed during audit'
    write_json(out/'error_audit.json',dict(method='Explicit manual field-level source review; counts computed from the frozen adjudication ledger.',
        exact_mismatches=459,beyond_tolerance=424,within_tolerance_only=35,
        affected_encounters=124,scorable_fields=2408,excluded_unlabeled_or_invalid_fields=266,
        categories=summaries,source_hashes=source_hashes,
        ledger_hash=file_hash(out/'manual_error_classifications.txt'),cases=reviewed))
    write_json(out/'incidence_summary.json',summaries)
    field_rows=[]
    for fid,q in questions.items():
        rows=[r for r in reviewed if r['field_id']==fid]
        field_rows.append(dict(field_id=fid,target=q['target_column'],exact_errors=len(rows),
            beyond_tolerance=sum(r['tolerance_match']!='True' for r in rows),
            categories={str(c):sum(r['category']==c for r in rows) for c in CATEGORIES}))
    write_json(out/'incidence_by_field.json',field_rows)
    # A compact readable ledger accompanies the full source-linked JSON.
    lines=['# Complete remaining-error review ledger','',
        'Every row is one scorable exact mismatch. Category 4 is a review flag, not a declaration that the clinical label is wrong. Full daily findings, source passages with offsets, and annotation comments are in `error_audit.json`.','',
        '| Error | Case | Field | Gold | Final | Category | Within tolerance | Assessment |',
        '| --- | ---: | --- | --- | --- | ---: | --- | --- |']
    for r in reviewed:
        cells=[r['error_id'],r['case'],r['field_id'],r['gold_answer'],r['predicted_answer'] or '[failed]',r['category'],r['tolerance_match'],r['assessment']]
        lines.append('| '+' | '.join(str(v).replace('|','/').replace('\n',' ') for v in cells)+' |')
    (out/'error_ledger.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-dir',type=Path,default=ROOT/'outputs/gemma_remaining_errors_20261005')
    summarize(parser.parse_args().review_dir)
