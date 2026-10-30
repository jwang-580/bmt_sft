"""Audit saved note answers that become 'not mentioned', without model inference.

Reconstruct final source excerpts using exact cache/prompt fingerprints. Emit all
losses and their note evidence, gold comparison, and final rationale for review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_eval.research_common import (DEFAULT_DATA, DEFAULT_QUESTIONS, DATE_FIELDS,
    YN_FIELDS, gold_label, load_encounters, load_ground_truth, load_questions,
    normalize_value, render_note, candidate_spans, write_csv, write_json, file_hash)
from research_eval.research_eval import prediction_failed, scoring_value, SCORING_POLICY
from research_eval.research_extract import (SYSTEM_PROMPT, build_prompt, collect_excerpts,
    note_jobs, quote_spans, render_evidence_notes)
from datetime import date


def matches(field, value, target, target_status, tolerance=False):
    value, status = normalize_value(field, value)
    if target_status not in {"valid", "not_applicable"} or status not in {"valid", "empty"}:
        return False
    if value == target:
        return True
    if not tolerance or status != "valid" or target_status != "valid" or field in YN_FIELDS:
        return False
    error = abs((date.fromisoformat(value)-date.fromisoformat(target)).days) if field in DATE_FIELDS else abs(value-target)
    return error <= (1 if field in DATE_FIELDS else .1+1e-9 if field == "max_temp" else 0)


def analyze(run_dir, output_dir, data_dir=DEFAULT_DATA, questions_path=DEFAULT_QUESTIONS):
    run_dir, output_dir, data_dir = map(Path, (run_dir, output_dir, data_dir))
    predictions = json.loads((run_dir/'research_predictions.json').read_text(encoding='utf-8'))
    if any(p.get('final_pass_contract') == 'structured_findings_integration_v1' for p in predictions):
        raise ValueError('This audit reconstructs historical excerpt-based runs; structured integration uses daily JSON findings instead')
    cache = json.loads((run_dir/'note_evidence.json').read_text(encoding='utf-8'))
    questions = load_questions(questions_path)
    gold = load_ground_truth(data_dir/'toci_ptcy_crs.csv')
    encounters = {e['record_id']:e for e in load_encounters(data_dir)}
    bundles = defaultdict(list)
    for p in predictions:
        bundles[(p['record_id'],p['model'],p['mode'],p['bundle_index'])].append(p)
    candidates = defaultdict(list)
    for v in cache.values():
        for r in v['fields']:
            if r['extraction_status'] == 'extracted':
                candidates[(v['record_id'],v['model'],v['mode'],r['field_id'])].append(r)
    losses = [p for p in predictions if not prediction_failed(p)
              and normalize_value(p['target_column'],p['answers'])[1]=='not_mentioned'
              and (p['record_id'],p['model'],p['mode'],p['field_id']) in candidates]
    output_dir.mkdir(parents=True,exist_ok=True)
    inputs = output_dir/'reconstructed_final_inputs'
    inputs.mkdir(exist_ok=True)
    rebuilt = {}
    rows=[]
    for case_index,p in enumerate(losses,1):
        key=(p['record_id'],p['model'],p['mode'],p['bundle_index'])
        if key not in rebuilt:
            encounter=encounters[p['record_id']]
            fields=[r['field_id'] for r in bundles[key]]
            findings=[]
            note_answers=defaultdict(list)
            for note_index,note,relevant in note_jobs(encounter,fields,questions):
                prompt=build_prompt(encounter,relevant,questions,p['mode'],note_text=render_note(note),note_date=note['date'])
                cache_key=hashlib.sha256((p['model']+'\n'+p['mode']+'\n'+SYSTEM_PROMPT+'\n'+prompt).encode()).hexdigest()
                if cache_key not in cache:
                    raise ValueError(f'Cannot reconstruct note review: cache fingerprint missing for {key}, note {note_index}')
                saved=cache[cache_key]
                assert saved['record_id']==p['record_id'] and saved['note_index']==note_index
                note_rows=saved['fields']
                excerpts=collect_excerpts(note,note_rows,[questions[f]['target_column'] for f in relevant])
                original_crs_spans=candidate_spans(note['text'],['crs_y_n'])
                selected_crs_spans=candidate_spans('\n'.join(excerpts),['crs_y_n'])
                findings.append(dict(note={k:v for k,v in note.items() if k!='text'},excerpts=excerpts))
                for r in note_rows:
                    if r['extraction_status']=='extracted':
                        note_answers[r['field_id']].append(dict(**r,source=note['source'],note_date=note['date'],note_index=note_index,
                            quote_in_original=bool(quote_spans(note['text'],r['evidence_from_text'])),
                            quote_in_selected_excerpts=bool(quote_spans('\n'.join(excerpts),r['evidence_from_text'])),
                            original_crs_context_count=len(original_crs_spans),selected_crs_context_count=len(selected_crs_spans),
                            original_crs_context=[note['text'][max(0,start-100):min(len(note['text']),end+320)] for start,end in original_crs_spans]))
            evidence_text=render_evidence_notes(findings)
            final_prompt=build_prompt(encounter,fields,questions,p['mode'],note_text=evidence_text)
            input_path=inputs/f'bundle_{len(rebuilt)+1:03d}.txt'
            input_path.write_text(final_prompt,encoding='utf-8')
            rebuilt[key]=(note_answers,evidence_text,input_path,final_prompt)
        note_answers,evidence_text,input_path,final_prompt=rebuilt[key]
        field=p['target_column']
        gt=gold[(p['MRN'],p['bmt_date'])]
        target,gt_status=gold_label(field,gt)
        note_rows=note_answers[p['field_id']]
        if not note_rows:
            raise ValueError('Candidate answers in cache are not from the actual bundle')
        for r in note_rows:
            r['quote_in_final_input']=bool(quote_spans(evidence_text,r['evidence_from_text']))
            r['gold_exact']=matches(field,r['answers'],target,gt_status)
            r['gold_tolerant']=matches(field,r['answers'],target,gt_status,True)
        final_value,final_status=scoring_value(field,None,'not_mentioned')
        scorable=gt_status in {'valid','not_applicable'}
        final_exact=scorable and final_value==target
        rows.append(dict(case_id=f'L{case_index:02d}',record_id=p['record_id'],field_id=p['field_id'],target_column=field,
            gold_answer=gt[field],gold_status=gt_status,normalized_gold=target,
            final_answer=p['answers'],final_scored_value=final_value,final_exact=final_exact,
            final_reasoning=p.get('reasoning_trace',''),note_answers=note_rows,
            candidate_exact=any(r['gold_exact'] for r in note_rows),candidate_tolerant=any(r['gold_tolerant'] for r in note_rows),
            unique_note_values=list(dict.fromkeys(str(r['answers']) for r in note_rows)),
            all_quotes_retained=all(r['quote_in_final_input'] for r in note_rows),
            crs_context_dropped=any(r['original_crs_context_count'] and not r['selected_crs_context_count'] for r in note_rows),
            reconstructed_final_input=str(input_path),reconstructed_prompt_hash=hashlib.sha256(final_prompt.encode()).hexdigest(),
            outcome='unlabeled_gold' if not scorable else 'final_negative_correct' if final_exact else 'exact_answer_lost' if any(r['gold_exact'] for r in note_rows) else 'tolerant_answer_lost' if any(r['gold_tolerant'] for r in note_rows) else 'no_gold_matching_note_answer'))
    summary=dict(scoring_policy=SCORING_POLICY,losses=len(rows),encounters=len({r['record_id'] for r in rows}),bundles=len(rebuilt),
        by_field=dict(Counter(r['field_id'] for r in rows)),outcomes=dict(Counter(r['outcome'] for r in rows)),
        all_quotes_retained=all(r['all_quotes_retained'] for r in rows),
        tocilizumab_crs_context_dropped_cases=sum(r['field_id'] in {'R11','R12','R13','R14'} and r['crs_context_dropped'] for r in rows),
        note_answer_conflicts=sum(len(r['unique_note_values'])>1 for r in rows),
        source_hashes={str(run_dir/name):file_hash(run_dir/name) for name in ('research_predictions.json','note_evidence.json','run_manifest.json')},
        limitation='Reconstructed from matching saved note-cache fingerprints; original final raw generations are unavailable. Outcomes compare saved answers with labels and do not adjudicate clinical truth.')
    write_json(output_dir/'aggregation_losses.json',dict(summary=summary,cases=rows))
    flat=[{**{k:v for k,v in r.items() if k not in {'note_answers','unique_note_values'}},
           'unique_note_values':json.dumps(r['unique_note_values']),
           'note_answers':json.dumps(r['note_answers'],ensure_ascii=False)} for r in rows]
    write_csv(output_dir/'aggregation_losses.csv',flat,list(flat[0]) if flat else [])
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--data-dir',default=str(DEFAULT_DATA))
    args=parser.parse_args()
    print(json.dumps(analyze(args.run_dir,args.output_dir,args.data_dir),indent=2))
