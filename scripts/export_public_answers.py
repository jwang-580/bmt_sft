"""Export answer-only records; omit notes, evidence, rationale, and dates outside answers."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
from pathlib import Path
import secrets
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.publication import ANSWER_FIELDS, validate_answers
from prompt_eval.score_predictions import flatten_predictions, read_json_or_jsonl


def export_rows(rows, salt):
    result = []
    for row in rows:
        rid = row.get('record_id') or row.get('hadm_id') or row.get('MRN')
        if rid is None or not str(rid).strip() or 'field_id' not in row or 'answers' not in row:
            raise ValueError('Every exported row needs an identity, field_id, and answers')
        # Incorporate the encounter date into the private HMAC input when present.
        identity = json.dumps([str(row.get('dataset', '')), str(rid), str(row.get('date') or row.get('admit_date') or row.get('BMT_date') or '')])
        public = {k:v for k,v in row.items() if k in ANSWER_FIELDS and k != 'record_id'}
        public['record_id'] = 'record_' + hmac.new(salt, identity.encode(), hashlib.sha256).hexdigest()[:24]
        result.append(public)
    return validate_answers(result)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args(argv)
    output=args.output.resolve()
    if output == args.input.resolve() or output.exists():
        parser.error('Choose a new output path; source files and existing exports are never overwritten')
    rows=flatten_predictions(read_json_or_jsonl(args.input))
    if not rows:
        parser.error('No field-level answer records were found')
    try:
        payload=export_rows(rows, secrets.token_bytes(32))
    except ValueError as exc:
        parser.error(str(exc))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
    print(f'Exported {len(payload)} answer rows without note/evidence/rationale columns to {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
