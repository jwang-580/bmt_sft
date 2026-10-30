"""Check release boundaries and build a source archive without Git history.

The allowlist is intentional: raw text can occur in JSON, CSV, logs, figures,
manuscript drafts, or training examples. Excluded files are never read by the
archive builder. This check is not a clinical deidentification system.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {
    '.gitignore', '.gitattributes', '.env.example', 'README.md', 'CONTRIBUTING.md',
    'DATA_AVAILABILITY.md', 'CITATION.cff', 'LICENSE', 'LICENSE.md', 'pyproject.toml',
    'requirements.txt', 'extraction_schema.json',
}
SOURCE_DIRS = {'dataset_creation', 'fine_tuning', 'prompt_eval', 'research_eval', 'scripts', 'tests'}
SOURCE_SUFFIXES = {'.py', '.md', '.txt'}
ANALYSIS_SOURCES = {
    'data_analysis/README.md',
    'data_analysis/Figs_scripts/fig2ab_model_size.R',
    'data_analysis/Figs_scripts/fig2cd_extraction_settings.R',
    'data_analysis/Figs_scripts/fig3_finetuned_model.R',
    'data_analysis/Figs_scripts/fig4_research_extraction.R',
    'data_analysis/Figs_scripts/figS1_runtime.R',
    'data_analysis/Figs_scripts/figS2_question_accuracy.R',
}
FIGURE_FILES = {
    'data_analysis/figs/fig1_study_design.png': b'\x89PNG\r\n\x1a\n',
    'data_analysis/figs/fig1_study_design.pdf': b'%PDF-',
}
FIGURE_CELLS = 'data_analysis/data/form_paired_cells.json'
SENSITIVE_PARTS = {'data', 'outputs', 'results', 'tmp', '.git', '.env', 'wandb', '__pycache__', 'venv', '.venv', 'releases'}
ANSWER_FIELDS = {'record_id', 'field_id', 'answers', 'dataset', 'model', 'mode', 'fields_per_call',
                 'note_strategy', 'extraction_success', 'extraction_status', 'response_contract'}
SECRET_PATTERNS = (
    re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    re.compile(r'\b(?:ghp_|github_pat_|hf_)[A-Za-z0-9_]{20,}\b'),
    re.compile(r'\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{24,}\b'),
    re.compile(r'\bAIza[A-Za-z0-9_-]{30,}\b'),
)


def permitted_path(value):
    path = PurePosixPath(value.replace('\\', '/'))
    if path.is_absolute() or '..' in path.parts:
        return False
    if path.as_posix() in ANALYSIS_SOURCES | FIGURE_FILES.keys() | {FIGURE_CELLS}:
        return True
    if any(p in SENSITIVE_PARTS for p in path.parts):
        return False
    if len(path.parts) == 1:
        return path.name in ROOT_FILES
    if path.parts == ('.github', 'workflows', 'checks.yml'):
        return True
    if path.parts == ('research_eval', '.gitignore'):
        return True
    if path.parts == ('research_eval', 'research_questions.csv'):
        return True
    if path.parts == ('prompt_eval', 'results_explorer.html'):
        return True
    if path.parts == ('resources', 'CIBMTR_Form_2400_R11_Items_simplified.csv'):
        return True
    if path.parts == ('resources', 'README.md'):
        return True
    if path.parts[0] == 'docs':
        return path.suffix == '.md'
    if path.parts[0] == 'public_results':
        return len(path.parts) == 2 and path.suffix == '.json'
    return path.parts[0] in SOURCE_DIRS and (
        path.suffix in {'.py', '.md'} or (path.suffix == '.txt' and path.name.startswith('requirements'))
    )


def validate_answers(payload):
    """Allow only bounded, answer-only rows with replacement record identifiers."""
    if not isinstance(payload, list) or not payload:
        raise ValueError('Answer export must be a nonempty JSON list')
    for row in payload:
        if not isinstance(row, dict) or set(row) - ANSWER_FIELDS:
            raise ValueError('Answer export includes disallowed columns')
        if not {'record_id', 'field_id', 'answers'} <= set(row):
            raise ValueError('Answer export is missing required columns')
        if not re.fullmatch(r'record_[0-9a-f]{24}', str(row['record_id'])):
            raise ValueError('Answer exports must use replacement record identifiers')
        if not re.fullmatch(r'(?:R)?[0-9]+', str(row['field_id'])):
            raise ValueError('Invalid field identifier')
        for key, value in row.items():
            if key == 'answers':
                values = value if isinstance(value, list) else [value]
                if len(values) > 100 or any(isinstance(v, (dict, list)) for v in values):
                    raise ValueError('Answer value has an unsupported structure')
                if any(not isinstance(v, (str, int, float, bool, type(None))) for v in values):
                    raise ValueError('Answer values must be scalar or a list of scalars')
            else:
                values = [value]
                if isinstance(value, (dict, list)):
                    raise ValueError('Metadata must be scalar')
            if any(isinstance(v, str) and (len(v) > 500 or '\n' in v or '\r' in v) for v in values):
                raise ValueError('Public answer export contains multiline or oversized text')
            if any(isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isfinite(v) for v in values):
                raise ValueError('Public answer export contains a non-finite numeric value')
    return payload


def validate_figure_cells(payload):
    """Validate paired plotting indicators without answers or narrative fields."""
    if not isinstance(payload, dict) or set(payload) != {'mimic', 'osu'}:
        raise ValueError('Figure cells must contain only the MIMIC and OSU cohorts')
    required = {'case', 'field_id', 'field_name', 'field_category', 'base', 'fine'}
    categories = {'recipient_information', 'donor_information', 'hct',
                  'recipient_status', 'comorbidity', 'conditioning'}
    for cells in payload.values():
        if not isinstance(cells, list) or not 1 <= len(cells) <= 5000:
            raise ValueError('Figure cells must be a bounded nonempty list')
        seen = set()
        for cell in cells:
            if not isinstance(cell, dict) or set(cell) != required:
                raise ValueError('Figure cells contain unexpected columns')
            if (not isinstance(cell['case'], str) or
                    not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}', cell['case']) or
                    not isinstance(cell['field_id'], str) or
                    not re.fullmatch(r'[0-9]{1,4}', cell['field_id'])):
                raise ValueError('Invalid figure record or question identifier')
            if (not isinstance(cell['field_name'], str) or
                    not 1 <= len(cell['field_name']) <= 500 or
                    any(c in cell['field_name'] for c in '\r\n\0') or
                    cell['field_category'] not in categories):
                raise ValueError('Invalid figure question metadata')
            for model in ('base', 'fine'):
                flags = cell[model]
                if (not isinstance(flags, dict) or set(flags) != {'answer_match'} or
                        not isinstance(flags['answer_match'], bool)):
                    raise ValueError('Figure model data must contain only a boolean answer_match')
            key = (cell['case'], cell['field_id'])
            if key in seen:
                raise ValueError('Duplicate figure record/question pair')
            seen.add(key)
    return payload


def inspect_bytes(name, data):
    if not permitted_path(name):
        return ['path is outside the publication allowlist']
    limit = 2_000_000 if name == FIGURE_CELLS else 1_000_000
    if len(data) > limit:
        return ['file exceeds the publication size limit']
    if name in FIGURE_FILES:
        return [] if data.startswith(FIGURE_FILES[name]) else ['figure has an invalid file signature']
    try:
        text = data.decode('utf-8-sig')
    except UnicodeDecodeError:
        return ['file is not UTF-8 text']
    if '\0' in text:
        return ['binary content is not permitted']
    errors = ['possible credential material'] if any(p.search(text) for p in SECRET_PATTERNS) else []
    if name in {'resources/CIBMTR_Form_2400_R11_Items_simplified.csv', 'research_eval/research_questions.csv'}:
        reader = csv.DictReader(io.StringIO(text))
        expected = {'Question Number', 'category', 'Question Text', 'Options / Instructions'}
        count = 129
        if name.startswith('research_eval/'):
            expected |= {'target_column', 'note_sources'}
            count = 14
        rows = list(reader)
        if (set(reader.fieldnames or []) != expected or len(rows) != count or
            any(None in row or any(v is None for v in row.values()) for row in rows) or
            len({row.get('Question Number') for row in rows}) != count):
            errors.append('question catalog must contain only the expected public metadata columns and unique question rows')
    if name == 'extraction_schema.json':
        try:
            json.loads(text)
        except ValueError:
            errors.append('extraction schema is not valid JSON')
    if name == FIGURE_CELLS:
        try:
            validate_figure_cells(json.loads(text))
        except (ValueError, TypeError) as exc:
            errors.append(str(exc))
    if name.startswith('public_results/'):
        try:
            validate_answers(json.loads(text))
        except (ValueError, TypeError) as exc:
            errors.append(str(exc))
    return errors


def git_paths(root, *args):
    result = subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True)
    return [p for p in result.stdout.decode('utf-8').split('\0') if p]


def check_index(root=ROOT):
    errors = []
    for name in git_paths(root, 'ls-files', '-z'):
        # Do not read private blobs just to report that their paths are forbidden.
        if not permitted_path(name):
            errors.append((name, 'path is outside the publication allowlist'))
            continue
        data = subprocess.run(['git', '-C', str(root), 'show', ':' + name], check=True, capture_output=True).stdout
        errors.extend((name, error) for error in inspect_bytes(name, data))
    return errors


def history_private_paths(root=ROOT):
    # Inspect only object paths, never clinical blob contents.
    result = subprocess.run(['git', '-C', str(root), 'rev-list', '--objects', '--all'],
                            check=True, capture_output=True, text=True, encoding='utf-8')
    typed = subprocess.run(['git', '-C', str(root), 'cat-file', '--batch-check=%(objecttype) %(rest)'],
                           input=result.stdout, check=True, capture_output=True, text=True, encoding='utf-8')
    return sorted({line[5:] for line in typed.stdout.splitlines()
                   if line.startswith('blob ') and not permitted_path(line[5:])})


def source_files(root=ROOT):
    files = []
    public_roots = SOURCE_DIRS | {'docs', 'resources', 'public_results', '.github', 'data_analysis'}
    for current, directories, names in os.walk(root, followlinks=False):
        parent = Path(current)
        retained = [d for d in directories if (d not in SENSITIVE_PARTS or
                    (parent / d).relative_to(root).as_posix() == 'data_analysis/data') and
                    (parent != root or d in public_roots)]
        for name in retained:
            if (parent / name).is_symlink():
                raise ValueError('Symlink directories are not permitted in publication sources')
        directories[:] = retained
        for name in names:
            item = parent / name
            if not permitted_path(item.relative_to(root).as_posix()):
                continue
            if item.is_symlink():
                raise ValueError('Symlinks are not permitted in publication sources')
            files.append(item)
    return sorted(files)


def build_archive(destination, root=ROOT):
    destination = Path(destination).resolve()
    private_output = (root / 'releases').resolve()
    if private_output not in destination.parents or destination.suffix != '.zip':
        raise ValueError('Release archives must be written below releases/ with a .zip suffix')
    if destination.exists():
        raise ValueError('Refusing to overwrite an existing release archive')
    selected = []
    for path in source_files(root):
        name = path.relative_to(root).as_posix()
        data = path.read_bytes()
        errors = inspect_bytes(name, data)
        if errors:
            raise ValueError(f'{name}: {", ".join(errors)}')
        selected.append((name, data))
    if not selected:
        raise ValueError('No publication sources found')
    manifest = {'format': 'bmt_source_release_v1', 'contains_git_history': False,
                'files': [{'path': n, 'sha256': hashlib.sha256(d).hexdigest()} for n,d in selected]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in selected:
            archive.writestr(name, data)
        archive.writestr('RELEASE_MANIFEST.json', json.dumps(manifest, indent=2)+'\n')
    return len(selected)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('check', help='Check current Git index paths and contents')
    sub.add_parser('history', help='Find non-public paths in existing Git history')
    pack = sub.add_parser('package', help='Build an allowlisted archive from current source files, without Git history')
    pack.add_argument('--output', type=Path, default=ROOT/'releases/BMT-GRAFT-12B-source.zip')
    args = parser.parse_args(argv)
    if args.command == 'check':
        errors = check_index()
        for name, reason in errors:
            print(f'BLOCKED {name}: {reason}')
        print(f'Publication index check: {len(errors)} issue(s)')
        return int(bool(errors))
    if args.command == 'history':
        paths = history_private_paths()
        for name in paths:
            print('PRIVATE HISTORY PATH ' + name)
        print(f'History check: {len(paths)} non-public path(s); no history was changed')
        return int(bool(paths))
    try:
        count = build_archive(args.output)
    except ValueError as exc:
        parser.exit(1, str(exc)+'\n')
    print(f'Created source archive with {count} allowlisted files and no Git history: {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
