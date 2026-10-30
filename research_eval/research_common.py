"""Research field definitions, encounter-safe inputs, and label normalization.

Only cohort identity/admission columns and clinical notes enter model prompts.
The original CSV files are never modified.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "research"
DEFAULT_QUESTIONS = Path(__file__).with_name("research_questions.csv")
EMPTY = {"", "na", "n/a", "none", "null", "nan", "unknown", "not found"}
DATE_FIELDS = {"fever_onset_date", "last_fever_date", "toci_start_date", "toci_stop_date"}
COUNT_FIELDS = {"pressor_use_num", "total_dose_toci"}
YN_FIELDS = {"crs_y_n", "hypotension_y_n", "hypoxia_y_n", "high_flow_o2_y_n",
             "bipap_or_intubation_y_n", "neurotox_y_n", "toci_y_n"}
RESEARCH_FIELDS = ["crs_y_n", "fever_onset_date", "last_fever_date", "max_temp",
                   "hypotension_y_n", "pressor_use_num", "hypoxia_y_n", "high_flow_o2_y_n",
                   "bipap_or_intubation_y_n", "neurotox_y_n", "toci_y_n", "toci_start_date",
                   "toci_stop_date", "total_dose_toci"]

# Candidate passages only: these terms never determine an answer or CRS attribution.
_CRS = r"\b(?:CRS|cytokine\s+release)\b"
_FEVER = r"\b(?:fevers?|febrile|afebrile|T\.?\s*max|temp(?:erature)?)\b"
_TOCI = r"\b(?:tocilizumab|toci|Actemra)\b"
CANDIDATE_PATTERNS = {
    "crs_y_n": _CRS,
    "fever_onset_date": _FEVER, "last_fever_date": _FEVER, "max_temp": _FEVER,
    "hypotension_y_n": _CRS + r"|\b(?:hypotens\w*|blood\s+pressure|BP|orthostat\w*)\b",
    "pressor_use_num": _CRS + r"|\b(?:vasopress\w*|pressors?|norepinephrine|epinephrine|phenylephrine|dopamine|levophed)\b",
    "hypoxia_y_n": _CRS + r"|\b(?:hypoxi\w*|oxygen|O2|SpO2|nasal\s+cannula|room\s+air|respiratory\s+failure)\b",
    "high_flow_o2_y_n": _CRS + r"|\b(?:high[ -]flow|HFNC|oxygen|O2|NRB|non[ -]?rebreather)\b",
    "bipap_or_intubation_y_n": _CRS + r"|\b(?:BiPAP|CPAP|intubat\w*|ventilat\w*|airway)\b",
    "neurotox_y_n": r"\b(?:neurotox\w*|confus\w*|encephalopath\w*|deliri\w*|altered\s+mental|mental\s+status|cogniti\w*|dysarthri\w*|word[ -]finding)\b",
    "toci_y_n": _TOCI, "toci_start_date": _TOCI,
    "toci_stop_date": _TOCI, "total_dose_toci": _TOCI,
}


def candidate_spans(text, targets):
    patterns = dict.fromkeys(CANDIDATE_PATTERNS[t] for t in targets)
    if not patterns:
        return []
    return [(m.start(), m.end()) for m in re.finditer("|".join(patterns), text, re.I)]


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Missing or duplicate CSV headers: {path}")
        rows = list(reader)
        if any(None in row or any(v is None for v in row.values()) for row in rows):
            raise ValueError(f"Malformed CSV row: {path}")
        return rows


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def is_empty(value):
    return value is None or str(value).strip().lower() in EMPTY


def normalize_mrn(value):
    value = str(value).strip()
    if is_empty(value):
        raise ValueError("Missing MRN")
    # CSV exports sometimes serialize an identifier as an integral float.
    return re.sub(r"\.0+$", "", value)


def normalize_date(value):
    if is_empty(value):
        raise ValueError("Missing date")
    value = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
                "%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"Invalid date: {value!r}")


def encounter_key(row):
    return normalize_mrn(row["MRN"]), normalize_date(row.get("BMT_date") or row.get("bmt_date"))


def record_id(key):
    return f"{key[0]}|{key[1]}"


def unique_encounters(rows, source):
    result = {}
    for row in rows:
        key = encounter_key(row)
        if key in result:
            raise ValueError(f"Duplicate MRN + BMT date in {source}; resolve duplicates before running")
        result[key] = row
    return result


def load_questions(path=DEFAULT_QUESTIONS):
    result = {}
    for row in read_csv(path):
        fid = row["Question Number"].strip()
        target = row["target_column"].strip()
        sources = row["note_sources"].split("|")
        if fid in result or target not in RESEARCH_FIELDS or not set(sources) <= {"progress", "discharge", "history"}:
            raise ValueError("Invalid or duplicate research question metadata")
        result[fid] = dict(field_id=fid, field_category=row["category"],
                           field_name=row["Question Text"], field_instructions=row["Options / Instructions"],
                           target_column=target, note_sources=sources)
    if Counter(q["target_column"] for q in result.values()) != Counter(RESEARCH_FIELDS):
        raise ValueError("Question metadata must cover each of the 14 research targets exactly once")
    return result


def normalize_value(field, value):
    """Return (value, status); invalid values never become valid missing answers."""
    if isinstance(value, str) and value.strip().lower() == "not mentioned":
        return None, "not_mentioned"
    if is_empty(value):
        return None, "empty"
    if isinstance(value, (list, dict, bool)):
        return None, "invalid"
    text = str(value).strip()
    if field == "max_temp":
        # A unit suffix does not change a Fahrenheit value. Never strip arbitrary
        # prose, ranges or Celsius units and silently treat them as Fahrenheit.
        match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)\s*(?:\N{DEGREE SIGN}\s*)?(?:F|Fahrenheit)", text, re.I)
        if match:
            text = match[1]
    if field in YN_FIELDS:
        result = {"Y": "Y", "YES": "Y", "N": "N", "NO": "N"}.get(text.upper())
        return result, "valid" if result else "invalid"
    if field in DATE_FIELDS:
        try:
            return normalize_date(text), "valid"
        except ValueError:
            return None, "invalid"
    try:
        number = float(text)
        if not math.isfinite(number) or (field in COUNT_FIELDS and (number < 0 or not number.is_integer())):
            return None, "invalid"
        if field == "max_temp" and not 80 <= number <= 115:
            return None, "invalid"
        return number, "valid"
    except ValueError:
        return None, "invalid"


def gold_label(field, row):
    value, status = normalize_value(field, row.get(field))
    if status != "empty":
        return value, status
    # An explicit N/A label is different from an unannotated blank. CRS absence
    # does not establish absence of infectious/other fever during admission.
    if field in DATE_FIELDS | {"max_temp"} and str(row.get(field, "")).strip().lower() in {"na", "n/a"}:
        return None, "not_applicable"
    # Tocilizumab dates alone have a directly applicable negative parent.
    if field in {"toci_start_date", "toci_stop_date"} and normalize_value("toci_y_n", row.get("toci_y_n"))[0] == "N":
        return None, "not_applicable"
    return None, "unlabeled"


def load_ground_truth(path):
    rows = read_csv(path)
    if not rows or any(f not in rows[0] for f in RESEARCH_FIELDS):
        raise ValueError("Ground truth must contain all 14 research target columns")
    if not any(not is_empty(r[f]) for r in rows for f in RESEARCH_FIELDS):
        raise ValueError("All research labels are blank; target_filtered.csv is a cohort file, not labeled GT")
    return unique_encounters(rows, path)


def progress_date(row):
    """Preserve offset/header conflicts instead of silently relabeling notes."""
    raw = row["note_date"].strip()
    if re.fullmatch(r"-?\d+(?:\.0+)?", raw):
        expected = (datetime.fromisoformat(normalize_date(row["bmt_date"]))
                    + timedelta(days=int(float(raw)))).date().isoformat()
    else:
        expected = normalize_date(raw)
    match = re.search(r"(?:PROGRESS NOTE|Date of (?:Service|note))\s*:?\s*(\d{1,2}/\d{1,2}/\d{2,4})",
                      row["note"][:700], flags=re.I)
    header = normalize_date(match[1]) if match else ""
    return header or expected, expected, bool(header and header != expected)


def load_encounters(data_dir=DEFAULT_DATA, cohort_path=None):
    data_dir = Path(data_dir)
    cohort = unique_encounters(read_csv(cohort_path or data_dir / "target_filtered.csv"), "cohort")
    progress = defaultdict(list)
    admission_sources = {"history": defaultdict(list), "discharge": defaultdict(list)}
    for row in read_csv(data_dir / "progress_notes.csv"):
        progress[encounter_key(row)].append(row)
    for kind, name in (("history", "h_p_notes.csv"), ("discharge", "dc_summary_notes.csv")):
        for row in read_csv(data_dir / name):
            admission_sources[kind][(normalize_mrn(row["MRN"]), normalize_date(row["admit_date"]))].append(row)
    encounters = []
    for key, row in cohort.items():
        admission = normalize_date(row["admit_date"])
        if admission > key[1]:
            raise ValueError("Cohort admission occurs after transplant")
        notes = []
        seen = set()
        for p in progress.get(key, []):
            actual, expected, conflict = progress_date(p)
            identity = (actual, p["note"].strip())
            if not identity[1] or identity in seen:
                continue
            seen.add(identity)
            notes.append(dict(source="progress", date=actual, expected_date=expected,
                              date_conflict=conflict, slot=p["note_date"], text=p["note"].strip()))
        for kind, index in admission_sources.items():
            for p in index.get((key[0], admission), []):
                end = normalize_date(p["discharge_date"]) if kind == "discharge" else admission
                if kind == "discharge" and end < key[1]:
                    raise ValueError("Matched discharge precedes transplant")
                identity = (kind, end, p["note"].strip())
                if not identity[2] or identity in seen:
                    continue
                seen.add(identity)
                notes.append(dict(source=kind, date=end, admit_date=admission,
                                  date_conflict=False, text=p["note"].strip()))
        notes.sort(key=lambda n: (n["date"], n["source"]))
        encounters.append(dict(record_id=record_id(key), record_id_type="MRN+BMT_date",
                               MRN=key[0], bmt_date=key[1], date=key[1], admit_date=admission, notes=notes))
    return encounters


def select_notes(encounter, fields, questions):
    required = {s for fid in fields for s in questions[fid]["note_sources"]}
    return [n for n in encounter["notes"] if n["source"] in required]


def render_note(note):
    header = f"Source: {note['source']}; note date: {note['date']}"
    if note["source"] == "progress":
        header += f"; source day slot: {note['slot']}; slot-derived date: {note['expected_date']}"
    if note.get("date_conflict"):
        header += "; DATE CONFLICT: header and slot disagree"
    if note.get("admit_date"):
        header += f"; admission: {note['admit_date']}"
    return f"[{header}]\n{note['text']}"


def data_audit(data_dir, cohort_path, gt_path, encounters, questions):
    gold = load_ground_truth(gt_path)
    keys = {(e["MRN"], e["bmt_date"]) for e in encounters}
    labels = Counter()
    per_field = {}
    label_issues = []
    for field in RESEARCH_FIELDS:
        counts = Counter(gold_label(field, gold[k])[1] for k in keys if k in gold)
        per_field[field] = dict(counts)
        labels.update(counts)
        for key in sorted(keys & gold.keys()):
            status = gold_label(field, gold[key])[1]
            if status in {"invalid", "unlabeled"}:
                label_issues.append(dict(record_id=record_id(key), target_column=field,
                                         raw_value=gold[key].get(field), status=status))
    raw_progress = [r for r in read_csv(Path(data_dir) / "progress_notes.csv") if encounter_key(r) in keys]
    coverage = []
    candidate_gaps = []
    for encounter in encounters:
        counts = Counter(n["source"] for n in encounter["notes"])
        progress_dates = sorted(n["date"] for n in encounter["notes"] if n["source"] == "progress")
        coverage.append(dict(record_id=encounter["record_id"],
            note_counts={s: counts[s] for s in ("progress", "discharge", "history")},
            missing_sources=[s for s in ("progress", "discharge", "history") if not counts[s]],
            first_progress_date=progress_dates[0] if progress_dates else None,
            last_progress_date=progress_dates[-1] if progress_dates else None))
        label = gold.get((encounter["MRN"], encounter["bmt_date"]))
        if label is None:
            continue
        for fid, question in questions.items():
            target = question["target_column"]
            value, status = gold_label(target, label)
            if status != "valid" or value == "N" or value == 0:
                continue
            notes = select_notes(encounter, [fid], questions)
            if not any(candidate_spans(n["text"], [target]) for n in notes):
                candidate_gaps.append(dict(record_id=encounter["record_id"], field_id=fid,
                    target_column=target, ground_truth=value, available_note_count=len(notes),
                    status="no_candidate_terms_found"))
    return dict(cohort_file=str(cohort_path), ground_truth_file=str(gt_path),
                ground_truth_encounters=len(gold), cohort_encounters=len(encounters),
                matched_ground_truth=len(keys & gold.keys()),
                missing_ground_truth=len(keys - gold.keys()),
                note_coverage={s: sum(any(n["source"] == s for n in e["notes"]) for e in encounters)
                               for s in ("progress", "discharge", "history")},
                note_date_conflicts=sum(n.get("date_conflict", False) for e in encounters for n in e["notes"]),
                raw_progress_rows=len(raw_progress),
                raw_progress_date_conflicts=sum(progress_date(r)[2] for r in raw_progress),
                deduplicated_progress_notes=sum(n["source"] == "progress" for e in encounters for n in e["notes"]),
                label_status=dict(labels), labels_by_field=per_field, label_issues=label_issues,
                encounter_note_coverage=coverage, labeled_value_candidate_gaps=candidate_gaps,
                candidate_screening_note="Keyword screening only: a match does not prove support, and a missing match does not prove a GT error. Review source coverage. Never supplied to extraction prompts.",
                field_count=len(questions),
                decisions=["Use target_filtered.csv only for cohort/encounter identity and admission date.",
                           "Use toci_ptcy_crs.csv for labels; do not send label columns or annotation comments to the model.",
                           "Use exact MRN+BMT_date for progress and GT; exact MRN+admit_date for H&P/discharge.",
                           "Invalid GT labels and unexplained blanks are excluded, with counts reported."])
