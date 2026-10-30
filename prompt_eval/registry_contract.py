"""Validate new registry responses without inventing successful N/A answers.

Historical results are not rewritten. This module has no GPU dependencies.
"""
from __future__ import annotations

import math
import re

try:
    from .extraction_prompts import mode_settings
except ImportError:
    from extraction_prompts import mode_settings

CONTRACT_VERSION = "registry_response_v2"


def normalize_field_id(value):
    return re.sub(r"^(?:Question|Field)\s+", "", str(value).strip(), flags=re.I)


def parse_non_json_generation(text, field_ids):
    allowed = set(field_ids)
    records = []
    seen = set()
    for line in text.splitlines():
        match = re.match(r"\s*(?:(?:Question|Field)\s*)?([A-Za-z]*\d+[A-Za-z]*)\s*[:=-]\s*(.*?)\s*$", line)
        if not match:
            continue
        fid = normalize_field_id(match.group(1))
        if fid not in allowed:
            raise ValueError("Response contains an unrequested field")
        if fid in seen:
            raise ValueError("Response contains a duplicate field")
        seen.add(fid)
        records.append({"field_id": fid, "answers": match.group(2)})
    return records


def valid_answer(value):
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, list):
        return bool(value) and all(isinstance(v, str) and v.strip() for v in value)
    return False


def coerce_records(raw_records, note, field_ids, question_metadata, mode):
    """Bind fields explicitly; never guess their identity from list position."""
    if not isinstance(raw_records, list):
        raise ValueError("Response must contain a list of field objects")
    if len(field_ids) != len(set(field_ids)):
        raise ValueError("Requested field identifiers must be unique")
    allowed = set(field_ids)
    names = {}
    for fid in field_ids:
        name = question_metadata[fid]["field_name"].strip().casefold()
        names.setdefault(name, []).append(fid)
    by_field = {}
    for raw in raw_records:
        if not isinstance(raw, dict):
            raise ValueError("Every response field must be an object")
        explicit = [normalize_field_id(raw[k]) for k in ("field_id", "question_id", "question_number")
                    if raw.get(k) is not None and str(raw[k]).strip()]
        if len(set(explicit)) > 1:
            raise ValueError("Response contains conflicting field identifiers")
        fid = explicit[0] if explicit else ""
        if not fid:
            candidates = names.get(str(raw.get("field_name", "")).strip().casefold(), [])
            if len(candidates) == 1:
                fid = candidates[0]
        if fid not in allowed:
            raise ValueError("Response field cannot be bound to a requested identifier")
        if fid in by_field:
            raise ValueError("Response contains a duplicate field")
        by_field[fid] = raw
    output = []
    for fid in field_ids:
        raw = by_field.get(fid, {})
        answer = raw.get("answers")
        success = valid_answer(answer)
        error = "" if success else ("missing_field" if fid not in by_field else "missing_or_invalid_answer")
        row = {
            "record_id": note["record_id"], "record_id_type": note["record_id_type"],
            "date": note["date"], "field_id": fid,
            **{k: question_metadata[fid][k] for k in ("field_category", "field_name", "field_instructions")},
            "answers": answer if success else None,
            "extraction_success": success,
            "extraction_status": "extracted" if success else "failed",
            "response_contract": CONTRACT_VERSION, "parse_error": error,
        }
        for key in mode_settings(mode)["extra_fields"]:
            value = raw.get(key, "")
            if not isinstance(value, str):
                raise ValueError(f"{key} must be a string")
            row[key] = value
        output.append(row)
    return output


def require_complete_predictions(predictions):
    if not predictions or any(not prediction_is_usable(p) for p in predictions):
        raise ValueError("Incomplete response: every requested field requires a returned answer")


def validate_resume_predictions(predictions):
    if any(p.get("response_contract") != CONTRACT_VERSION for p in predictions):
        raise ValueError("Cannot resume historical registry outputs under a new response contract; choose a new output directory")


def prediction_is_usable(prediction):
    """Old rows can be scored, but their raw omissions cannot be recovered here."""
    return (
        not prediction.get("parse_error")
        and prediction.get("extraction_success") is not False
        and prediction.get("extraction_status") not in {"failed", "invalid", "missing"}
        and valid_answer(prediction.get("answers"))
    )
