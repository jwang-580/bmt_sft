"""Shared prompt text for SFT dataset generation and extraction evaluation.

The baseline evaluation prompt intentionally calls ``build_sft_user_prompt``
so its wording stays identical to the prompt stored in the SFT dataset.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping


ANNOTATION_SYSTEM_PROMPT = """You create high-quality supervised fine-tuning annotations for clinical information extraction.

The ground-truth answer supplied for each field is fixed and authoritative. Do not revise, reinterpret, normalize, or replace it. Your task is only to add:
1. evidence_from_text: the shortest exact quote from the clinical note that supports the fixed answer. Use an empty string when the answer is unsupported because the information is absent, unknown, or not applicable.
2. reasoning_trace: one brief, visible rationale explaining how the evidence or form branching supports the fixed answer. This is a concise annotation, not hidden chain-of-thought.

Do not introduce facts that are not present in the note or implied by the supplied form instructions."""

EXTRACTION_SYSTEM_PROMPT = (
    "You are a careful clinical information extraction system for BMT notes. "
    "Extract only information supported by the supplied note. "
    "Use standard CIBMTR Form 2400 option wording when an option list is supplied."
)

DEFAULT_MODES = [
    "baseline",
    "non_json",
    "no_reasoning",
    "few_shot",
    "no_reasoning_trace_field",
    "no_evidence_field",
]

JSON_MODE_FIELDS = {
    "baseline": ["evidence_from_text", "reasoning_trace"],
    "few_shot": ["evidence_from_text", "reasoning_trace"],
    "no_reasoning": ["evidence_from_text"],
    "no_reasoning_trace_field": ["evidence_from_text"],
    "no_evidence_field": ["reasoning_trace"],
}

SFT_OUTPUT_FIELDS = [
    "record_id",
    "record_id_type",
    "date",
    "field_id",
    "field_category",
    "field_name",
    "field_instructions",
    "answers",
    "evidence_from_text",
    "reasoning_trace",
]


def mode_settings(mode: str) -> dict[str, Any]:
    if mode not in DEFAULT_MODES:
        raise ValueError(f"Unknown mode {mode}. Valid modes: {', '.join(DEFAULT_MODES)}")
    return {
        "json": mode != "non_json",
        "thinking": mode
        in {"baseline", "non_json", "few_shot", "no_reasoning_trace_field", "no_evidence_field"},
        "few_shot": mode == "few_shot",
        "extra_fields": JSON_MODE_FIELDS.get(mode, []),
    }


def build_system_prompt() -> str:
    return EXTRACTION_SYSTEM_PROMPT


def build_questions_block(
    field_ids: Iterable[str], question_metadata: Mapping[str, Mapping[str, Any]]
) -> str:
    blocks = []
    for field_id in field_ids:
        field = question_metadata[field_id]
        block = [
            f"Field {field_id}",
            f"Category: {field['field_category']}",
            f"Field name: {field['field_name']}",
        ]
        if field["field_instructions"] and field["field_instructions"] != "[]":
            block.append(f"Options/Instructions: {field['field_instructions']}")
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


def build_sft_user_prompt(
    *,
    record_id: str,
    record_id_type: str,
    date: str,
    note_text: str,
    fields_block: str,
) -> str:
    """Return the canonical SFT user prompt; do not change its wording casually."""
    return f"""Record:
{record_id_type}: {record_id}
date: {date}

Clinical note:
{note_text}

Fields:

{fields_block}

Return a JSON array with exactly one object per field. Each object must have these keys:
{json.dumps(SFT_OUTPUT_FIELDS)}

Rules:
- Use the exact metadata supplied in the prompt.
- Put the extracted answer in answers. Use "N/A" when the answer is not found.
- For check-all-that-apply fields, answers may be an array of selected options.
- evidence_from_text must be a concise exact quote from the note. Use "" when no supporting text exists.
- reasoning_trace must be one brief visible rationale, not hidden chain-of-thought.
- Do not add any text outside the JSON array."""


def build_annotation_user_prompt(
    *, record_id: str, date: str, note_text: str, supplied_fields: list[dict[str, Any]]
) -> str:
    return f"""Record:
hadm_id: {record_id}
date: {date}

Clinical note:
{note_text}

Fields and fixed ground-truth answers:
{json.dumps(supplied_fields, ensure_ascii=False, indent=2)}

Return exactly one annotation for every supplied field, in the same order. Copy each field_id exactly. Do not return or alter the fixed answer. For an N/A, unknown, or not-applicable answer, evidence_from_text may be empty, but reasoning_trace must still briefly explain absence or form-branch logic."""


def few_shot_block() -> str:
    return """Few-shot examples:

Example 1 clinical text:
Patient admitted for allogeneic matched sibling peripheral blood stem cell transplant. Donor is male. GVHD prophylaxis tacrolimus and methotrexate.

Example 1 answer:
[
  {
    "field_id": "32",
    "field_category": "donor_information",
    "field_name": "Specify donor",
    "field_instructions": "['Autologous', 'Allogeneic, related', 'Allogeneic, unrelated']",
    "answers": "Allogeneic, related",
    "evidence_from_text": "allogeneic matched sibling",
    "reasoning_trace": "Matched sibling indicates a related allogeneic donor."
  },
  {
    "field_id": "33",
    "field_category": "donor_information",
    "field_name": "Specify product type (check all that apply)",
    "field_instructions": "['Bone marrow - Go to question 35', 'PBSC - Go to question 35']",
    "answers": ["PBSC - Go to question 35"],
    "evidence_from_text": "peripheral blood stem cell transplant",
    "reasoning_trace": "Peripheral blood stem cell transplant maps to PBSC."
  }
]

Example 2 clinical text:
The note does not mention health insurance.

Example 2 answer:
[
  {
    "field_id": "17",
    "field_category": "recipient_information",
    "field_name": "Specify type of health insurance (check all that apply)",
    "field_instructions": "",
    "answers": "N/A",
    "evidence_from_text": "",
    "reasoning_trace": ""
  }
]"""


def build_user_prompt(note, field_ids, question_metadata, mode):
    settings = mode_settings(mode)
    questions = build_questions_block(field_ids, question_metadata)

    # Baseline evaluation must be distribution-aligned with the completed SFT
    # dataset, including exact rule wording and whitespace.
    if mode == "baseline":
        return build_sft_user_prompt(
            record_id=note["record_id"],
            record_id_type=note["record_id_type"],
            date=note["date"],
            note_text=note["note"],
            fields_block=questions,
        )

    common = f"""Record:
{note['record_id_type']}: {note['record_id']}
date: {note['date']}

Clinical note:
{note['note']}

Fields:

{questions}
"""

    if settings["few_shot"]:
        common = few_shot_block() + "\n\nNow extract from the target note.\n\n" + common

    if not settings["json"]:
        ordered_field_ids = ", ".join(field_ids)
        return (
            common
            + f"""
Return plain text only.

You must output exactly one line for every requested field id below, in this exact order:
{ordered_field_ids}

Use exactly this format for every line:
Field <field_id>: <answer>

Do not stop after the first answer. If an answer is not found, still include the line and use N/A."""
        )

    keys = SFT_OUTPUT_FIELDS[:8] + settings["extra_fields"]
    field_guidance = []
    if "evidence_from_text" in settings["extra_fields"]:
        field_guidance.append(
            '- evidence_from_text must be a concise quote or paraphrase from the note. Use "" for N/A answers.'
        )
    if "reasoning_trace" in settings["extra_fields"]:
        field_guidance.append(
            '- reasoning_trace must be a brief visible rationale, not hidden chain-of-thought. Use "" for N/A answers.'
        )

    return (
        common
        + f"""
Return a JSON array with exactly one object per field. Each object must have these keys:
{json.dumps(keys)}

Rules:
- Use the exact record_id, record_id_type, date, field_id, field_category, field_name, and field_instructions supplied in the prompt.
- Put the extracted answer in answers. Use "N/A" if the answer is not found.
- For check-all-that-apply fields, answers may be an array of selected options.
{chr(10).join(field_guidance)}
- Do not add any extra text outside the JSON array."""
    )


def build_synthesis_prompt(note, field_ids, question_metadata, mode, chunk_predictions):
    settings = mode_settings(mode)
    questions = build_questions_block(field_ids, question_metadata)
    compact_predictions = [
        {
            "chunk_index": prediction.get("chunk_index"),
            "field_id": prediction.get("field_id"),
            "field_name": prediction.get("field_name"),
            "answers": prediction.get("answers", "N/A"),
            "evidence_from_text": prediction.get("evidence_from_text", ""),
            "parse_error": prediction.get("parse_error", ""),
        }
        for prediction in chunk_predictions
    ]
    common = f"""Record:
{note['record_id_type']}: {note['record_id']}
date: {note['date']}

Fields:

{questions}

Chunk-level extractions:
{json.dumps(compact_predictions, indent=2, ensure_ascii=False)}
"""

    if not settings["json"]:
        ordered_field_ids = ", ".join(field_ids)
        return (
            common
            + f"""
Synthesize one final answer per field using the chunk-level extractions.
Prefer supported non-empty answers over N/A. Use N/A only when no chunk supports
an answer.

Return plain text only.

You must output exactly one line for every requested field id below, in this exact order:
{ordered_field_ids}

Use exactly this format for every line:
Field <field_id>: <answer>

Do not stop after the first answer. If an answer is not found, still include the line and use N/A."""
        )

    keys = SFT_OUTPUT_FIELDS[:8] + settings["extra_fields"]
    field_guidance = []
    if "evidence_from_text" in settings["extra_fields"]:
        field_guidance.append(
            '- evidence_from_text must summarize the best supporting chunk evidence. Use "" for N/A answers.'
        )
    if "reasoning_trace" in settings["extra_fields"]:
        field_guidance.append(
            '- reasoning_trace must be a brief visible rationale, not hidden chain-of-thought. Use "" for N/A answers.'
        )

    return (
        common
        + f"""
Synthesize one final answer per field using the chunk-level extractions.
Prefer supported non-empty answers over N/A. Use N/A only when no chunk supports
an answer. Resolve conflicts conservatively using the most specific supported
evidence.

Return a JSON array with exactly one object per field. Each object must have these keys:
{json.dumps(keys)}

Rules:
- Use the exact record_id, record_id_type, date, field_id, field_category, field_name, and field_instructions supplied in the prompt.
- Put the final synthesized answer in answers.
{chr(10).join(field_guidance)}
- Do not add any extra text outside the JSON array."""
    )
