import argparse
import csv
import json
import os
import re
import time
from datetime import datetime

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv:
    load_dotenv()


def read_csv_records(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def normalize_question_number(value):
    normalized = str(value).strip()
    match = re.fullmatch(r"Question\s+(.+)", normalized, flags=re.IGNORECASE)
    return match.group(1).strip() if match else normalized


def question_number_column(rows):
    if not rows:
        raise ValueError("Question CSV is empty.")
    for column in rows[0].keys():
        if column.lstrip("\ufeff") == "Question Number":
            return column
    raise KeyError("Could not find Question Number column in questions CSV.")


def load_question_metadata(question_rows):
    q_col = question_number_column(question_rows)
    metadata = {}

    for row in question_rows:
        field_id = normalize_question_number(row[q_col])
        metadata[field_id] = {
            "field_id": field_id,
            "field_category": str(row.get("category", "") or "").strip(),
            "field_name": str(row.get("Question Text", "") or "").strip(),
            "field_instructions": str(row.get("Options / Instructions", "") or "").strip(),
        }

    return metadata


def parse_date(value):
    value = str(value or "").strip()
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    return None


def group_references_by_mrn(reference_rows):
    grouped = {}
    for row in reference_rows:
        mrn = str(row.get("MRN", "") or "").strip()
        if not mrn:
            continue
        grouped.setdefault(mrn, []).append(row)
    return grouped


def choose_reference_row(mrn, admit_date, references_by_mrn):
    candidates = references_by_mrn.get(str(mrn), [])
    if not candidates:
        return None

    admit = parse_date(admit_date)
    if not admit:
        return candidates[0]

    def distance(row):
        bmt_date = parse_date(row.get("BMT_date"))
        if not bmt_date:
            return 10**9
        return abs((bmt_date - admit).days)

    return min(candidates, key=distance)


def compact_reference(row, max_chars):
    if not row:
        return "No matching final_merged.csv reference row was found."

    pairs = []
    for key, value in row.items():
        value = str(value or "").strip()
        if not value:
            continue
        pairs.append(f"{key}: {value}")

    text = "\n".join(pairs)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit("\n", 1)[0] + "\n[reference truncated]"


def chunks(values, size):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def build_questions_prompt(field_ids, question_metadata):
    questions = []
    for field_id in field_ids:
        question = question_metadata[field_id]
        q_str = (
            f"Question {field_id}\n"
            f"Category: {question['field_category']}\n"
            f"Field name: {question['field_name']}"
        )
        if question["field_instructions"] and question["field_instructions"] != "[]":
            q_str += f"\nOptions/Instructions: {question['field_instructions']}"
        questions.append(q_str)

    return "\n\n".join(questions)


def build_prompt(mrn, admit_date, note_text, reference_text, field_ids, question_metadata):
    questions_prompt = build_questions_prompt(field_ids, question_metadata)
    return f"""You are extracting CIBMTR Form 2400 fields from a BMT history and physical note.

Record identifiers:
MRN: {mrn}
admit_date: {admit_date}

History and physical note:
{note_text}

Reference data from final_merged.csv:
{reference_text}

Questions to answer:

{questions_prompt}

Return a JSON array with exactly one object per question above. Each object must have these keys:
- MRN
- admit_date
- field_id
- field_category
- field_name
- field_instructions
- answers
- evidence_from_text
- reasoning_trace

Rules:
- Use the exact MRN, admit_date, field_id, field_category, field_name, and field_instructions supplied in the prompt.
- Prefer evidence from the H&P note. Use final_merged.csv reference data only when the note is missing, abbreviated, or ambiguous.
- Put the extracted answer in answers. Use "N/A" if neither the note nor the reference data supports an answer.
- For check-all-that-apply fields, answers may be an array of selected options.
- evidence_from_text must be a concise quote or paraphrase from the note or reference data. Use "" for N/A answers.
- reasoning_trace must be a brief explanation of how the evidence maps to the answer. Do not include hidden chain-of-thought. Use "" for N/A answers.
- Do not add any extra text outside the JSON array.

Useful transplant abbreviations in the reference data:
- Gn: recipient sex; donGn: donor sex.
- Tx_Type, donRel, HLA, Source_x, Prep, AB, gvhdpr, donage, donabo, doncmv, CMV, aborh may map to donor, graft, conditioning, GVHD prophylaxis, and serology fields.
- PB usually means peripheral blood stem cells.
- FluBu means fludarabine/busulfan; FluBuATG includes ATG.
- FK usually means tacrolimus; MTX means methotrexate.
"""


def parse_answer_json(answer_text):
    parsed = json.loads(answer_text)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        if isinstance(parsed.get("records"), list):
            return parsed["records"]
        if isinstance(parsed.get("answers"), list):
            return parsed["answers"]
    raise ValueError("Response JSON must be an array of per-question records.")


def coerce_record(record, mrn, admit_date, field_id, question_metadata):
    metadata = question_metadata[field_id]
    return {
        "MRN": str(mrn),
        "admit_date": str(admit_date),
        "field_id": field_id,
        "field_category": record.get("field_category") or metadata["field_category"],
        "field_name": record.get("field_name") or metadata["field_name"],
        "field_instructions": record.get("field_instructions") or metadata["field_instructions"],
        "answers": record.get("answers", "N/A"),
        "evidence_from_text": record.get("evidence_from_text", ""),
        "reasoning_trace": record.get("reasoning_trace", ""),
    }


def load_existing_results(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Existing output is not a JSON array: {path}")
    return data


def completed_keys(records):
    keys = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        mrn = str(record.get("MRN", ""))
        admit_date = str(record.get("admit_date", ""))
        field_id = normalize_question_number(record.get("field_id", ""))
        if mrn and field_id:
            keys.add((mrn, admit_date, field_id))
    return keys


def write_json(path, records):
    with open(path, "w") as f:
        json.dump(records, f, indent=4)


def generate_content(client, types, model, prompt):
    return client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(include_thoughts=True),
        ),
    )


def split_response_parts(response):
    thoughts_text = ""
    answer_text = ""
    if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if not part.text:
                continue
            if getattr(part, "thought", False):
                thoughts_text += part.text
            else:
                answer_text += part.text
    else:
        answer_text = response.text
    return answer_text, thoughts_text


def write_thoughts_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["MRN", "admit_date", "field_ids", "thoughts"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Extract all CIBMTR Form 2400 fields from H&P notes using Gemini."
    )
    parser.add_argument("--notes", default="data/h_p_notes.csv", help="Input H&P notes CSV")
    parser.add_argument(
        "--questions",
        default="resources/CIBMTR_Form_2400_R11_Items_simplified.csv",
        help="CIBMTR question metadata CSV",
    )
    parser.add_argument("--reference", default="data/final_merged.csv", help="Reference CSV keyed by MRN")
    parser.add_argument(
        "--output",
        default="data/h_p_form_2400_all_extractions.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--thoughts_output",
        default="data/h_p_form_2400_all_thoughts.csv",
        help="Output thoughts CSV path",
    )
    parser.add_argument("--model", default="gemini-3-flash-preview", help="Gemini model to use")
    parser.add_argument("--limit", type=int, default=None, help="Number of H&P notes to process")
    parser.add_argument("--batch_size", type=int, default=43, help="Number of questions per model request")
    parser.add_argument(
        "--reference_max_chars",
        type=int,
        default=6000,
        help="Maximum characters of final_merged.csv reference data to include per note",
    )
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds to sleep between model requests")
    parser.add_argument("--resume", action="store_true", help="Append to an existing output and skip completed fields")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        print("Please run: export GEMINI_API_KEY='your_api_key'")
        return

    if args.batch_size < 1:
        parser.error("--batch_size must be at least 1")

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("Error: google-genai is not installed.")
        print("Please run: pip install google-genai")
        return

    print("Loading input files...")
    notes = read_csv_records(args.notes)
    questions = read_csv_records(args.questions)
    references = read_csv_records(args.reference)

    if args.limit:
        notes = notes[: args.limit]
        print(f"Limiting to first {args.limit} notes.")

    question_metadata = load_question_metadata(questions)
    field_ids = list(question_metadata.keys())
    references_by_mrn = group_references_by_mrn(references)

    results = load_existing_results(args.output) if args.resume else []
    done = completed_keys(results)
    thoughts_rows = []

    client = genai.Client(api_key=api_key)

    total_batches = sum(
        1
        for note in notes
        for batch in chunks(field_ids, args.batch_size)
        if not all((str(note.get("MRN", "")), str(note.get("admit_date", "")), field_id) in done for field_id in batch)
    )
    completed_batches = 0

    for note in notes:
        mrn = str(note.get("MRN", "") or "").strip()
        admit_date = str(note.get("admit_date", "") or "").strip()
        note_text = str(note.get("note", "") or "").strip()
        reference_row = choose_reference_row(mrn, admit_date, references_by_mrn)
        reference_text = compact_reference(reference_row, args.reference_max_chars)

        for field_batch in chunks(field_ids, args.batch_size):
            batch_keys = [(mrn, admit_date, field_id) for field_id in field_batch]
            if args.resume and all(key in done for key in batch_keys):
                continue

            completed_batches += 1
            print(
                f"Processing MRN {mrn}, admit_date {admit_date}, "
                f"batch {completed_batches}/{total_batches}: {field_batch[0]}-{field_batch[-1]}"
            )

            prompt = build_prompt(
                mrn,
                admit_date,
                note_text,
                reference_text,
                field_batch,
                question_metadata,
            )

            try:
                response = generate_content(client, types, args.model, prompt)
                answer_text, thoughts_text = split_response_parts(response)
                raw_records = parse_answer_json(answer_text)
                raw_by_field_id = {
                    normalize_question_number(record.get("field_id", "")): record
                    for record in raw_records
                    if isinstance(record, dict)
                }

                for field_id in field_batch:
                    if args.resume and (mrn, admit_date, field_id) in done:
                        continue
                    raw_record = raw_by_field_id.get(field_id, {})
                    coerced = coerce_record(raw_record, mrn, admit_date, field_id, question_metadata)
                    results.append(coerced)
                    done.add((mrn, admit_date, field_id))

                thoughts_rows.append(
                    {
                        "MRN": mrn,
                        "admit_date": admit_date,
                        "field_ids": ",".join(field_batch),
                        "thoughts": thoughts_text,
                    }
                )
            except Exception as e:
                print(f"Error processing MRN {mrn}, batch {field_batch[0]}-{field_batch[-1]}: {e}")
                for field_id in field_batch:
                    if args.resume and (mrn, admit_date, field_id) in done:
                        continue
                    results.append(
                        {
                            "MRN": mrn,
                            "admit_date": admit_date,
                            **question_metadata[field_id],
                            "answers": "N/A",
                            "evidence_from_text": "",
                            "reasoning_trace": "",
                            "error": str(e),
                        }
                    )
                    done.add((mrn, admit_date, field_id))

            write_json(args.output, results)
            write_thoughts_csv(args.thoughts_output, thoughts_rows)
            time.sleep(args.sleep)

    print(f"\nFinished. Results saved to {args.output} and {args.thoughts_output}")


if __name__ == "__main__":
    main()
