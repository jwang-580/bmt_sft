import argparse
import csv
import hashlib
import json
import os
import random
import re
import time

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv:
    load_dotenv()


EMPTY_ANSWER_STRINGS = {"", "N/A", "NA", "NONE", "NULL"}


def clean_note_text(row):
    prompt_text = str(row["progression_prompt"])
    assess_plan = str(row.get("assess_plan", "") or "")

    prompt_text = prompt_text.replace(
        "What medication should be started for this patient today based on the above information?",
        "",
    ).strip()
    prompt_text = re.sub(r"[^.!?]*\?\s*$", "", prompt_text).strip()

    if assess_plan and assess_plan.lower() != "nan":
        prompt_text += f"\n\nASSESSMENT AND PLAN:\n{assess_plan}"

    return prompt_text


def normalize_question_number(value):
    normalized = str(value).strip()
    match = re.fullmatch(r"Question\s+(.+)", normalized, flags=re.IGNORECASE)
    return match.group(1).strip() if match else normalized


def question_number_column(questions_df):
    for column in questions_df[0].keys():
        if column.lstrip("\ufeff") == "Question Number":
            return column
    raise KeyError("Could not find Question Number column in questions CSV.")


def load_question_metadata(questions_df):
    q_col = question_number_column(questions_df)
    metadata = {}

    for row in questions_df:
        field_id = normalize_question_number(row[q_col])
        metadata[field_id] = {
            "field_id": field_id,
            "field_category": str(row.get("category", "") or "").strip(),
            "field_name": str(row.get("Question Text", "") or "").strip(),
            "field_instructions": str(row.get("Options / Instructions", "") or "").strip(),
        }

    return metadata


def read_csv_records(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def deduplicate_notes(notes):
    deduped = []
    seen_hadm_ids = set()
    for row in notes:
        hadm_id = row.get("hadm_id")
        if hadm_id in seen_hadm_ids:
            continue
        seen_hadm_ids.add(hadm_id)
        deduped.append(row)
    return deduped


def write_thoughts_csv(path, thoughts_results):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["hadm_id", "thoughts"])
        writer.writeheader()
        writer.writerows(thoughts_results)


def is_empty_answer(value):
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().upper() in EMPTY_ANSWER_STRINGS
    if isinstance(value, list):
        return not value or all(is_empty_answer(item) for item in value)
    if isinstance(value, dict):
        return not value
    return False


def deterministic_rng(hadm_id):
    digest = hashlib.sha256(str(hadm_id).encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def load_previous_answers(path):
    if not path or not os.path.exists(path):
        return {}

    with open(path) as f:
        records = json.load(f)

    previous_answers = {}
    for record in records:
        answers = record.get("answers") if isinstance(record, dict) else None
        if not isinstance(answers, dict):
            continue
        previous_answers[str(record.get("hadm_id"))] = {
            normalize_question_number(k): v for k, v in answers.items()
        }

    return previous_answers


def sample_question_ids(hadm_id, question_ids, previous_answers, total=10, non_empty_target=8):
    answers = previous_answers.get(str(hadm_id), {})
    non_empty_ids = [qid for qid in question_ids if qid in answers and not is_empty_answer(answers[qid])]
    empty_ids = [qid for qid in question_ids if qid in answers and is_empty_answer(answers[qid])]
    unknown_ids = [qid for qid in question_ids if qid not in answers]

    rng = deterministic_rng(hadm_id)
    rng.shuffle(non_empty_ids)
    rng.shuffle(empty_ids)
    rng.shuffle(unknown_ids)

    non_empty_count = min(non_empty_target, total, len(non_empty_ids))
    empty_count = min(total - non_empty_count, len(empty_ids))

    selected = non_empty_ids[:non_empty_count] + empty_ids[:empty_count]

    if len(selected) < total:
        selected.extend(non_empty_ids[non_empty_count : non_empty_count + total - len(selected)])
    if len(selected) < total:
        selected.extend(empty_ids[empty_count : empty_count + total - len(selected)])
    if len(selected) < total:
        selected.extend(unknown_ids[: total - len(selected)])

    rng.shuffle(selected)
    return selected[:total]


def build_questions_prompt(selected_ids, question_metadata):
    questions = []
    for field_id in selected_ids:
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


def build_prompt(hadm_id, prompt_text, selected_ids, question_metadata):
    questions_prompt = build_questions_prompt(selected_ids, question_metadata)
    return f"""You are reading a BMT clinical note and extracting CIBMTR Form 2400 fields.

hadm_id: {int(hadm_id)}

Clinical Text:
{prompt_text}

Questions to answer based only on the clinical text:

{questions_prompt}

Return a JSON array with exactly one object per question above. Each object must have these keys:
- hadm_id
- field_id
- field_category
- field_name
- field_instructions
- answers
- evidence_from_text
- reasoning_trace

Rules:
- Use the exact hadm_id, field_id, field_category, field_name, and field_instructions supplied in the prompt.
- Put the extracted answer in answers. Use "N/A" if the answer is not found.
- For check-all-that-apply fields, answers may be an array of selected options.
- evidence_from_text must be a concise quote or paraphrase from the clinical text supporting the answer. Use "" for N/A answers.
- reasoning_trace include your full chain-of-thought for this field. Use "" for N/A answers.
- Do not add any extra text outside the JSON array.

Typical GVHD prophylaxis regimens include MTX+calcineurin inhibitors (CI) or PTCy-based (+tacro/MMF). Sometimes notes may only mention a CI but not MTX; in that context, MTX is most likely combined with a CI because CI alone is not enough for GVHD prophylaxis."""


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


def coerce_record(record, hadm_id, question_metadata):
    field_id = normalize_question_number(record.get("field_id", ""))
    metadata = question_metadata.get(field_id, {})
    return {
        "hadm_id": int(hadm_id),
        "field_id": field_id,
        "field_category": record.get("field_category") or metadata.get("field_category", ""),
        "field_name": record.get("field_name") or metadata.get("field_name", ""),
        "field_instructions": record.get("field_instructions")
        or metadata.get("field_instructions", ""),
        "answers": record.get("answers", "N/A"),
        "evidence_from_text": record.get("evidence_from_text", ""),
        "reasoning_trace": record.get("reasoning_trace", ""),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract sampled CIBMTR Form 2400 fields from clinical notes using Gemini."
    )
    parser.add_argument("--limit", type=int, default=None, help="Number of records to process (default: all)")
    parser.add_argument(
        "--output",
        type=str,
        default="data/bmt_form_2400_sampled_extractions.json",
        help="Path to save output JSON",
    )
    parser.add_argument(
        "--thoughts_output",
        type=str,
        default="data/bmt_form_2400_thoughts.csv",
        help="Path to save output thoughts CSV",
    )
    parser.add_argument(
        "--previous_answers",
        type=str,
        default="data/bmt_form_2400_answers.json",
        help="Existing answers JSON used to sample roughly 8 non-empty and 2 empty fields per note",
    )
    parser.add_argument("--questions_per_note", type=int, default=10, help="Number of questions to extract per note")
    parser.add_argument(
        "--non_empty_per_note",
        type=int,
        default=8,
        help="Target number of previously non-empty questions among the sampled questions",
    )
    parser.add_argument("--model", type=str, default="gemini-3.1-pro-preview", help="Gemini model to use")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        print("Please run: export GEMINI_API_KEY='your_api_key'")
        return

    if args.non_empty_per_note > args.questions_per_note:
        parser.error("--non_empty_per_note cannot be greater than --questions_per_note")

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("Error: google-genai is not installed.")
        print("Please run: pip install google-genai")
        return

    client = genai.Client(api_key=api_key)

    print("Loading data...")
    try:
        notes = read_csv_records("data/bmt_initial_progression_notes_all.csv")
        questions = read_csv_records("resources/CIBMTR_Form_2400_R11_Items_simplified.csv")
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    question_metadata = load_question_metadata(questions)
    question_ids = list(question_metadata.keys())
    previous_answers = load_previous_answers(args.previous_answers)

    initial_count = len(notes)
    notes = deduplicate_notes(notes)
    print(f"Deduplicated from {initial_count} to {len(notes)} unique hadm_ids.")

    if args.limit:
        notes = notes[: args.limit]
        print(f"Limiting to first {args.limit} records.")

    results = []
    thoughts_results = []

    for row in notes:
        hadm_id = row["hadm_id"]
        prompt_text = clean_note_text(row)
        selected_ids = sample_question_ids(
            hadm_id,
            question_ids,
            previous_answers,
            total=args.questions_per_note,
            non_empty_target=args.non_empty_per_note,
        )
        full_prompt = build_prompt(hadm_id, prompt_text, selected_ids, question_metadata)

        print(
            f"Processing hadm_id: {hadm_id} "
            f"({len(thoughts_results) + 1}/{len(notes)}), questions: {', '.join(selected_ids)}"
        )

        try:
            response = client.models.generate_content(
                model=args.model,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(include_thoughts=True),
                ),
            )

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

            try:
                answer_records = parse_answer_json(answer_text)
                records_by_field_id = {
                    normalize_question_number(record.get("field_id", "")): record
                    for record in answer_records
                    if isinstance(record, dict)
                }
                coerced_records = []
                for field_id in selected_ids:
                    raw_record = records_by_field_id.get(field_id, {"field_id": field_id, "answers": "N/A"})
                    coerced_records.append(coerce_record(raw_record, hadm_id, question_metadata))
            except Exception as e:
                print(f"Warning: Could not parse JSON records for {hadm_id}: {e}. Saving raw output.")
                coerced_records = [
                    {
                        "hadm_id": int(hadm_id),
                        "field_id": field_id,
                        **question_metadata[field_id],
                        "answers": "N/A",
                        "evidence_from_text": "",
                        "reasoning_trace": "",
                        "raw_output": answer_text,
                    }
                    for field_id in selected_ids
                ]

            results.extend(coerced_records)
            thoughts_results.append({"hadm_id": int(hadm_id), "thoughts": thoughts_text})

            with open(args.output, "w") as f:
                json.dump(results, f, indent=4)

            write_thoughts_csv(args.thoughts_output, thoughts_results)
            time.sleep(0.5)

        except Exception as e:
            print(f"Error processing hadm_id {hadm_id}: {e}")
            for field_id in selected_ids:
                results.append(
                    {
                        "hadm_id": int(hadm_id),
                        **question_metadata[field_id],
                        "answers": "N/A",
                        "evidence_from_text": "",
                        "reasoning_trace": "",
                        "error": str(e),
                    }
                )

            with open(args.output, "w") as f:
                json.dump(results, f, indent=4)

    print(f"\nFinished processing. Results saved to {args.output} and {args.thoughts_output}")


if __name__ == "__main__":
    main()
