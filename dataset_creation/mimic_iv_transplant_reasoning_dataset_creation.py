import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import ast
import os
from openai import OpenAI
from google import genai
from dotenv import load_dotenv
import sys
import argparse
import time


load_dotenv()


AVAILABLE_MODELS = {
    "gpt-4o": "gpt-4o-2024-11-20",
    "gpt-o3": "o3-2025-04-16",
    "gpt-o4-mini": "o4-mini-2025-04-16",
    "gemini":"gemini-2.5-flash"
}

class LLMConfig:
    def __init__(self, model_key: str):
        if model_key not in AVAILABLE_MODELS:
            raise ValueError(f"Model {model_key} not in available models: {list(AVAILABLE_MODELS.keys())}")
        
        self.model_key = model_key  # Key from AVAILABLE_MODELS (e.g., "gpt-4o", "gemini")
        self.model_name = AVAILABLE_MODELS[model_key]  # Actual model string for API (e.g., "gpt-4o-2024-11-20")
        self.provider = "gemini" if model_key == "gemini" else "openai"  # API provider
        
        if self.provider == "gemini":
            self.client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
        else:
            self.client = OpenAI(api_key=os.getenv('OAI_API_KEY'))
        
    def generate_response(self, prompt: str) -> str:
        try:
            if self.provider == "gemini": 
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                return response.text
            else:  # OpenAI
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return None

def load_progress_notes(input_file: str = 'results/mimic_iv_initial_transplant_dataset/hct_initial_progression_notes.csv'):
    """Load progress notes with error handling."""
    try:
        if not os.path.exists(input_file):
            print(f"Error: Input file not found: {input_file}")
            sys.exit(1)
        return pd.read_csv(input_file)
    except Exception as e:
        print(f"Error loading progress notes from {input_file}: {str(e)}")
        sys.exit(1)

def process_notes_with_reasoning(llm_config: LLMConfig, output_dir: str, num_rows=None, input_file: str = None, output_filename: str = None):
    """
    Process progress notes with LLM reasoning.
    
    Args:
        llm_config (LLMConfig): Configuration for LLM model (OpenAI or Gemini)
        output_dir (str): Directory to save output files
        num_rows (int, optional): Number of rows to process. If None, process all rows.
        input_file (str, optional): Path to input CSV file. If None, uses default.
        output_filename (str, optional): Custom output filename. If None, generates automatically.
    """
    
    # Load progress notes
    if input_file:
        progress_notes = load_progress_notes(input_file)
    else:
        progress_notes = load_progress_notes()
    
    # Determine how many rows to process
    if num_rows is not None:
        rows_to_process = progress_notes.head(num_rows)
        print(f"Processing {num_rows} rows out of {len(progress_notes)} total rows")
    else:
        rows_to_process = progress_notes
        print(f"Processing all {len(progress_notes)} rows")

    print(f"Using {llm_config.provider.upper()} model: {llm_config.model_name}")

    # Create list to store all medication-reasoning pairs
    reasoning_records = []
    successful_count = 0
    failed_count = 0
    total_medications = 0

    for idx, row in rows_to_process.iterrows():
        try:
            assessment_plan = row['assess_plan']
            if pd.isna(assessment_plan) or assessment_plan is None:
                failed_count += 1
                continue
            
            # Parse newly_started_meds safely
            newly_started_meds = row['newly_started_meds']
            if pd.isna(newly_started_meds) or newly_started_meds == '' or newly_started_meds == '[]':
                failed_count += 1
                continue
            
            # Handle string representation of lists
            if isinstance(newly_started_meds, str):
                try:
                    if newly_started_meds.startswith('[') and newly_started_meds.endswith(']'):
                        med_list = ast.literal_eval(newly_started_meds)
                    else:
                        med_list = [newly_started_meds]
                except:
                    med_list = [newly_started_meds]
            else:
                med_list = newly_started_meds
            
            # Process every medication in the list
            if len(med_list) > 0:
                for med_idx, med in enumerate(med_list):
                    total_medications += 1
                    
                    cot_prompt = f"""
                        You are provided with the following:
                        1. A progress note in a certain day of a hosptial stay for allogeneic stem cell transplant or related complications
                        2. The Assessment and Plan section of the note
                        3. A specific medication that was started on the same day as the note
                        Your task is to reason step-by-step like the treating physician that lead to the final conclusion that this medication needs to be started.

                        Guidelines for Reasoning:

                        1. Use a clinical reasoning tone (e.g. Let's summarize the clinical evidence...Okay, let's reconsider...Oh, wait a second, maybe we're missing something simpler...). Some aspects to consider: conditioing regimen and GVHD prophylaxis for transplant, current graft function, underlying hematological disease, infectious disease and immune reconstitution, GVHD, other transplant related complications, etc. As you are reasoning to the conclusion (the medication), the medication itself should not occur in the reasoning process.

                        2.  Summarize the clinical course and intergrate clinical data (history, lab results, imaging findings, current medications, recent medications etc. MENTION SPECIFIC CLINICAL DATA!) in your reasoning. 

                        3. Review the Assessment and Plan section.
                            - It may explicitly state the reasoning for the medication.
                            - However, do not assume the reasoning is correct, assess it critically.
                            - DO NOT CITE ANYTHING FROM THE ASSESSMENT AND PLAN SECTION, but use it as if you come up with the reasoning yourself.
                        
                        4. Use this as the final sentence: Therefore, due to (summary of reasons), {med} was started today.

                        ### Input Format:
                        Progress Note:
                        {row['progression_prompt']}

                        Assessment and Plan:
                        {assessment_plan}

                        Medication Started:
                        {med}

                        ### Output Format:
                        Reasoning (Step-by-step clinical reasoning process):
                        """
                            
                    reasoning = llm_config.generate_response(cot_prompt)
                    
                    # Create a new record for each medication
                    new_record = row.copy()
                    new_record['target_medication'] = med
                    new_record['medication_index'] = med_idx
                    
                    if reasoning is None:
                        new_record['llm_reasoning'] = None
                        new_record['processing_status'] = 'api_error'
                        failed_count += 1
                    else:
                        new_record['llm_reasoning'] = reasoning
                        new_record['processing_status'] = 'success'
                        successful_count += 1
                    
                    reasoning_records.append(new_record)
                    
                    # Print progress every 10 medications
                    if total_medications % 10 == 0:
                        print(f"Processed {total_medications} medications from {idx - rows_to_process.index[0] + 1} records...")
            else:
                failed_count += 1
                    
        except Exception as e:
            print(f"Error processing hadm_id {row['hadm_id']}: {str(e)}")
            failed_count += 1

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert list of records to DataFrame
    if reasoning_records:
        reasoning_dataset = pd.DataFrame(reasoning_records)
    else:
        print("No reasoning records generated!")
        return None
    
    # Save the reasoning dataset
    if output_filename:
        # Use custom filename, add directory if not already included
        if not os.path.dirname(output_filename):
            final_output_path = os.path.join(output_dir, output_filename)
        else:
            final_output_path = output_filename
    else:
        # Generate default filename
        final_output_path = os.path.join(output_dir, f'bmt_initial_progression_notes_with_reasoning{"_" + str(num_rows) if num_rows else "" + "_" + llm_config.model_key}.csv')
    
    try:
        reasoning_dataset.to_csv(final_output_path, index=False)
        print("\nReasoning dataset saved successfully!")
        print(f"Total original records processed: {len(rows_to_process)}")
        print(f"Total medications processed: {total_medications}")
        print(f"Successfully processed medications: {successful_count}")
        print(f"Failed to process medications: {failed_count}")
        print(f"Final dataset rows: {len(reasoning_dataset)}")
        print(f"Records with reasoning: {reasoning_dataset['llm_reasoning'].notna().sum()}")
        print(f"Output saved to: {final_output_path}")
    except Exception as e:
        print(f"Error saving dataset to {final_output_path}: {str(e)}")
        sys.exit(1)
    
    return reasoning_dataset

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Process progress notes with LLM reasoning (OpenAI or Gemini) and create a reasoning dataset.'
    )
    parser.add_argument(
        '--num-rows', 
        type=int,
        default=None,
        help='Number of rows to process. If not specified, processes all rows.'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        choices=list(AVAILABLE_MODELS.keys()),
        default='gemini',
        help='LLM model to use (OpenAI or Gemini)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/datasets',
        help='Directory to save output files'
    )
    parser.add_argument(
        '--input-file',
        type=str,
        default=None,
        help='Path to input CSV file containing progress notes. If not specified, uses default path.'
    )
    parser.add_argument(
        '--output-filename',
        type=str,
        default=None,
        help='Custom output filename (with .csv extension). If not specified, generates automatically based on model and parameters.'
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    llm_config = LLMConfig(model_key=args.model_name)
    reasoning_dataset = process_notes_with_reasoning(
        llm_config=llm_config,
        output_dir=args.output_dir,
        num_rows=args.num_rows,
        input_file=args.input_file,
        output_filename=args.output_filename
    )

