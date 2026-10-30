# furhter filter to only include HCT initial admissions (admitted for transplant); plus more meds exclusion

import pandas as pd
import numpy as np
import ast
import argparse
import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Optional

def remove_excluded_meds(med_list: List[str], exclusion_keywords: List[str]) -> List[str]:
    """
    Remove excluded medications from a medication list
    """
    if pd.isna(med_list):
        return []
    
    # Convert string representation of list to actual list
    if isinstance(med_list, str):
        try:
            med_list = ast.literal_eval(med_list)
        except:
            med_list = [med_list]
    
    # Keep only medications that don't contain exclusion keywords
    return [med for med in med_list 
            if not any(keyword.lower() in str(med).lower() for keyword in exclusion_keywords)]

def filter_medications(progress_note: pd.DataFrame, admission_note: pd.DataFrame, 
                      exclusion_keywords: List[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter out excluded medications and remove rows with empty medication lists
    """
    print("Filtering medications...")
    print(f"Original progression notes: {len(progress_note)}")
    print(f"Original admission notes: {len(admission_note)}")
    
    # Remove excluded medications
    progress_note['newly_started_meds'] = progress_note['newly_started_meds'].apply(
        lambda x: remove_excluded_meds(x, exclusion_keywords)
    )
    admission_note['admission_med'] = admission_note['admission_med'].apply(
        lambda x: remove_excluded_meds(x, exclusion_keywords)
    )
    
    # Remove rows with empty medication lists
    progress_note = progress_note[progress_note['newly_started_meds'].apply(len) > 0]
    admission_note = admission_note[admission_note['admission_med'].apply(len) > 0]
    
    print(f"After filtering progression notes: {len(progress_note)}")
    print(f"After filtering admission notes: {len(admission_note)}")
    
    return progress_note, admission_note

def screen_with_openai(admission_note: pd.DataFrame, start_row: int = 0, 
                      end_row: Optional[int] = None, 
                      api_key: Optional[str] = None) -> List[int]:
    """
    Screen admission notes with OpenAI to identify transplant-related admissions
    """
    if api_key is None:
        load_dotenv()
        api_key = os.getenv('OAI_API_KEY')
    
    if not api_key:
        raise ValueError("OpenAI API key not found. Set OAI_API_KEY environment variable or pass api_key parameter.")
    
    client = OpenAI(api_key=api_key)
    
    # Determine range of rows to process
    if end_row is None:
        end_row = len(admission_note)
    
    end_row = min(end_row, len(admission_note))
    rows_to_process = admission_note.iloc[start_row:end_row]
    
    print(f"Screening rows {start_row} to {end_row-1} with OpenAI...")
    print(f"Processing {len(rows_to_process)} admission notes...")
    
    transplant_admission_hadm_ids = []
    
    for i, (idx, row) in enumerate(rows_to_process.iterrows()):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-2024-11-20",
                messages=[
                    {"role": "user", 
                     "content": f"Is the admission reason for the following clinical note mainly related to the INITIAL ADMISSION for stem cell transplant (i.e. patient was admitted for transplant)? Return only 'YES' or 'NO'.\n\n{row['admission_prompt']}"}
                ]
            )
            
            if response.choices[0].message.content.strip().upper() == 'YES':
                transplant_admission_hadm_ids.append(row['hadm_id'])
                
        except Exception as e:
            print(f"Error processing row {start_row + i} (hadm_id {row['hadm_id']}): {str(e)}")
            continue
    
    print(f"Found {len(transplant_admission_hadm_ids)} transplant-related admissions")
    return transplant_admission_hadm_ids

def main():
    parser = argparse.ArgumentParser(description='Optimize transplant dataset with medication filtering and OpenAI screening')
    parser.add_argument('--data_dir', type=str, default='results/mimic_iv_transplant_dataset',
                       help='Directory containing transplant dataset CSV files')
    parser.add_argument('--start_row', type=int, default=0,
                       help='Starting row for OpenAI screening (0-based index)')
    parser.add_argument('--end_row', type=int, default=None,
                       help='Ending row for OpenAI screening (exclusive). If not provided, processes all rows.')
    parser.add_argument('--output_dir', type=str, default='results/mimic_iv_initial_transplant_dataset',
                       help='Output directory for optimized dataset')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading transplant dataset...")
    progress_note = pd.read_csv(f'{args.data_dir}/transplant_progression_notes.csv')
    admission_note = pd.read_csv(f'{args.data_dir}/transplant_admission_notes.csv')
    
    print(f"Loaded {len(progress_note)} progression notes and {len(admission_note)} admission notes")
    
    # Step 1: Filter medications
    med_exclusion = [
        'replacement', 'powder', 'nasal', 'patch', 'alteplase', 'calcium gluconate', 'dextrose 50%', 'phosphorus', 'ointment',
        'potassium chloride', 'hydromorphone', 'oxycodone', 'morphine', 'gabapentin', 'fentanyl', 'magnesium oxide', 
        'multivitamin', 'neutra-phos', 'dakins', 'artificial tears', 'senna', 'folic acid', 'lorazepam', 'calcium carbonate', 'potassium phosphate',
        'zolpidem', 'omeprazole', 'pantoprazole', 'magnesium sulfate', 'vitamin', 'heparin', 'ondansetron', 'bisacodyl', 'calcium gluconate',
        'artificial tear', 'trazadone', 'insulin', 'glucagon', '__', 'diphenhydramine', 'epinephrine', 'dental',
        'polyethylene glycol', 'magneisum citrate', 'polyethylene glycol', 'cepacol', 'glucose gel', 'lidocaine', 'sodium chloride'
    ]
    
    progress_note, admission_note = filter_medications(progress_note, admission_note, med_exclusion)
    
    # Step 2: OpenAI screening
    transplant_admission_hadm_ids = screen_with_openai(
        admission_note, 
        start_row=args.start_row,
        end_row=args.end_row
    )
    
    # Filter both datasets to only include transplant-related admissions
    print("Filtering datasets to transplant-related admissions...")
    original_admission_count = len(admission_note)
    original_progress_count = len(progress_note)
    
    admission_note = admission_note[admission_note['hadm_id'].isin(transplant_admission_hadm_ids)]
    progress_note = progress_note[progress_note['hadm_id'].isin(transplant_admission_hadm_ids)]
    
    print(f"Admission notes: {original_admission_count} → {len(admission_note)}")
    print(f"Progression notes: {original_progress_count} → {len(progress_note)}")
    
    # Save optimized datasets
    print(f"Saving optimized datasets to {args.output_dir}")
    admission_note.to_csv(f'{args.output_dir}/hct_initial_admission_notes.csv', index=False)
    progress_note.to_csv(f'{args.output_dir}/hct_initial_progression_notes.csv', index=False)
    
    print("Optimization complete!")
    print(f"Final counts:")
    print(f"  Admission notes: {len(admission_note)}")
    print(f"  Progression notes: {len(progress_note)}")

if __name__ == "__main__":
    main() 