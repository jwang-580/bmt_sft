import pandas as pd
import json
import re
import os
import argparse
import gc
import ast
from typing import List, Dict, Tuple, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict
import sys
from openai import OpenAI
from dotenv import load_dotenv

# Import shared functions from the main MIMIC-IV dataset creation script
from mimic_iv_dataset_creation import (
    clean_text_for_csv,
    create_labs_with_labels_if_needed,
    extract_hpi_and_medications,
    process_patient_data_to_prompts,
    process_medications_data,
    process_imaging_data,
    process_micro_data,
    process_labs_data,
    generate_admission_data,
    generate_progression_data,
    PatientDataProcessor,
    EXCLUDED_MEDICATION_KEYWORDS,
    save_dataframe_to_csv_safely,
    convert_list_fields_to_string,
    extract_drug_name,
    deduplicate_medications_by_clean_name,
    filter_medications_mentioned_in_prompt
)

# Keywords to search for in discharge summaries to identify transplant patients
TRANSPLANT_KEYWORDS = [
    'gvhd',
    'allogeneic stem cell transplant', 
    'allogenic stem cell transplant',
    'hematopoietic stem cell transplant',
    'alloHCT',
    'allo-HCT',
    'allogeneic HCT',
    'allogenic HCT',
    'allogeneic transplant',
    'allogenic transplant',
    'HSCT'
]

# Keywords to exclude (autologous transplant cases)
EXCLUSION_KEYWORDS = [
    'autologous',
    'ASCT',
    'auto-SCT'
]

def contains_transplant_keywords(text: str) -> bool:
    """
    Check if discharge summary contains any transplant-related keywords
    """
    if not text or pd.isna(text):
        return False
    
    text_lower = str(text).lower()
    return any(keyword.lower() in text_lower for keyword in TRANSPLANT_KEYWORDS)

def contains_exclusion_keywords(text: str) -> bool:
    """
    Check if discharge summary contains any exclusion keywords (autologous transplant indicators)
    """
    if not text or pd.isna(text):
        return False
    
    text_lower = str(text).lower()
    return any(keyword.lower() in text_lower for keyword in EXCLUSION_KEYWORDS)

def is_allogeneic_transplant(text: str) -> bool:
    """
    Check if discharge summary indicates an allogeneic transplant
    (contains transplant keywords but not exclusion keywords)
    """
    return contains_transplant_keywords(text) and not contains_exclusion_keywords(text)

def get_transplant_hadm_ids(data_dir: str = 'data/MIMIC-IV', chunk_size: int = 10000) -> List[int]:
    """
    Get hadm_ids for patients with allogeneic transplant-related discharge summaries
    (excludes autologous transplant cases)
    """
    print("Searching for allogeneic transplant-related discharge summaries...")
    print("Excluding autologous transplant cases...")
    discharge_file = f'{data_dir}/discharge.csv'
    
    if not os.path.exists(discharge_file):
        raise FileNotFoundError(f"Discharge file not found: {discharge_file}")
    
    transplant_hadm_ids = []
    excluded_hadm_ids = []
    total_processed = 0
    
    for chunk in pd.read_csv(discharge_file, chunksize=chunk_size):
        # Filter for rows containing transplant keywords
        transplant_mask = chunk['text'].apply(contains_transplant_keywords)
        transplant_chunk = chunk[transplant_mask]
        
        if not transplant_chunk.empty:
            # Further filter to exclude autologous cases
            allogeneic_mask = transplant_chunk['text'].apply(is_allogeneic_transplant)
            allogeneic_chunk = transplant_chunk[allogeneic_mask]
            
            # Add allogeneic cases
            if not allogeneic_chunk.empty:
                hadm_ids = allogeneic_chunk['hadm_id'].dropna().astype(int).tolist()
                transplant_hadm_ids.extend(hadm_ids)
                print(f"Found {len(hadm_ids)} allogeneic transplant patients in current chunk")
        
        total_processed += len(chunk)
      
    # Remove duplicates and sort
    transplant_hadm_ids = sorted(list(set(transplant_hadm_ids)))
    excluded_hadm_ids = sorted(list(set(excluded_hadm_ids)))
    
    print(f"Final results:")
    print(f"  Total allogeneic transplant patients found: {len(transplant_hadm_ids)}")
    print(f"  Total discharge summaries processed: {total_processed}")
    
    return transplant_hadm_ids

def select_hadm_range(hadm_ids: List[int], range_spec: str) -> List[int]:
    """
    Select hadm_ids based on range specification
    Format examples:
    - "100" or "first:100" -> first 100
    - "1000" or "first:1000" -> first 1000  
    - "200:1000" -> from index 200 to 1000
    - "500:1500" -> from index 500 to 1500
    - "all" -> all available hadm_ids
    """
    total_available = len(hadm_ids)
    print(f"Total transplant hadm_ids available: {total_available}")
    
    if range_spec.lower() == "all":
        selected_ids = hadm_ids
        print(f"Selected all {len(selected_ids)} transplant hadm_ids")
        return selected_ids
    
    # Handle range specifications
    if ":" in range_spec:
        parts = range_spec.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid range format: {range_spec}. Use 'start:end' format.")
        
        start_str, end_str = parts
        
        if start_str.lower() == "first":
            start_idx = 0
            end_idx = int(end_str)
        else:
            start_idx = int(start_str)
            end_idx = int(end_str)
    else:
        # Single number means "first N"
        start_idx = 0
        end_idx = int(range_spec)
    
    # Validate indices
    if start_idx < 0:
        start_idx = 0
    if end_idx > total_available:
        end_idx = total_available
        print(f"Warning: End index adjusted to {end_idx} (total available)")
    
    if start_idx >= end_idx:
        raise ValueError(f"Invalid range: start_idx ({start_idx}) >= end_idx ({end_idx})")
    
    selected_ids = hadm_ids[start_idx:end_idx]
    print(f"Selected transplant hadm_ids from index {start_idx} to {end_idx}: {len(selected_ids)} patients")
    
    return selected_ids

class TransplantPatientDataProcessor(PatientDataProcessor):
    """
    Class to handle efficient batch processing of transplant patient data
    Inherits from PatientDataProcessor and only overrides the processing message
    """
    
    def process_patients_batch(self, hadm_ids: List[int], chunk_size: int = 10000) -> Dict:
        """
        Process multiple transplant patients efficiently by reusing loaded chunks
        """
        target_hadm_set = set(hadm_ids)
        processed_hadm_ids = set()
        
        print(f"Processing {len(hadm_ids)} transplant patients in batches...")
        
        # Step 1: Process discharge summaries and admissions
        print("Processing discharge summaries and admissions...")
        self._process_discharge_and_admissions(target_hadm_set, processed_hadm_ids, chunk_size)
        
        # Step 2: Process medications (pharmacy and EMAR)
        print("Processing medications...")
        self._process_medications(target_hadm_set, processed_hadm_ids, chunk_size)
        
        # Step 3: Process radiology
        print("Processing radiology...")
        self._process_radiology(target_hadm_set, processed_hadm_ids, chunk_size)
        
        # Step 4: Process microbiology
        print("Processing microbiology...")
        self._process_microbiology(target_hadm_set, processed_hadm_ids, chunk_size)
        
        # Step 5: Process labs
        print("Processing labs...")
        self._process_labs(target_hadm_set, processed_hadm_ids, chunk_size)
        
        # Filter to only successfully processed patients
        final_results = {hadm_id: data for hadm_id, data in self.patient_results.items() 
                        if hadm_id in processed_hadm_ids}
        
        print(f"Successfully processed {len(final_results)} out of {len(hadm_ids)} transplant patients")
        return final_results

def remove_excluded_meds(med_list: List[str], exclusion_keywords: List[str]) -> List[str]:
    """
    Remove excluded medications from a medication list and deduplicate by clean drug names
    """
    if pd.isna(med_list):
        return []
    
    # Convert string representation of list to actual list
    if isinstance(med_list, str):
        try:
            med_list = ast.literal_eval(med_list)
        except:
            med_list = [med_list]
    
    # First, remove excluded medications
    filtered_meds = [med for med in med_list 
                    if not any(keyword.lower() in str(med).lower() for keyword in exclusion_keywords)]
    
    # Then apply deduplication by clean drug names
    deduplicated_meds = deduplicate_medications_by_clean_name(filtered_meds)
    
    return deduplicated_meds

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
                     "content": f"Is the admission reason for the following clinical note mainly related to initial admission for stem cell transplant or complications from stem cell transplant? Return only 'YES' or 'NO'.\n\n{row['admission_prompt']}"}
                ]
            )
            
            if response.choices[0].message.content.strip().upper() == 'YES':
                transplant_admission_hadm_ids.append(row['hadm_id'])
                
        except Exception as e:
            print(f"Error processing row {start_row + i} (hadm_id {row['hadm_id']}): {str(e)}")
            continue
    
    print(f"Found {len(transplant_admission_hadm_ids)} transplant-related admissions")
    return transplant_admission_hadm_ids

def apply_comprehensive_medication_filtering(df: pd.DataFrame, med_column: str, prompt_column: str) -> pd.DataFrame:
    """
    Apply comprehensive medication filtering:
    1. Remove medications mentioned in prompts
    2. Remove "__" placeholders
    3. Deduplicate by clean drug names
    """
    filtered_rows = []
    
    for idx, row in df.iterrows():
        # Apply filtering for medications mentioned in prompt
        filtered_meds = filter_medications_mentioned_in_prompt(
            row[med_column], 
            row[prompt_column]
        )
        
        # Only keep rows with remaining medications
        if filtered_meds:
            new_row = row.copy()
            new_row[med_column] = filtered_meds
            filtered_rows.append(new_row)
    
    return pd.DataFrame(filtered_rows)

def main():
    parser = argparse.ArgumentParser(description='Create MIMIC-IV transplant dataset with efficient batch processing. Optionally includes medication filtering and OpenAI screening for optimization.')
    parser.add_argument('--data_dir', type=str, default='data/MIMIC-IV',
                      help='Directory containing MIMIC-IV CSV files')
    parser.add_argument('--range', type=str, required=True,
                      help='Range specification: "100", "first:1000", "200:1000", or "all"')
    parser.add_argument('--chunk_size', type=int, default=10000,
                      help='Chunk size for processing CSV files (default: 10000)')
    parser.add_argument('--output_dir', type=str, default='results/mimic_iv_transplant_dataset',
                      help='Output directory for results')
    
    # Optimization arguments
    parser.add_argument('--optimize', action='store_true',
                      help='Enable optimization with medication filtering and OpenAI screening. Replaces standard output with optimized dataset.')
    parser.add_argument('--openai_start_row', type=int, default=0,
                      help='Starting row for OpenAI screening (0-based index). Only used with --optimize flag.')
    parser.add_argument('--openai_end_row', type=int, default=None,
                      help='Ending row for OpenAI screening (exclusive). If not provided, processes all rows. Only used with --optimize flag.')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Get transplant-related hadm_ids
        transplant_hadm_ids = get_transplant_hadm_ids(args.data_dir, args.chunk_size)
        
        if not transplant_hadm_ids:
            print("No transplant patients found in the dataset!")
            sys.exit(1)
        
        # Select range
        selected_hadm_ids = select_hadm_range(transplant_hadm_ids, args.range)
        
        # Save selected hadm_ids with transplant info
        with open(f'{args.output_dir}/selected_transplant_hadm_ids.json', 'w') as f:
            json.dump({
                'hadm_ids': selected_hadm_ids, 
                'total_count': len(selected_hadm_ids),
                'total_allogeneic_transplant_patients': len(transplant_hadm_ids),
                'transplant_keywords': TRANSPLANT_KEYWORDS,
                'exclusion_keywords': EXCLUSION_KEYWORDS,
                'description': 'This dataset includes only allogeneic transplant patients. Autologous transplant cases have been excluded.'
            }, f, indent=2)
        
        # Process transplant patients in batch
        processor = TransplantPatientDataProcessor(args.data_dir)
        patient_results = processor.process_patients_batch(selected_hadm_ids, args.chunk_size)
        
        # Convert to prompts and save
        all_admission_data = []
        all_progression_data = []
        
        print("Converting transplant patient data to prompts...")
        for hadm_id, patient_data in patient_results.items():
            admission_data, progression_notes = process_patient_data_to_prompts(patient_data)
            
            if admission_data and admission_data.get('assess_plan', '').lower() != "no assessment and plan available" or admission_data.get('admission_prompt', '').lower() != "see worksheet":
                all_admission_data.append(admission_data)
            
            if progression_notes:
                for note in progression_notes:
                    if note.get('assess_plan', '').lower() != "no assessment and plan available" or note.get('progression_prompt', '').lower() != "see worksheet":
                        all_progression_data.append(note)
        
        # Clean text fields before saving to prevent CSV formatting issues
        print("Cleaning text fields for CSV formatting...")
        
        # Clean admission data
        for entry in all_admission_data:
            entry['admission_prompt'] = clean_text_for_csv(entry['admission_prompt'])
            entry['assess_plan'] = clean_text_for_csv(entry['assess_plan'])
        
        # Clean progression data  
        for entry in all_progression_data:
            entry['progression_prompt'] = clean_text_for_csv(entry['progression_prompt'])
            entry['assess_plan'] = clean_text_for_csv(entry['assess_plan'])

        # Convert list fields to strings for better CSV compatibility (only if not optimizing)
        if not args.optimize:
            print("Converting list fields to CSV-compatible format...")
            all_admission_data = convert_list_fields_to_string(all_admission_data)
            all_progression_data = convert_list_fields_to_string(all_progression_data)

            # Save initial results with robust CSV formatting
            if all_admission_data:
                admission_df = pd.DataFrame(all_admission_data)
                save_dataframe_to_csv_safely(admission_df, f'{args.output_dir}/transplant_admission_notes.csv', "transplant admission notes")
            
            if all_progression_data:
                progression_df = pd.DataFrame(all_progression_data)
                save_dataframe_to_csv_safely(progression_df, f'{args.output_dir}/transplant_progression_notes.csv', "transplant progression notes")
            
            print(f"Transplant dataset creation complete! Results saved to {args.output_dir}")
            print(f"Found and processed {len(patient_results)} transplant patients out of {len(transplant_hadm_ids)} total transplant cases")
        
        # Optimization step (if enabled)
        if args.optimize:
            print("\n" + "="*60)
            print("Starting dataset optimization...")
            print("="*60)
            
            if not all_admission_data or not all_progression_data:
                print("Warning: No data available for optimization. Skipping optimization step.")
                return
            
            # Load data as DataFrames for optimization (keep lists as lists)
            progress_note = pd.DataFrame(all_progression_data)
            admission_note = pd.DataFrame(all_admission_data)
            
            # Step 1: Filter medications
            med_exclusion = [
                'replacement', 'powder', 'nasal', 'patch', 'alteplase', 'calcium gluconate', 
                'potassium chloride', 'hydromorphone', 'oxycodone', 'magnesium oxide', 
                'multivitamin', 'neutra-phos', 'dakins', 'artificial tears', 'senna', 'loction'
            ]
            
            progress_note, admission_note = filter_medications(progress_note, admission_note, med_exclusion)
            
            # Step 1.5: Apply comprehensive medication filtering (remove meds mentioned in prompts, duplicates)
            print("Applying comprehensive medication filtering...")
            original_progress_count_2 = len(progress_note)
            original_admission_count_2 = len(admission_note)
            
            progress_note = apply_comprehensive_medication_filtering(progress_note, 'newly_started_meds', 'progression_prompt')
            admission_note = apply_comprehensive_medication_filtering(admission_note, 'admission_med', 'admission_prompt')
            
            print(f"After comprehensive filtering:")
            print(f"  Progression notes: {original_progress_count_2} → {len(progress_note)}")
            print(f"  Admission notes: {original_admission_count_2} → {len(admission_note)}")
            
            # Step 2: OpenAI screening (if we have data after medication filtering)
            if len(admission_note) > 0:
                transplant_admission_hadm_ids = screen_with_openai(
                    admission_note, 
                    start_row=args.openai_start_row,
                    end_row=args.openai_end_row
                )
                
                # Filter both datasets to only include transplant-related admissions
                print("Filtering datasets to transplant-related admissions...")
                original_admission_count = len(admission_note)
                original_progress_count = len(progress_note)
                
                admission_note = admission_note[admission_note['hadm_id'].isin(transplant_admission_hadm_ids)]
                progress_note = progress_note[progress_note['hadm_id'].isin(transplant_admission_hadm_ids)]
                
                print(f"Admission notes: {original_admission_count} → {len(admission_note)}")
                print(f"Progression notes: {original_progress_count} → {len(progress_note)}")
                
                # Convert list fields to strings for CSV compatibility
                print("Converting optimized list fields to CSV-compatible format...")
                admission_note_csv = convert_list_fields_to_string(admission_note.to_dict('records'))
                progress_note_csv = convert_list_fields_to_string(progress_note.to_dict('records'))
                
                # Save optimized datasets directly to main output directory
                print(f"Saving optimized datasets to {args.output_dir}")
                if len(admission_note_csv) > 0:
                    admission_df_optimized = pd.DataFrame(admission_note_csv)
                    save_dataframe_to_csv_safely(admission_df_optimized, f'{args.output_dir}/transplant_admission_notes.csv', "optimized admission notes")
                if len(progress_note_csv) > 0:
                    progress_df_optimized = pd.DataFrame(progress_note_csv)
                    save_dataframe_to_csv_safely(progress_df_optimized, f'{args.output_dir}/transplant_progression_notes.csv', "optimized progression notes")
                
                print("Optimization complete!")
                print(f"Final optimized dataset saved to {args.output_dir}")
                print(f"Final counts:")
                print(f"  Admission notes: {len(admission_note_csv)}")
                print(f"  Progression notes: {len(progress_note_csv)}")
            else:
                print("No data remaining after medication filtering. Skipping OpenAI screening.")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 