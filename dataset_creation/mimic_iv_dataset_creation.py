import pandas as pd
import json
import re
import os
import argparse
import gc
import csv
import ast
from typing import List, Dict, Tuple, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict
import sys

# Configuration: Medications to exclude from analysis
# These are typically routine, maintenance, or non-therapeutic medications
EXCLUDED_MEDICATION_KEYWORDS = [
    'flush',
    'vaccine', 
    'acetaminophen',
    'simethicone',
    'docusate',
]

def extract_drug_name(med_string):
    if pd.isna(med_string) or not isinstance(med_string, str):
        return med_string
    
    # Remove content in parentheses: "drug (brand)" -> "drug"
    med_clean = re.sub(r'\([^)]*\)', '', med_string).strip()
    
    # Remove dosage patterns (mg, mcg, units, etc.)
    med_clean = re.sub(r'\s+\d+\.?\d*\s*(mg|mcg|g|ml|units?)\b', '', med_clean, flags=re.IGNORECASE).strip()
    
    # Remove pharmaceutical form indicators
    med_clean = re.sub(r'\s+(MODIFIED|EXTENDED|IMMEDIATE|RELEASE|TABLET|CAPSULE|INJECTION)$', '', med_clean, flags=re.IGNORECASE).strip()
    
    # Clean up multiple spaces
    med_clean = re.sub(r'\s+', ' ', med_clean).strip()
    
    return med_clean if med_clean else med_string

def deduplicate_medications_by_clean_name(med_list):
    """
    Remove duplicate medications based on clean drug names within a list
    Also removes "__" placeholder medications
    """
    if pd.isna(med_list):
        return []
    
    # Convert string to list if needed
    if isinstance(med_list, str):
        try:
            med_list = ast.literal_eval(med_list)
        except:
            med_list = [med_list]
    
    if not isinstance(med_list, list):
        return [med_list]
    
    # Keep track of seen clean names and preserve first occurrence
    seen_clean_names = set()
    deduplicated = []
    
    for med in med_list:
        if med == "__":  # Skip placeholder medications
            continue
        clean_name = extract_drug_name(med).lower()
        if clean_name not in seen_clean_names:
            seen_clean_names.add(clean_name)
            deduplicated.append(med)
    
    return deduplicated

def filter_medications_mentioned_in_prompt(med_list, prompt_text):
    """
    Remove medications that are already mentioned in the prompt text
    Also applies deduplication by clean drug names
    """
    if pd.isna(med_list) or pd.isna(prompt_text):
        return []
    
    # Convert string to list if needed
    if isinstance(med_list, str):
        try:
            med_list = ast.literal_eval(med_list)
        except:
            med_list = [med_list]
    
    if not isinstance(med_list, list):
        return [med_list]
    
    prompt_lower = str(prompt_text).lower()
    kept_meds = []
    seen_clean_names = set()
    
    for med in med_list:
        if med == "__":  # Skip placeholder medications
            continue
        clean_med = extract_drug_name(med)
        clean_med_lower = clean_med.lower()
        
        # Skip if already seen this clean medication name
        if clean_med_lower in seen_clean_names:
            continue
            
        # Skip if mentioned in prompt
        if clean_med_lower in prompt_lower:
            continue
            
        # Keep this medication
        kept_meds.append(med)
        seen_clean_names.add(clean_med_lower)
    
    return kept_meds

def clean_text_for_csv(text: str) -> str:
    """
    Clean text to prevent CSV formatting issues
    Enhanced for complex medical text with quotes and structured formatting
    """
    if not text or pd.isna(text):
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    # Replace problematic line endings and whitespace
    text = text.replace('\r\n', ' ')  # Windows line endings
    text = text.replace('\n', ' ')    # Unix line endings
    text = text.replace('\r', ' ')    # Mac line endings
    text = text.replace('\t', ' ')    # Tabs
    
    # Handle quotes more aggressively - escape ALL quotes
    text = text.replace('"', '""')
    
    # Fix datetime colons that confuse CSV parsers
    # Replace colons in time portions (HH:MM:SS) with periods
    # Pattern to match time portions like "16:54:00" in datetime strings
    text = re.sub(r'\b(\d{4}-\d{2}-\d{2}) (\d{2}):(\d{2}):(\d{2})\b', r'\1 \2.\3.\4', text)
    
    # Remove other problematic characters
    text = text.replace('\x00', '')  # Null bytes
    text = text.replace('\x01', '')  # Start of heading
    text = text.replace('\x02', '')  # Start of text
    text = text.replace('\x03', '')  # End of text
    
    # Clean up excessive whitespace
    text = ' '.join(text.split())
    
    # Truncate extremely long text (CSV parsers may have limits)
    if len(text) > 32000:  # Excel's cell limit is ~32K characters
        text = text[:32000] + "... [TRUNCATED]"
        print(f"Warning: Text truncated to 32,000 characters for CSV compatibility")
    
    return text.strip()

def get_available_hadm_ids(data_dir: str = 'data/MIMIC-IV') -> List[int]:
    """
    Get all available hadm_ids from discharge.csv
    """
    print("Reading available hadm_ids from discharge.csv...")
    discharge_file = f'{data_dir}/discharge.csv'
    
    if not os.path.exists(discharge_file):
        raise FileNotFoundError(f"Discharge file not found: {discharge_file}")
    
    # Read only the hadm_id column to minimize memory usage
    hadm_ids = []
    chunk_size = 10000
    
    for chunk in pd.read_csv(discharge_file, chunksize=chunk_size, usecols=['hadm_id']):
        hadm_ids.extend(chunk['hadm_id'].dropna().astype(int).tolist())
    
    # Remove duplicates and sort
    hadm_ids = sorted(list(set(hadm_ids)))
    print(f"Found {len(hadm_ids)} unique hadm_ids")
    
    return hadm_ids

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
    print(f"Total available hadm_ids: {total_available}")
    
    if range_spec.lower() == "all":
        selected_ids = hadm_ids
        print(f"Selected all {len(selected_ids)} hadm_ids")
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
    print(f"Selected hadm_ids from index {start_idx} to {end_idx}: {len(selected_ids)} patients")
    
    return selected_ids

def create_labs_with_labels_if_needed(data_dir: str):
    """
    Create labs_with_labels.csv file if it doesn't exist.
    """
    labs_file = f'{data_dir}/labs_with_labels.csv'
    if os.path.exists(labs_file):
        print("Labs with labels file already exists.")
        return
    
    print("Creating labs with labels file (this may take a while)...")
    
    # Load lab labels
    lab_labels = pd.read_csv(f'{data_dir}/d_labitems.csv')
    lab_labels['label'] = lab_labels['label'] + ', ' + lab_labels['fluid']
    lab_labels = lab_labels[['itemid', 'label']]
    
    # Process lab events in chunks to reduce memory usage
    labevents_chunk = pd.read_csv(f'{data_dir}/labevents.csv', chunksize=50000)
    
    first_chunk = True
    for chunk in labevents_chunk:
        labs_with_labels = chunk.merge(lab_labels, on='itemid', how='left')
        
        if first_chunk:
            labs_with_labels.to_csv(labs_file, index=False)
            first_chunk = False
        else:
            labs_with_labels.to_csv(labs_file, mode='a', header=False, index=False)
        
        # Force garbage collection
        del labs_with_labels
        gc.collect()
    
    print("Labs with labels file created successfully.")

def extract_hpi_and_medications(discharge_summary: str) -> Tuple[str, str, str, str]:
    """
    Extract HPI, home medications, discharge medications, and assessment/plan from discharge summary.
    """
    # Extract HPI
    HPI = None
    match = re.search(r'Service:(.*?)Pertinent Results:', discharge_summary, re.DOTALL|re.IGNORECASE) or re.search(r'Service:(.*?)Brief Hospital Course:', discharge_summary, re.DOTALL|re.IGNORECASE)
    if match:
        text = match.group(1).replace('\n', ' ').replace(r'\/', '')
        if 'DISCHARGE: VS' in text.upper():
            secondary_match = re.search(r'(.*?)DISCHARGE:', text, re.DOTALL|re.IGNORECASE)
            if secondary_match:
                HPI = secondary_match.group(1)
        elif 'DISCHARGE PHYSICAL' in text.upper():
            secondary_match = re.search(r'(.*?)DISCHARGE PHYSICAL', text, re.DOTALL|re.IGNORECASE)
            if secondary_match:
                HPI = secondary_match.group(1)
        else:
            HPI = text
    
    # Extract Assessment and Plan
    assess_plan = None
    a_p = (re.search(r'Brief Hospital Course:([\s\S]*)Medications on Admission:', discharge_summary, re.DOTALL|re.IGNORECASE) or 
           re.search(r'Brief Hospital Course:([\s\S]*)The Preadmission Medication list is accurate and complete\.', discharge_summary, re.DOTALL|re.IGNORECASE) or
           re.search(r'DISCHARGE LABS:([\s\S]*)The Preadmission Medication list is accurate and complete\.', discharge_summary, re.DOTALL|re.IGNORECASE))
    if a_p:
        assess_plan = a_p.group(1).replace('\n', ' ').strip()
    else:
        assess_plan = "No assessment and plan available"
    
    # Extract medications
    home_meds_match = re.search(r'The Preadmission Medication list is accurate and complete.(.*?)Discharge Medications:', discharge_summary, re.DOTALL) or re.search(r'Medications on Admission:(.*?)Discharge Medications:', discharge_summary, re.DOTALL)
    if home_meds_match:
        home_meds = home_meds_match.group(1).replace('\n', ',').strip()
    else:
        home_meds = "No home medications listed"
    
    discharge_meds_match = re.search(r'Discharge Medications:(.*?)Discharge Disposition:', discharge_summary, re.DOTALL)
    if discharge_meds_match:
        discharge_meds = discharge_meds_match.group(1).replace('\n', ',').strip()
    else:
        discharge_meds = "No discharge medications listed"
    
    return HPI, home_meds, discharge_meds, assess_plan

class PatientDataProcessor:
    """
    Class to handle efficient batch processing of patient data
    """
    
    def __init__(self, data_dir: str = 'data/MIMIC-IV'):
        self.data_dir = data_dir
        
        # Create labs file if needed
        create_labs_with_labels_if_needed(data_dir)
        
        # Initialize data containers
        self.patient_results = {}
    
    def process_patients_batch(self, hadm_ids: List[int], chunk_size: int = 10000) -> Dict:
        """
        Process multiple patients efficiently by reusing loaded chunks
        """
        target_hadm_set = set(hadm_ids)
        processed_hadm_ids = set()
        
        print(f"Processing {len(hadm_ids)} patients in batches...")
        
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
        
        print(f"Successfully processed {len(final_results)} out of {len(hadm_ids)} patients")
        return final_results
    
    def _process_discharge_and_admissions(self, target_hadm_set: Set[int], processed_hadm_ids: Set[int], chunk_size: int):
        """Process discharge summaries and admission data"""
        # Process discharge summaries
        discharge_file = f'{self.data_dir}/discharge.csv'
        for chunk in pd.read_csv(discharge_file, chunksize=chunk_size):
            chunk_patients = chunk[chunk['hadm_id'].isin(target_hadm_set)]
            
            for _, row in chunk_patients.iterrows():
                hadm_id = row['hadm_id']
                if hadm_id not in self.patient_results:
                    self.patient_results[hadm_id] = {
                        'hadm_id': hadm_id,
                        'subject_id': row['subject_id'],
                        'discharge_summary': row['text'],
                        'discharge_time': row['charttime']
                    }
            
            del chunk_patients
            gc.collect()
        
        # Process admissions
        admissions_file = f'{self.data_dir}/admissions.csv'
        for chunk in pd.read_csv(admissions_file, chunksize=chunk_size):
            chunk_patients = chunk[chunk['hadm_id'].isin(target_hadm_set)]
            
            for _, row in chunk_patients.iterrows():
                hadm_id = row['hadm_id']
                if hadm_id in self.patient_results:
                    self.patient_results[hadm_id]['admission_time'] = row['admittime']
                    processed_hadm_ids.add(hadm_id)
            
            del chunk_patients
            gc.collect()
    
    def _process_medications(self, target_hadm_set: Set[int], processed_hadm_ids: Set[int], chunk_size: int):
        """Process pharmacy and EMAR data"""
        # Initialize medication data
        for hadm_id in processed_hadm_ids:
            self.patient_results[hadm_id]['meds'] = []
            self.patient_results[hadm_id]['emar'] = []
        
        # Process pharmacy data
        pharmacy_file = f'{self.data_dir}/pharmacy.csv'
        if os.path.exists(pharmacy_file):
            for chunk in pd.read_csv(pharmacy_file, chunksize=chunk_size, low_memory=False):
                chunk_patients = chunk[chunk['hadm_id'].isin(target_hadm_set)]
                
                for hadm_id, group in chunk_patients.groupby('hadm_id'):
                    if hadm_id in processed_hadm_ids:
                        self.patient_results[hadm_id]['meds'].extend(group.to_dict('records'))
                
                del chunk_patients
                gc.collect()
        
        # Process EMAR data
        emar_file = f'{self.data_dir}/emar_detail.csv'
        if os.path.exists(emar_file):
            # Get subject_ids for target patients
            subject_ids = {self.patient_results[hadm_id]['subject_id'] for hadm_id in processed_hadm_ids}
            
            for chunk in pd.read_csv(emar_file, chunksize=chunk_size):
                chunk_patients = chunk[chunk['subject_id'].isin(subject_ids)]
                
                # Map back to hadm_id
                for _, row in chunk_patients.iterrows():
                    subject_id = row['subject_id']
                    # Find corresponding hadm_id
                    for hadm_id in processed_hadm_ids:
                        if self.patient_results[hadm_id]['subject_id'] == subject_id:
                            self.patient_results[hadm_id]['emar'].append(row.to_dict())
                
                del chunk_patients
                gc.collect()
    
    def _process_radiology(self, target_hadm_set: Set[int], processed_hadm_ids: Set[int], chunk_size: int):
        """Process radiology data"""
        # Initialize radiology data
        for hadm_id in processed_hadm_ids:
            self.patient_results[hadm_id]['radiology'] = []
        
        radiology_file = f'{self.data_dir}/radiology.csv'
        if os.path.exists(radiology_file):
            for chunk in pd.read_csv(radiology_file, chunksize=chunk_size):
                chunk_patients = chunk[chunk['hadm_id'].isin(target_hadm_set)]
                
                for hadm_id, group in chunk_patients.groupby('hadm_id'):
                    if hadm_id in processed_hadm_ids:
                        self.patient_results[hadm_id]['radiology'].extend(group.to_dict('records'))
                
                del chunk_patients
                gc.collect()
    
    def _process_microbiology(self, target_hadm_set: Set[int], processed_hadm_ids: Set[int], chunk_size: int):
        """Process microbiology data"""
        # Initialize microbiology data
        for hadm_id in processed_hadm_ids:
            self.patient_results[hadm_id]['micro'] = []
        
        micro_file = f'{self.data_dir}/microbiologyevents.csv'
        if os.path.exists(micro_file):
            for chunk in pd.read_csv(micro_file, chunksize=chunk_size):
                chunk_patients = chunk[chunk['hadm_id'].isin(target_hadm_set)]
                
                for hadm_id, group in chunk_patients.groupby('hadm_id'):
                    if hadm_id in processed_hadm_ids:
                        self.patient_results[hadm_id]['micro'].extend(group.to_dict('records'))
                
                del chunk_patients
                gc.collect()
    
    def _process_labs(self, target_hadm_set: Set[int], processed_hadm_ids: Set[int], chunk_size: int):
        """Process lab data"""
        # Initialize lab data
        for hadm_id in processed_hadm_ids:
            self.patient_results[hadm_id]['labs'] = []
        
        # Get subject_ids for target patients
        subject_ids = {self.patient_results[hadm_id]['subject_id'] for hadm_id in processed_hadm_ids}
        
        labs_file = f'{self.data_dir}/labs_with_labels.csv'
        for chunk in pd.read_csv(labs_file, chunksize=chunk_size):
            chunk_patients = chunk[chunk['subject_id'].isin(subject_ids)]
            
            # Map back to hadm_id
            for _, row in chunk_patients.iterrows():
                subject_id = row['subject_id']
                # Find corresponding hadm_id
                for hadm_id in processed_hadm_ids:
                    if self.patient_results[hadm_id]['subject_id'] == subject_id:
                        self.patient_results[hadm_id]['labs'].append(row.to_dict())
            
            del chunk_patients
            gc.collect()

def process_patient_data_to_prompts(patient_data: Dict) -> Tuple[Dict, List[Dict]]:
    """
    Convert raw patient data to admission and progression prompts
    """
    hadm_id = patient_data['hadm_id']
    
    try:
        # Extract HPI and medications from discharge summary
        HPI, home_meds, discharge_meds, assess_plan = extract_hpi_and_medications(patient_data['discharge_summary'])
        
        # Parse times
        admission_time = datetime.strptime(patient_data['admission_time'], '%Y-%m-%d %H:%M:%S')
        discharge_time = datetime.strptime(patient_data['discharge_time'], '%Y-%m-%d %H:%M:%S')
        
        # Process medications
        hospital_meds = process_medications_data(patient_data['meds'], patient_data['emar'])
        
        # Process imaging studies
        imaging_studies = process_imaging_data(patient_data['radiology'])
        
        # Process microbiology
        micro_results = process_micro_data(patient_data['micro'])
        
        # Process labs
        labs_list, admission_labs, baseline_labs = process_labs_data(patient_data['labs'], admission_time)
        
        # Generate admission data
        admission_data = generate_admission_data(
            hadm_id, admission_time, HPI, home_meds, baseline_labs, 
            admission_labs, imaging_studies, hospital_meds, assess_plan
        )
        
        # Generate progression data
        progression_notes = generate_progression_data(
            hadm_id, admission_time, discharge_time, HPI, admission_labs,
            labs_list, imaging_studies, micro_results, home_meds, 
            hospital_meds, assess_plan
        )
        
        return admission_data, progression_notes
        
    except Exception as e:
        print(f"Error processing patient {hadm_id}: {str(e)}")
        return None, []

def process_medications_data(meds_data: List[Dict], emar_data: List[Dict]) -> List[Dict]:
    """Process medication data"""
    hospital_meds = []
    
    # Convert emar_data to DataFrame for easier lookup
    emar_df = pd.DataFrame(emar_data) if emar_data else pd.DataFrame()
    
    for med_record in meds_data:
        try:
            if 'pharmacy_id' not in med_record:
                continue
                
            pharmacy_id = med_record['pharmacy_id']
            
            # Find corresponding EMAR entry
            emar_entry = emar_df[emar_df['pharmacy_id'] == pharmacy_id] if 'pharmacy_id' in emar_df.columns else pd.DataFrame()
            
            med_name = None
            if not emar_entry.empty and 'product_description' in emar_entry.columns:
                med_name = emar_entry['product_description'].iloc[0]
            
            if not med_name and 'medication' in med_record:
                med_name = med_record['medication']
            
            if med_name and pd.notna(med_name):
                end_time = med_record.get('stoptime')
                hospital_meds.append({
                    'med_name': med_name,
                    'start_time': str(med_record.get('entertime', '')),
                    'end_time': str(end_time) if end_time and pd.notna(end_time) else None,
                })
                
        except Exception:
            continue
    
    return hospital_meds

def process_imaging_data(radiology_data: List[Dict]) -> List[Dict]:
    """Process radiology data"""
    imaging_studies = []
    
    for study in radiology_data:
        try:
            text = study.get('text', '')
            imaging_studies.append({
                'imaging_study_time': study['charttime'],
                'imaging_study_text': re.search(r'(.*?)IMPRESSION:', text, re.DOTALL).group(1).strip().replace('\n', ' ') if 'IMPRESSION:' in text else text.replace('\n', ' '),
                'imaging_study_impression': re.search(r'IMPRESSION:(.*)', text, re.DOTALL).group(1).strip().replace('\n', ' ') if 'IMPRESSION:' in text else None,
            })
        except Exception:
            continue
    
    return imaging_studies

def process_micro_data(micro_data: List[Dict]) -> List[Dict]:
    """Process microbiology data"""
    micro_results = []
    
    for micro in micro_data:
        try:
            micro_results.append({
                'micro_study_time': micro['charttime'],
                'micro_study_item': micro['test_name'],
                'micro_study_results': str(micro.get('interpretation', '')) + ', ' + str(micro.get('comments', '')),
            })
        except Exception:
            continue
    
    return micro_results

def process_labs_data(labs_data: List[Dict], admission_time: datetime) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Process lab data"""
    labs_list = []
    admission_labs = []
    baseline_labs = []
    
    for lab in labs_data:
        try:
            if lab.get('flag') != 'abnormal':
                continue
                
            lab_time = datetime.strptime(lab['charttime'], '%Y-%m-%d %H:%M:%S')
            
            lab_value = f"{str(lab['valuenum'])} {str(lab['valueuom']).strip()}" if pd.notna(lab.get('valueuom')) else str(lab.get('value', ''))
            
            lab_dict = {
                'lab_name': lab['label'],
                'lab_value': lab_value,
                'lab_time': lab['charttime']
            }
            labs_list.append(lab_dict)
            
            lab_entry = {'lab': f"{lab['label']}: {lab_value}"}
            
            # Categorize labs by timing
            if admission_time - timedelta(hours=24) <= lab_time <= admission_time + timedelta(hours=6):
                admission_labs.append(lab_entry)
            elif lab_time < admission_time - timedelta(days=2) and len(baseline_labs) < 20:
                baseline_labs.append(lab_entry)
                
        except Exception:
            continue
    
    return labs_list, admission_labs, baseline_labs

def generate_admission_data(hadm_id: int, admission_time: datetime, HPI: str, home_meds: str, 
                          baseline_labs: List[Dict], admission_labs: List[Dict], 
                          imaging_studies: List[Dict], hospital_meds: List[Dict], assess_plan: str) -> Dict:
    """Generate admission note data"""
    
    # Filter admission medications
    admission_meds = []
    for med in hospital_meds:
        try:
            if (not med['med_name'] or 
                str(med['med_name']).lower() in ['nan', 'null', 'none'] or
                str(med['med_name']).strip() == ''):
                continue
            med_start_time = datetime.strptime(med['start_time'], '%Y-%m-%d %H:%M:%S')
            if admission_time <= med_start_time <= admission_time + timedelta(hours=6):
                admission_meds.append(med)
        except Exception:
            continue
    
    # Filter admission scans
    admission_scans = []
    for study in imaging_studies:
        try:
            study_time = datetime.strptime(study['imaging_study_time'], '%Y-%m-%d %H:%M:%S')
            if admission_time - timedelta(hours=6) <= study_time <= admission_time + timedelta(hours=6):
                admission_scans.append(study)
        except Exception:
            continue
    
    # Filter out excluded medications
    filtered_admission_meds = []
    for med in admission_meds:
        med_name_lower = med['med_name'].lower()
        if not any(keyword in med_name_lower for keyword in EXCLUDED_MEDICATION_KEYWORDS):
            filtered_admission_meds.append(med)
    
    # Skip if no admission medications after filtering
    if not filtered_admission_meds:
        return None
    
    # Apply medication deduplication and filtering
    med_names = [med['med_name'] for med in filtered_admission_meds]
    med_names = deduplicate_medications_by_clean_name(med_names)
    
    # Skip if no medications after deduplication
    if not med_names:
        return None
    
    # Generate admission prompt
    admission_prompt = f"""
Admission date: {admission_time}

HPI:
{HPI}

Home medications:
{home_meds}

Abnormal labs at baseline (prior to admission):
{chr(10).join([f"- {lab['lab']}" for lab in baseline_labs]) if baseline_labs else "No baseline abnormal labs available"}

Abnormal labs at admission:
{chr(10).join([f"- {lab['lab']}" for lab in admission_labs]) if admission_labs else "No abnormal labs at admission"}

Imaging studies at admission:
{chr(10).join([f"- {study['imaging_study_time']}: {study['imaging_study_text']}" + (f"{chr(10)}  Impression: {study['imaging_study_impression']}" if study['imaging_study_impression'] else "") for study in admission_scans]) if admission_scans else "No imaging studies at admission"}

What medication should be started for this patient at admission based on the above information?
"""
    
    return {
        'hadm_id': hadm_id,
        'admission_time': admission_time.strftime('%Y-%m-%d %H:%M:%S'),
        'admission_prompt': admission_prompt,
        'admission_med': med_names,
        'assess_plan': assess_plan
    }

def generate_progression_data(hadm_id: int, admission_time: datetime, discharge_time: datetime,
                            HPI: str, admission_labs: List[Dict], labs_list: List[Dict],
                            imaging_studies: List[Dict], micro_results: List[Dict], 
                            home_meds: str, hospital_meds: List[Dict], assess_plan: str) -> List[Dict]:
    """Generate progression notes data"""
    
    progression_notes = []
    current_time = admission_time + timedelta(hours=6)
    
    while current_time < discharge_time + timedelta(hours=24):
        next_time = current_time + timedelta(hours=24)
        
        # Filter newly started medications
        newly_started_meds = []        
        for med in hospital_meds:
            try:
                if (not med['med_name'] or 
                    str(med['med_name']).lower() in ['nan', 'null', 'none'] or 
                    str(med['med_name']).strip() == ''):
                    continue
                med_start_time = datetime.strptime(med['start_time'], '%Y-%m-%d %H:%M:%S')
                if current_time < med_start_time <= next_time:
                    newly_started_meds.append(med)
            except Exception:
                continue
        
        # Filter recent labs (past 72 hours from current_time)
        recent_labs = []
        for lab in labs_list:
            try:
                lab_time = datetime.strptime(lab['lab_time'], '%Y-%m-%d %H:%M:%S')
                if current_time - timedelta(hours=72) <= lab_time <= current_time:
                    recent_labs.append(lab)
            except Exception:
                continue
        
        # Filter image studies (from admission until current_time)
        current_imaging = []
        for study in imaging_studies:
            try:
                study_time = datetime.strptime(study['imaging_study_time'], '%Y-%m-%d %H:%M:%S')
                if admission_time <= study_time <= current_time:
                    current_imaging.append(study)
            except Exception:
                continue
        
        # Filter microbiology (from admission until current_time)
        current_micro = []
        for micro in micro_results:
            try:
                micro_time = datetime.strptime(micro['micro_study_time'], '%Y-%m-%d %H:%M:%S')
                if admission_time <= micro_time <= current_time:
                    current_micro.append(micro)
            except Exception:
                continue
        
        # Filter current medications (active at current_time)
        current_meds = []
        for med in hospital_meds:
            try:
                if (not med['med_name'] or 
                    str(med['med_name']).lower() in ['nan', 'null', 'none'] or 
                    str(med['med_name']).strip() == ''):
                    continue
                start_time = datetime.strptime(med['start_time'], '%Y-%m-%d %H:%M:%S')
                # Check if medication is active at current_time
                if start_time <= current_time:
                    # If no end_time or end_time is after current_time, medication is active
                    if not med['end_time'] or med['end_time'] == 'Ongoing':
                        current_meds.append(med)
                    else:
                        try:
                            end_time = datetime.strptime(med['end_time'], '%Y-%m-%d %H:%M:%S')
                            if end_time >= current_time:
                                current_meds.append(med)
                        except (ValueError, TypeError):
                            # If end_time can't be parsed, assume medication is ongoing
                            current_meds.append(med)
            except Exception:
                continue
        
        # Filter recently stopped medications (stopped between admission and current time)
        recently_stopped_meds = []
        for med in hospital_meds:
            try:
                if (not med['med_name'] or 
                    str(med['med_name']).lower() in ['nan', 'null', 'none'] or 
                    str(med['med_name']).strip() == ''):
                    continue
                if med['end_time'] and med['end_time'] != 'Ongoing':
                    try:
                        end_time = datetime.strptime(med['end_time'], '%Y-%m-%d %H:%M:%S')
                        if admission_time <= end_time <= current_time:
                            recently_stopped_meds.append(med)
                    except (ValueError, TypeError):
                        # Skip if end_time can't be parsed
                        continue
            except Exception:
                continue
        
        # Remove medications from newly_started_meds if they already exist in current_meds
        # This prevents duplicate listing of medications that are both newly started and currently active
        current_med_names = {med['med_name'].lower().strip() for med in current_meds}
        newly_started_meds = [med for med in newly_started_meds 
                             if med['med_name'].lower().strip() not in current_med_names]
        
        # Filter out excluded medications from newly started meds
        filtered_newly_started_meds = []
        for med in newly_started_meds:
            med_name_lower = med['med_name'].lower()
            if not any(keyword in med_name_lower for keyword in EXCLUDED_MEDICATION_KEYWORDS):
                filtered_newly_started_meds.append(med)
        
        # Skip this time period if no newly started medications after filtering
        if not filtered_newly_started_meds:
            current_time += timedelta(hours=24)
            continue
        
        # Apply medication deduplication and filtering
        newly_started_med_names = [med['med_name'] for med in filtered_newly_started_meds]
        newly_started_med_names = deduplicate_medications_by_clean_name(newly_started_med_names)
        
        # Skip this time period if no medications after deduplication
        if not newly_started_med_names:
            current_time += timedelta(hours=24)
            continue
        
        # Generate progression prompt
        progression_prompt = f"""
Current time: {current_time}
Admission date: {admission_time}

HPI:
{HPI}

Abnormal labs at admission:
{chr(10).join([f"- {lab['lab']}" for lab in admission_labs]) if admission_labs else "No abnormal labs at admission"}

Recent labs (past 72 hours):
{chr(10).join([f"- {lab['lab_name']}: {lab['lab_value']} at {lab['lab_time']}" for lab in recent_labs]) if recent_labs else "No recent abnormal labs"}

Image studies (since admission):
{chr(10).join([f"- {study['imaging_study_time']}: {study['imaging_study_text']}" + (f"{chr(10)}  Impression: {study['imaging_study_impression']}" if study['imaging_study_impression'] else "") for study in current_imaging]) if current_imaging else "No imaging studies"}

Microbiology (since admission):
{chr(10).join([f"- {micro['micro_study_time']}: {micro['micro_study_item']} - {micro['micro_study_results']}" for micro in current_micro]) if current_micro else "No microbiology results"}

Home medications:
{home_meds}

Current medications:
{chr(10).join([f"- {med['med_name']} (Started: {med['start_time']})" for med in current_meds]) if current_meds else "No current medications"}

Recently stopped medications:
{chr(10).join([f"- {med['med_name']} (Started: {med['start_time']}, Stopped: {med['end_time']})" for med in recently_stopped_meds]) if recently_stopped_meds else "No recently stopped medications"}

What medication should be started for this patient today based on the above information?
"""
        
        progression_data = {
            'hadm_id': hadm_id,
            'current_time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'admission_time': admission_time.strftime('%Y-%m-%d %H:%M:%S'),
            'progression_prompt': progression_prompt,
            'newly_started_meds': newly_started_med_names,
            'assess_plan': assess_plan
        }
        progression_notes.append(progression_data)
        
        current_time += timedelta(hours=24)
    
    return progression_notes

def save_dataframe_to_csv_safely(df: pd.DataFrame, filepath: str, description: str = "data") -> None:
    """
    Save DataFrame to CSV with robust formatting to handle problematic text
    """
    try:
        # Method 1: Use pandas with comprehensive quoting
        df.to_csv(filepath, 
                 index=False,
                 quoting=1,  # csv.QUOTE_ALL - quotes all fields
                 quotechar='"',  # standard quote character
                 escapechar=None,  # let pandas handle escaping
                 encoding='utf-8',  # handle special characters
                 lineterminator='\n')  # consistent line endings
        print(f"Successfully saved {len(df)} {description} using pandas method")
        
    except Exception as e:
        print(f"Warning: pandas method failed ({str(e)}), trying alternative method...")
        
        # Method 2: Manual CSV writing with Python's csv module
        import csv
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile, 
                                  quoting=csv.QUOTE_ALL,  # Quote all fields
                                  quotechar='"',
                                  escapechar=None,
                                  lineterminator='\n')
                
                # Write header
                writer.writerow(df.columns.tolist())
                
                # Write data rows
                for _, row in df.iterrows():
                    # Convert all values to strings to ensure consistent handling
                    row_values = [str(val) if pd.notna(val) else "" for val in row.values]
                    writer.writerow(row_values)
            
            print(f"Successfully saved {len(df)} {description} using csv module method")
            
        except Exception as e2:
            print(f"Error: Both CSV saving methods failed. pandas: {str(e)}, csv: {str(e2)}")
            
            # Method 3: Save as JSON as fallback
            json_filepath = filepath.replace('.csv', '.json')
            df.to_json(json_filepath, orient='records', indent=2)
            print(f"Fallback: Saved {description} as JSON to {json_filepath}")

def convert_list_fields_to_string(data_list: List[Dict]) -> List[Dict]:
    """
    Convert list fields to properly formatted strings to avoid CSV issues
    (excluding medication fields which should remain as lists)
    """
    # Fields that should remain as lists (medication fields)
    exclude_fields = {'admission_med', 'newly_started_meds'}
    
    for entry in data_list:
        for key, value in entry.items():
            if key not in exclude_fields:  # Skip medication fields
                if isinstance(value, list):
                    # Convert list to pipe-separated string for better CSV compatibility
                    entry[key] = ' | '.join([str(item) for item in value if item and str(item).strip()])
                elif isinstance(value, dict):
                    # Convert dict to JSON string
                    entry[key] = json.dumps(value)
    
    return data_list

def main():
    parser = argparse.ArgumentParser(description='Create MIMIC-IV dataset with efficient batch processing')
    parser.add_argument('--data_dir', type=str, default='data/MIMIC-IV',
                      help='Directory containing MIMIC-IV CSV files')
    parser.add_argument('--range', type=str, required=True,
                      help='Range specification: "100", "first:1000", "200:1000", or "all"')
    parser.add_argument('--chunk_size', type=int, default=10000,
                      help='Chunk size for processing CSV files (default: 10000)')
    parser.add_argument('--output_dir', type=str, default='results/mimic_iv_dataset',
                      help='Output directory for results')
    parser.add_argument('--apply_med_filtering', action='store_true',
                      help='Apply comprehensive medication filtering (remove duplicates and meds mentioned in prompts)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Get available hadm_ids
        available_hadm_ids = get_available_hadm_ids(args.data_dir)
        
        # Select range
        selected_hadm_ids = select_hadm_range(available_hadm_ids, args.range)
        
        # Save selected hadm_ids
        with open(f'{args.output_dir}/selected_hadm_ids.json', 'w') as f:
            json.dump({'hadm_ids': selected_hadm_ids, 'total_count': len(selected_hadm_ids)}, f, indent=2)
        
        # Process patients in batch
        processor = PatientDataProcessor(args.data_dir)
        patient_results = processor.process_patients_batch(selected_hadm_ids, args.chunk_size)
        
        # Convert to prompts and save
        all_admission_data = []
        all_progression_data = []
        
        print("Converting patient data to prompts...")
        for hadm_id, patient_data in patient_results.items():
            admission_data, progression_notes = process_patient_data_to_prompts(patient_data)
            
            if admission_data and admission_data.get('assess_plan', '').lower() != "no assessment and plan available":
                all_admission_data.append(admission_data)
            
            if progression_notes:
                for note in progression_notes:
                    if note.get('assess_plan', '').lower() != "no assessment and plan available":
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

        # Apply comprehensive medication filtering if requested
        if args.apply_med_filtering:
            print("Applying comprehensive medication filtering...")
            original_admission_count = len(all_admission_data)
            original_progression_count = len(all_progression_data)
            
            # Filter admission data
            filtered_admission_data = []
            for entry in all_admission_data:
                filtered_meds = filter_medications_mentioned_in_prompt(
                    entry['admission_med'], 
                    entry['admission_prompt']
                )
                if filtered_meds:
                    entry['admission_med'] = filtered_meds
                    filtered_admission_data.append(entry)
            
            # Filter progression data
            filtered_progression_data = []
            for entry in all_progression_data:
                filtered_meds = filter_medications_mentioned_in_prompt(
                    entry['newly_started_meds'], 
                    entry['progression_prompt']
                )
                if filtered_meds:
                    entry['newly_started_meds'] = filtered_meds
                    filtered_progression_data.append(entry)
            
            all_admission_data = filtered_admission_data
            all_progression_data = filtered_progression_data
            
            print(f"After comprehensive medication filtering:")
            print(f"  Admission notes: {original_admission_count} → {len(all_admission_data)}")
            print(f"  Progression notes: {original_progression_count} → {len(all_progression_data)}")

        # Save results with proper CSV formatting
        if all_admission_data:
            admission_df = pd.DataFrame(all_admission_data)
            save_dataframe_to_csv_safely(admission_df, f'{args.output_dir}/admission_notes.csv', "admission notes")
            print(f"Saved {len(all_admission_data)} admission notes")
        
        if all_progression_data:
            progression_df = pd.DataFrame(all_progression_data)
            save_dataframe_to_csv_safely(progression_df, f'{args.output_dir}/progression_notes.csv', "progression notes")
            print(f"Saved {len(all_progression_data)} progression notes")
        
        print(f"Dataset creation complete! Results saved to {args.output_dir}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
