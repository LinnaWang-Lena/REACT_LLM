#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retry failed prediction tasks
Used to detect and reprocess tasks that failed or were missing in prediction.py
"""

# Mode selection: True=manual mode, False=automatic mode
MANUAL_MODE = True # Choose the mode here

# Manual mode configuration - only effective when MANUAL_MODE=True
MANUAL_CONFIG = {
    # Input data file (relative path or full path)
    "input_csv": "your_input_csv_file.csv",
    
    # Result file to be retried - specify the full file path
    "csv_file": "your_result_csv_file.csv",
    "json_file": "your_result_json_file.json", 
    "txt_file": "your_result_txt_file.txt",
    
    # Manually specify model configuration (if None, automatically extracted from JSON file)
    "override_model_config": {
        "display_name":"your_model_name",        # Must modify the model name
        "model_name": "your_model_name",       # Must use the exact model name
        "cd_algorithm": "your_cd_algorithm",                 # CD algorithm configuration (used for CD_FEATURES_OPTIMIZED and CD_FILTERED modes): 'CORL', 'DirectLiNGAM', 'GES'
        "api_type": "openai",
        "openai_config": {  # Complete the API configuration
            "api_key": "your_api_key", # Here you need to change it to your own api_key
            "api_base": "your_api_base"
        },
        "generation_params": {  # Supplement generation parameters
            "temperature": 0.0,
            "max_tokens": 500
        },
    },
    
    # Optional: manually specify label and prompt mode (if None, automatically extracted from file name)
    "override_label": None,          # Example: "DIEINHOSPITAL"
    "override_prompt_mode": "your_prompt_mode",    # Optional values: 'DIRECTLY_PROMPTING', 'CHAIN_OF_THOUGHT', 'SELF_REFLECTION', 'ROLE_PLAYING', 'IN_CONTEXT_LEARNING', 'CSV_DIRECT', 'CSV_RAW', 'JSON_STRUCTURED', 'LATEX_TABLE', 'NATURAL_LANGUAGE', 'CORL_FILTERED', 'DirectLiNGAM_FILTERED', 'CD_FILTERED', 'CD_FEATURES_OPTIMIZED', 'LLM_CD_FEATURES'
}

import os
import json
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Set, Tuple, Optional
import sys
import time
import uuid
from datetime import datetime
import re
import shutil
from pathlib import Path

# ANSI color definition class
class Colors:
    """Academic terminal output color definition"""
    RED = '\033[91m'      # Error information
    GREEN = '\033[92m'    # Success information  
    YELLOW = '\033[93m'   # Warning information
    WHITE = '\033[97m'    # General information
    RESET = '\033[0m'     # Reset color

def safe_file_operation(file_path, operation_func, max_retries=3, retry_delay=1):
    """
    Safe file operation, handling permission errors and file occupation problems
    """
    for attempt in range(max_retries):
        try:
            return operation_func()
        except PermissionError as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} File is occupied, waiting {wait_time} seconds to retry... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"{Colors.RED}[ERROR]{Colors.RESET} File operation finally failed: {str(e)}")
                raise
        except Exception as e:
            print(f"{Colors.RED}[ERROR]{Colors.RESET} File operation exception: {str(e)}")
            raise

def create_file_backup(file_path):
    """Create file backup"""
    if os.path.exists(file_path):
        backup_path = f"{file_path}.backup"
        try:
            shutil.copy2(file_path, backup_path)
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} Backup created: {os.path.basename(backup_path)}")
            return backup_path
        except Exception as e:
            print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Backup creation failed: {str(e)}")
            return None
    return None

def restore_from_backup(file_path, backup_path):
    """Restore file from backup"""
    if backup_path and os.path.exists(backup_path):
        try:
            shutil.copy2(backup_path, file_path)
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} Restored from backup: {os.path.basename(file_path)}")
            return True
        except Exception as e:
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Backup restoration failed: {str(e)}")
            return False
    return False

def update_single_csv_record(csv_file_path: str, patient_id: str, prediction_data: Dict, label: str) -> bool:
    """
    Update a single record in the CSV file, save in real time
    """
    def update_operation():
        pred_col = f'{label.lower()}_prediction'
        prob_col = f'{label.lower()}_probability'
        
        # Read the existing CSV
        if os.path.exists(csv_file_path):
            # Explicitly specify data types to avoid type mismatch warnings
            df = pd.read_csv(csv_file_path, dtype={
                'icustay_id': str,
                'patient_id': str,
                pred_col: 'Int64',      # Nullable integer type
                prob_col: float,
                'ground_truth': 'Int64'
            })
            # Ensure the key column is a string type (to prevent type confusion)
            if 'patient_id' in df.columns:
                df['patient_id'] = df['patient_id'].astype(str)
            if 'icustay_id' in df.columns:
                df['icustay_id'] = df['icustay_id'].astype(str)
        else:
            # When creating a new DataFrame, explicitly specify the data type
            df = pd.DataFrame(columns=['icustay_id', 'patient_id', pred_col, prob_col, 'ground_truth'])
            df = df.astype({
                'icustay_id': str,
                'patient_id': str,
                pred_col: 'Int64',      # Nullable integer type
                prob_col: float,
                'ground_truth': 'Int64'
            })
        
        # Prepare new record, explicitly convert all value types
        new_record = {
            'icustay_id': str(prediction_data['icustay_id']),
            'patient_id': str(patient_id),
            pred_col: int(prediction_data['prediction']),
            prob_col: float(prediction_data['probability']),
            'ground_truth': int(prediction_data['groundtruth'])
        }
        
        # Check if there is a record for this patient_id
        mask = df['patient_id'].astype(str) == str(patient_id)
        
        if mask.any():
            # Update existing record
            for col, value in new_record.items():
                df.loc[mask, col] = value
            operation_type = "Update"
        else:
            # Add new record
            new_row_df = pd.DataFrame([new_record])
            df = pd.concat([df, new_row_df], ignore_index=True)
            operation_type = "Add"
        
        # Ensure column order
        column_order = ['icustay_id', 'patient_id', pred_col, prob_col, 'ground_truth']
        df = df[column_order]
        
        # Save file
        df.to_csv(csv_file_path, index=False)
        return operation_type
    
    try:
        operation_type = safe_file_operation(csv_file_path, update_operation)
        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} CSV {operation_type} successfully: {patient_id}")
        return True
    except Exception as e:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} CSV update failed: {patient_id}, error: {str(e)}")
        return False

def update_single_json_record(json_file_path: str, patient_id: str, experiment_log: Dict) -> bool:
    """
    Update a single record in the JSON file, save in real time
    """
    def update_operation():
        # Read the existing JSON
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        else:
            logs = []
        
        # Check if there is a record for this patient_id
        updated = False
        for i, log in enumerate(logs):
            if str(log.get('patient_id', '')) == str(patient_id):
                logs[i] = experiment_log
                updated = True
                break
        
        operation_type = "Update" if updated else "Add"
        if not updated:
            logs.append(experiment_log)
        
        # Save file
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
        
        return operation_type
    
    try:
        operation_type = safe_file_operation(json_file_path, update_operation)
        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} JSON {operation_type} successfully: {patient_id}")
        return True
    except Exception as e:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} JSON update failed: {patient_id}, error: {str(e)}")
        return False

def save_retry_progress(progress_file: str, completed_ids: set, failed_ids: set, task_metadata: dict = None):
    """Save retry progress to file, including task verification information"""
    progress_data = {
        'timestamp': datetime.now().isoformat(),
        'completed_ids': list(completed_ids),
        'failed_ids': list(failed_ids),
        'task_metadata': task_metadata or {}  # Add task metadata for verification
    }
    
    try:
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Progress saving failed: {str(e)}")
        return False

def load_retry_progress(progress_file: str, current_task_metadata: dict = None) -> Tuple[set, set]:
    """Load retry progress from file, including task verification"""
    if not os.path.exists(progress_file):
        return set(), set()
    
    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            progress_data = json.load(f)
        
        completed_ids = set(progress_data.get('completed_ids', []))
        failed_ids = set(progress_data.get('failed_ids', []))
        saved_metadata = progress_data.get('task_metadata', {})
        
        # Task verification: check if it is the same task
        is_same_task = True
        validation_info = []
        
        if current_task_metadata and saved_metadata:
            # Check if the key task identifier matches
            key_fields = ['csv_file', 'json_file', 'label', 'model_name']
            for field in key_fields:
                current_value = current_task_metadata.get(field)
                saved_value = saved_metadata.get(field)
                if current_value and saved_value and current_value != saved_value:
                    is_same_task = False
                    validation_info.append(f"{Colors.RED}[ERROR]{Colors.RESET} {field}: current='{current_value}' vs saved='{saved_value}'")
                elif current_value and saved_value:
                    validation_info.append(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} {field}: '{current_value}'")
        
        if completed_ids or failed_ids:
            timestamp = progress_data.get('timestamp', 'Unknown')
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} Found checkpoint file (time: {timestamp})")
            print(f"Completed: {len(completed_ids)} tasks")
            print(f"Failed: {len(failed_ids)} tasks")
            
            # Display task verification results
            if current_task_metadata and saved_metadata:
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} Task verification results:")
                for info in validation_info:
                    print(info)
                
                if not is_same_task:
                    print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Warning: checkpoint file belongs to different tasks, will be ignored and new progress will be created")
                    return set(), set()  # Return empty set, force restart
                else:
                    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Task verification passed, using checkpoint data")
            elif not saved_metadata:
                print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Old version progress file (no task verification information), will be used with caution")
        
        return completed_ids, failed_ids
    except Exception as e:
        print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Progress loading failed: {str(e)}")
        return set(), set()
    
def validate_manual_config() -> bool:
    """Validate the existence and validity of the manually configured file"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Validating manually configured file...")
    
    # Validate input data file
    input_csv = MANUAL_CONFIG["input_csv"]
    if not os.path.isabs(input_csv):
        input_csv = os.path.join(script_dir, input_csv)
    
    if not os.path.exists(input_csv):
        print(f"{Colors.RED}[ERROR]{Colors.RESET} Input data file does not exist: {input_csv}")
        return False
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Input data file: {input_csv}")
    
    # New: MIMIC version matching check
    input_filename = os.path.basename(MANUAL_CONFIG["input_csv"])
    csv_filename = os.path.basename(MANUAL_CONFIG["csv_file"])
    json_filename = os.path.basename(MANUAL_CONFIG["json_file"])
    txt_filename = os.path.basename(MANUAL_CONFIG["txt_file"])
    
    # Extract MIMIC version from input file name
    input_version = None
    if 'MIMIC3' in input_filename:
        input_version = '3'
    elif 'MIMIC4' in input_filename:
        input_version = '4'
    
    # Extract version prefix from result file name
    result_version = None
    if csv_filename.startswith('3_'):
        result_version = '3'
    elif csv_filename.startswith('4_'):
        result_version = '4'
    
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Version matching check:")
    print(f"Input file: {input_filename} -> MIMIC version: {input_version or 'Unknown'}")
    print(f"Result file: {csv_filename} -> Version prefix: {result_version or 'No prefix'}")
    
    # Check if the version matches
    if input_version and result_version and input_version != result_version:
        print(f"\n{Colors.RED}[ERROR]{Colors.RESET} Configuration error: MIMIC version mismatch!")
        print(f"Input data file is MIMIC{input_version}, but result file starts with {result_version}_")
        print(f"This could cause serious data confusion!")
        print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Solution:")
        print(f"1. If you want to retry the results of MIMIC{input_version}, please modify the result file path to start with {input_version}_")
        print(f"2. If you want to retry the results of MIMIC{result_version}, please modify the input file to New_MIMIC{result_version}_Test.csv")
        print(f"\nPlease modify the configuration in MANUAL_CONFIG and rerun.")
        return False
    
    if input_version and not result_version:
        print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Warning: input is MIMIC{input_version} but result file has no version prefix, please confirm this is the correct configuration")
    elif not input_version and result_version:
        print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Warning: input file has no MIMIC version identifier but result file has {result_version}_ prefix, please confirm this is the correct configuration")
    elif input_version and result_version and input_version == result_version:
        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Version matching: MIMIC{input_version}")
    
    # Validate result file
    for file_type, file_key in [("CSV result", "csv_file"), ("JSON log", "json_file"), ("TXT metrics", "txt_file")]:
        file_path = MANUAL_CONFIG[file_key]
        if not os.path.isabs(file_path):
            file_path = os.path.join(script_dir, file_path)
        
        if not os.path.exists(file_path):
            print(f"{Colors.RED}[ERROR]{Colors.RESET} {file_type} file does not exist: {file_path}")
            return False
        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} {file_type} file: {file_path}")
    
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} All configuration files validated successfully")
    return True

def filter_model_params(model_config: Dict) -> Dict:
    """
    Filter model configuration parameters, remove unsupported parameters
    Implement the same parameter filtering logic as prediction.py
    """
    filtered_config = model_config.copy()
    
    # Get model name
    model_name = model_config.get("model_name", "").lower()
    
    # Check if max_tokens parameter needs to be removed
    if "generation_params" in filtered_config:
        gen_params = filtered_config["generation_params"].copy()
        
        # Filter logic consistent with prediction.py
        # For gemini, claude, o1, o3 models, do not use max_tokens parameter
        if ("gemini" in model_name or "claude" in model_name or "o1" in model_name or "o3" in model_name):
            if "max_tokens" in gen_params:
                removed_max_tokens = gen_params.pop("max_tokens")
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} Model parameter filtering: remove max_tokens={removed_max_tokens} for {model_config.get('display_name', model_name)}")
        
        filtered_config["generation_params"] = gen_params
    
    return filtered_config

def extract_config_from_json(json_file_path: str) -> Tuple[Optional[Dict], Optional[str], Optional[str]]:
    """Extract model configuration, label and prompt mode from JSON log file"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if not os.path.isabs(json_file_path):
        json_file_path = os.path.join(script_dir, json_file_path)
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            logs = json.load(f)
        
        if not logs:
            print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} JSON file is empty")
            return None, None, None
        
        # Extract information from the first log entry
        first_log = logs[0]
        model_name = first_log.get('model', 'Unknown')
        prompt_mode = first_log.get('prompt_mode', 'DIRECTLY_PROMPTING')
        
        # Debug: display the extracted original model name
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Debug information:")
        print(f"Original model name: '{model_name}'")
        print(f"Model name lowercase: '{model_name.lower()}'")
        print(f"Contains 'claude': {'claude' in model_name.lower()}")
        print(f"Contains 'gemini': {'gemini' in model_name.lower()}")
        
        # Import the complete MODEL_CONFIG from prediction.py as the base
        try:
            prediction_module = import_prediction_module()
            base_config = prediction_module.MODEL_CONFIG.copy()
            
            # Only overwrite the specific fields extracted from JSON, do not perform model name mapping
            base_config["display_name"] = model_name
            
            # Apply model parameter filtering logic
            filtered_config = filter_model_params(base_config)
            
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} Using precise model configuration (apply parameter filtering):")
            print(f"Model display name: {model_name}")
            print(f"Keep original precise API name: {filtered_config.get('model_name', 'Unknown')}")
            print(f"API type: {filtered_config['api_type']}")
            print(f"Prompt mode: {prompt_mode}")
            print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Model parameter filtering logic applied")
            
            return filtered_config, None, prompt_mode  # label will be extracted from the file name
            
        except Exception as import_error:
            print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Failed to import prediction.py configuration, using base configuration: {str(import_error)}")
            
            # Fallback: use base configuration (without API key)
            model_config = {
                "display_name": model_name,
                "model_name": model_name.lower().replace("gpt-", "").replace("claude-", "").replace("gemini-", ""),
                "api_type": "openai",  # Default
                "generation_params": {
                    "temperature": 0.0,
                    "max_tokens": 500
                }
            }
            
            # Apply filtering logic to the base configuration
            filtered_config = filter_model_params(model_config)
            
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} Configuration extracted from JSON file (base mode + parameter filtering):")
            print(f"Model: {model_name}")
            print(f"Prompt mode: {prompt_mode}")
            print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Warning: missing API configuration, may cause call failure")
            
            return filtered_config, None, prompt_mode
        
    except Exception as e:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} Failed to extract JSON configuration: {str(e)}")
        return None, None, None

def extract_label_from_filename(filename: str) -> Optional[str]:
    """Extract label from file name"""
    # File name format: {version_prefix}{label}_predict_results_{model}_{prompt}_{id}.csv
    # For example: 3_DIEINHOSPITAL_predict_results_GPT_o1_directlyprompting_7a8e225b.csv
    
    basename = os.path.basename(filename)
    
    # Remove extension
    name_without_ext = basename.replace('.csv', '').replace('.json', '').replace('.txt', '')
    
    # Known label list
    known_labels = ['DIEINHOSPITAL', 'Readmission_30', 'Multiple_ICUs', 'sepsis_all', 
                   'FirstICU24_AKI_ALL', 'LOS_Hospital', 'ICU_within_12hr_of_admit']
    
    # Find matching label
    for label in known_labels:
        if label in name_without_ext:
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} Extracted label from file name: {label}")
            return label
    
    print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Failed to extract label from file name: {basename}")
    return None

def get_manual_file_paths() -> Tuple[str, str, str, str]:
    """Get file paths for manual mode (convert to absolute path)"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    input_csv = MANUAL_CONFIG["input_csv"]
    if not os.path.isabs(input_csv):
        input_csv = os.path.join(script_dir, input_csv)
    
    csv_file = MANUAL_CONFIG["csv_file"]
    if not os.path.isabs(csv_file):
        csv_file = os.path.join(script_dir, csv_file)
    
    json_file = MANUAL_CONFIG["json_file"]
    if not os.path.isabs(json_file):
        json_file = os.path.join(script_dir, json_file)
    
    txt_file = MANUAL_CONFIG["txt_file"]
    if not os.path.isabs(txt_file):
        txt_file = os.path.join(script_dir, txt_file)
    
    return input_csv, csv_file, json_file, txt_file

def import_prediction_module():
    """Dynamically import prediction module and its functions"""
    import importlib.util
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prediction_file = os.path.join(script_dir, 'prediction.py')
    spec = importlib.util.spec_from_file_location("prediction", prediction_file)
    prediction = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prediction)
    
    return prediction

# Determine whether to import the prediction module based on the mode
if not MANUAL_MODE:
    # Automatic mode: import prediction module
    prediction = import_prediction_module()
    predict_single_patient = prediction.predict_single_patient
    load_feature_descriptions = prediction.load_feature_descriptions
    load_feature_lists = prediction.load_feature_lists
    validate_label = prediction.validate_label
    evaluate_predictions = prediction.evaluate_predictions
    load_llm_cd_features_for_label = prediction.load_llm_cd_features_for_label  
    load_cd_optimized_features_for_label = prediction.load_cd_optimized_features_for_label  
    load_cd_filtered_features_for_label = prediction.load_cd_filtered_features_for_label  
    confirm_cd_optimized_config_with_mapping = prediction.confirm_cd_optimized_config_with_mapping  
    build_optimized_file_path = prediction.build_optimized_file_path  
    parse_optimized_features_file = prediction.parse_optimized_features_file  
    convert_features_desc_to_codes = prediction.convert_features_desc_to_codes  
    MODEL_CONFIG = prediction.MODEL_CONFIG
else:
    # Manual mode: delayed import, only import the required functions
    prediction = None
    predict_single_patient = None
    load_feature_descriptions = None
    load_feature_lists = None
    validate_label = None
    evaluate_predictions = None
    load_llm_cd_features_for_label = None  
    load_cd_optimized_features_for_label = None  
    load_cd_filtered_features_for_label = None  
    confirm_cd_optimized_config_with_mapping = None  
    build_optimized_file_path = None  
    parse_optimized_features_file = None  
    convert_features_desc_to_codes = None  
    MODEL_CONFIG = None

def load_input_data(csv_file_path: str) -> pd.DataFrame:
    """Load input CSV data"""
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"Input file does not exist: {csv_file_path}")
    
    # Explicitly specify the data type of the key columns to avoid type inference problems
    data = pd.read_csv(csv_file_path, dtype={
        'ICUSTAY_ID': str,
        'patient_id': str
        # Other columns keep default inference, because their names may change
    })
    
    # Ensure ICUSTAY_ID is a string type (this is the key ID column)
    if 'ICUSTAY_ID' in data.columns:
        data['ICUSTAY_ID'] = data['ICUSTAY_ID'].astype(str)
    if 'patient_id' in data.columns:
        data['patient_id'] = data['patient_id'].astype(str)
    
    print(f"Loaded input data: {len(data)} rows")
    return data

def load_existing_results(json_file_path: str) -> List[Dict]:
    """Load existing JSON result file"""
    if not os.path.exists(json_file_path):
        print(f"Result file does not exist: {json_file_path}")
        return []
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            existing_logs = json.load(f)
        print(f"Loaded existing results: {len(existing_logs)} records")
        return existing_logs
    except Exception as e:
        print(f"Error loading result file: {str(e)}")
        return []

def load_existing_csv_results(csv_file_path: str) -> pd.DataFrame:
    """Load existing CSV result file"""
    if not os.path.exists(csv_file_path):
        print(f"CSV result file does not exist: {csv_file_path}")
        return pd.DataFrame()
    
    try:
        # Explicitly specify the data type to avoid type inference problems
        csv_data = pd.read_csv(csv_file_path, dtype={
            'icustay_id': str,
            'patient_id': str
            # Other columns' types will be processed dynamically later, because the column names depend on the label
        })
        
        # Ensure the key columns are string types
        if 'patient_id' in csv_data.columns:
            csv_data['patient_id'] = csv_data['patient_id'].astype(str)
        if 'icustay_id' in csv_data.columns:
            csv_data['icustay_id'] = csv_data['icustay_id'].astype(str)
            
        print(f"Loaded CSV results: {len(csv_data)} rows")
        return csv_data
    except Exception as e:
        print(f"Error loading CSV result file: {str(e)}")
        return pd.DataFrame()

# Uniformly process data types, ensure consistency
def is_placeholder_id(patient_id_str):
    """Check if it is a placeholder ID (MISSING_, TIMEOUT_, EXCEPTION_, etc.)"""
    if not isinstance(patient_id_str, str):
        return False
    
    placeholder_prefixes = ['MISSING_', 'TIMEOUT_', 'EXCEPTION_', 'UNKNOWN_', 'RETRY_']
    return any(patient_id_str.startswith(prefix) for prefix in placeholder_prefixes)

def extract_index_from_placeholder(placeholder_str):
    """Extract the original index from the placeholder, if possible"""
    try:
        # Try to extract the numeric index from the placeholder
        match = re.search(r'_(\d+)$', placeholder_str)
        if match:
            return int(match.group(1))
    except:
        pass
    return None

def get_real_icustay_id_from_index(input_data, index):
    """Get the real ICUSTAY_ID from the input data based on the index"""
    try:
        if 0 <= index < len(input_data):
            row = input_data.iloc[index]
            return str(int(float(row['ICUSTAY_ID'])))
    except:
        pass
    return None

def safe_convert_to_patient_id(patient_id_raw, input_data=None):
    """
    Safely convert patient_id to standard format, handle placeholder cases
    Return: (converted_id, is_placeholder, original_index)
    """
    try:
        # First check if it is a null value
        if pd.isna(patient_id_raw):
            return None, False, None
        
        patient_id_str = str(patient_id_raw).strip()
        
        # Check if it is a placeholder
        if is_placeholder_id(patient_id_str):
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} Detected placeholder: {patient_id_str}")
            
            # Try to extract the index from the placeholder
            original_index = extract_index_from_placeholder(patient_id_str)
            if original_index is not None and input_data is not None:
                # Try to get the real ICUSTAY_ID
                real_id = get_real_icustay_id_from_index(input_data, original_index)
                if real_id:
                    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Successfully recovered real ID: {patient_id_str} -> {real_id} (index: {original_index})")
                    return real_id, True, original_index
                else:
                    print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Failed to recover real ID, index: {original_index}")
            else:
                print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Failed to extract index from placeholder: {patient_id_str}")
            
            # If it cannot be recovered, return the placeholder information
            return patient_id_str, True, original_index
        
        # Try to convert to a normal numeric format
        try:
            # Standard numeric ID conversion
            converted_id = str(int(float(patient_id_str)))
            return converted_id, False, None
        except ValueError:
            # If the conversion fails, it may be another format of placeholder
            print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Invalid patient_id format: {patient_id_str}")
            return patient_id_str, True, None
            
    except Exception as e:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} patient_id conversion exception: {str(e)}, original value: {patient_id_raw}")
        return None, False, None

def find_failed_tasks(input_data: pd.DataFrame, existing_logs: List[Dict], csv_data: pd.DataFrame) -> List[str]:
    """
    Find tasks that need to be retried
    Return the list of patient_id (format: ICUSTAY_ID)
    """
    # Get all patient_ids (ICUSTAY_ID) from the input data - convert to string format uniformly
    all_patient_ids = set()
    for _, row in input_data.iterrows():
        # Uniformly process data types, ensure consistency
        icustay_id = str(int(float(row['ICUSTAY_ID'])))  # Process possible floating point format
        patient_id = icustay_id
        all_patient_ids.add(patient_id)
    
    print(f"Input data contains {len(all_patient_ids)} unique records (ICUSTAY_ID)")
    
    # Add detailed CSV file status check
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} CSV file status check:")
    print(f"- Is CSV empty: {csv_data.empty}")
    if not csv_data.empty:
        print(f"- CSV row count: {len(csv_data)}")
        print(f"- CSV column names: {list(csv_data.columns)}")
        print(f"- Does CSV have patient_id column: {'patient_id' in csv_data.columns}")
        
        # Use the new safe conversion function to process CSV data
        if 'patient_id' in csv_data.columns:
            csv_patient_ids_raw = csv_data['patient_id'].values
            csv_patient_ids_processed = []
            csv_placeholder_count = 0
            csv_recovered_count = 0
            
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} Start processing patient_id in CSV...")
            
            for pid in csv_patient_ids_raw:
                converted_id, is_placeholder, original_index = safe_convert_to_patient_id(pid, input_data)
                if converted_id:
                    csv_patient_ids_processed.append(converted_id)
                    if is_placeholder:
                        csv_placeholder_count += 1
                        if original_index is not None:
                            csv_recovered_count += 1
            
            unique_csv_ids = len(set(csv_patient_ids_processed))
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} CSV processing results:")
            print(f"Valid ID count: {len(csv_patient_ids_processed)}")
            print(f"Unique ID count: {unique_csv_ids}")
            print(f"Placeholder count: {csv_placeholder_count}")
            print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Successfully recovered count: {csv_recovered_count}")
            
            if len(csv_patient_ids_processed) != unique_csv_ids:
                print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Detected duplicate ID!")
    
    # Get failed tasks in JSON - use the new safe conversion function
    failed_json_ids = set()
    json_patient_ids = set()  # Used to track all patient_ids in JSON
    json_placeholder_count = 0
    json_recovered_count = 0
    
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Start processing patient_id in JSON...")
    
    for log in existing_logs:
        if 'patient_id' in log:
            # Use the safe conversion function
            converted_id, is_placeholder, original_index = safe_convert_to_patient_id(log['patient_id'], input_data)
            
            if converted_id:
                # Update the patient_id in the log to standard format
                if is_placeholder:
                    json_placeholder_count += 1
                    if original_index is not None:
                        json_recovered_count += 1
                        # If the real ID is successfully recovered, update the log
                        if not is_placeholder_id(converted_id):
                            log['patient_id'] = converted_id
                            if 'icustay_id' in log:
                                log['icustay_id'] = converted_id
                
                json_patient_ids.add(converted_id)
                
                # Check if it is a failed task (answer is -1 or correctness is -1)
                if log.get('answer', 0) == -1 or log.get('correctness', 0) == -1:
                    failed_json_ids.add(converted_id)
                    print(f"{Colors.RED}[ERROR]{Colors.RESET} Found failed task in JSON: {converted_id}")
    
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} JSON processing results:")
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Valid ID count: {len(json_patient_ids)}")
    print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Placeholder count: {json_placeholder_count}")
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Successfully recovered count: {json_recovered_count}")
    
    # Get failed tasks in CSV - use the new safe conversion function
    failed_csv_ids = set()
    csv_patient_ids = set()  # Used to track all patient_ids in CSV
    
    if not csv_data.empty and 'patient_id' in csv_data.columns:
        # Find the prediction column
        pred_col = None
        prob_col = None
        for col in csv_data.columns:
            if 'prediction' in col.lower():
                pred_col = col
            if 'probability' in col.lower():
                prob_col = col
        
        # Use the safe conversion function to process CSV data
        seen_patient_ids = set()
        for _, row in csv_data.iterrows():
            converted_id, is_placeholder, original_index = safe_convert_to_patient_id(row['patient_id'], input_data)
            
            if converted_id:
                csv_patient_ids.add(converted_id)
                
                # Check for duplicates
                if converted_id in seen_patient_ids:
                    print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Detected duplicate patient_id: {converted_id}")
                    continue
                seen_patient_ids.add(converted_id)
                
                # Check if it is a failed task
                if pred_col and prob_col:
                    if pd.isna(row[pred_col]) or pd.isna(row[prob_col]) or row[pred_col] == -1 or row[prob_col] == -1:
                        failed_csv_ids.add(converted_id)
                        print(f"{Colors.RED}[ERROR]{Colors.RESET} Found failed task in CSV: {converted_id}")
    
    # Comprehensive judgment on tasks that need to be retried
    retry_ids = set()
    
    # 1. Input has but JSON has none
    missing_in_json = all_patient_ids - json_patient_ids
    retry_ids.update(missing_in_json)
    if missing_in_json:
        print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Found missing tasks in JSON: {len(missing_in_json)}")
        if len(missing_in_json) <= 10:  # If the number is not many, display the specific ID
            print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Missing ID: {sorted(list(missing_in_json))}")
    
    # 2. Failed tasks in JSON
    retry_ids.update(failed_json_ids)
    
    # 3. Failed tasks in CSV
    retry_ids.update(failed_csv_ids)
    
    # 4. Input has but CSV has none - use the corrected data type to process
    if not csv_data.empty and 'patient_id' in csv_data.columns:
        missing_in_csv = all_patient_ids - csv_patient_ids
        retry_ids.update(missing_in_csv)
        if missing_in_csv:
            print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Found missing tasks in CSV: {len(missing_in_csv)}")
            if len(missing_in_csv) <= 10:  # If the number is not many, display the specific ID
                print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Missing ID: {sorted(list(missing_in_csv))}")
    else:
        # Safe handling when CSV file is abnormal
        print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} CSV file status is abnormal!")
        if csv_data.empty:
            print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} CSV file is empty")
        elif 'patient_id' not in csv_data.columns:
            print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} CSV file is missing patient_id column")
        
        # Safe strategy: do not automatically retry all tasks, but only retry failed tasks in JSON
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Safe mode: only retry failed tasks in JSON, do not retry all data set")
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} If you need to retry all tasks, please manually delete the CSV file and re-run")
    
    retry_list = sorted(list(retry_ids))
    print(f"\nTotal number of tasks to be retried: {len(retry_list)}")
    
    # Add detailed statistics
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Detailed statistics:")
    print(f"- Total number of input data: {len(all_patient_ids)}")
    print(f"- JSON record count: {len(json_patient_ids)}")
    print(f"- CSV record count: {len(csv_patient_ids)}")
    print(f"- JSON failed count: {len(failed_json_ids)}")
    print(f"- CSV failed count: {len(failed_csv_ids)}")
    print(f"- JSON missing count: {len(missing_in_json) if 'missing_in_json' in locals() else 0}")
    print(f"- CSV missing count: {len(missing_in_csv) if 'missing_in_csv' in locals() else 0}")
    
    # Add safety check: if the number of retry tasks is too large, give a warning
    if len(retry_list) > len(all_patient_ids) * 0.8:  # If the number of retry tasks exceeds 80%
        print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Warning: the number of retry tasks is too large ({len(retry_list)}/{len(all_patient_ids)})!")
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} This may indicate that the CSV file has problems, please check the file status.")
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Continuing execution may reprocess a large number of completed tasks.")
    
    return retry_list

def retry_failed_predictions_batch_deprecated(
    input_data: pd.DataFrame,
    retry_patient_ids: List[str],
    model_config: Dict,
    feature_descriptions: Dict,
    feature_lists: Dict,
    label: str,
    max_workers: int = 10
) -> List[Dict]:
    """
    Retry failed prediction tasks
    """
    print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Start retrying {len(retry_patient_ids)} failed tasks (max_workers={max_workers})...")
    
    if not retry_patient_ids:
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} No tasks to retry")
        return []

    # Prepare data for retry tasks
    retry_tasks = []
    retry_patient_id_set = set(retry_patient_ids)
    
    for idx, row in input_data.iterrows():
        # Uniformly process data types, consistent with find_failed_tasks
        icustay_id = str(int(float(row['ICUSTAY_ID'])))
        patient_id = icustay_id
        if patient_id in retry_patient_id_set:
            groundtruth = int(row[label])
            retry_tasks.append((idx, row, groundtruth))
    
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Prepare to retry {len(retry_tasks)} tasks")
    
    if not retry_tasks:
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} No matching retry task data found")
        return []
    
    # Parallel retry tasks (add complete timeout control)
    retry_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit retry tasks
        futures = []
        for task_data in retry_tasks:
            future = executor.submit(
                predict_single_patient, 
                task_data, 
                model_config, 
                feature_descriptions, 
                label, 
                feature_lists
            )
            futures.append(future)
        
        # Collect retry results (add timeout control)
        completed_count = 0
        total_futures = len(futures)
        
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Waiting for {total_futures} retry tasks to complete...")
        
        # Timeout control parameters
        start_time = time.time()
        global_timeout = 600  # Use a shorter 10-minute global timeout when retrying
        single_task_timeout = 120  # Single task timeout is 2 minutes
        
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Set global timeout: {global_timeout//60} minutes, single task timeout: {single_task_timeout} seconds")
        
        try:
            for future in as_completed(futures, timeout=global_timeout):
                # Check global timeout
                elapsed_time = time.time() - start_time
                if elapsed_time > global_timeout:
                    print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Global timeout ({global_timeout//60} minutes), force end...")
                    break
                
                try:
                    result = future.result(timeout=single_task_timeout)  # Single task timeout control
                    completed_count += 1
                    retry_results.append(result)
                    
                    # Display progress and time information
                    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Retry progress: {completed_count}/{total_futures} (time: {elapsed_time:.1f}s)")
                    
                except Exception as e:
                    completed_count += 1
                    print(f"{Colors.RED}[ERROR]{Colors.RESET} Exception occurred when retrying task: {str(e)}")
                    # Create default result for failed tasks
                    retry_results.append({
                        'idx': completed_count - 1,
                        'icustay_id': f"RETRY_UNKNOWN_{completed_count-1}",
                        'patient_id': f"RETRY_UNKNOWN_{completed_count-1}",
                        'probability': -1,
                        'prediction': -1,
                        'groundtruth': -1,
                        'experiment_log': {
                            "patient_id": f"RETRY_UNKNOWN_{completed_count-1}",
                            "icustay_id": f"RETRY_UNKNOWN_{completed_count-1}",
                            "model": model_config["display_name"],
                            "prompt_mode": model_config.get("prompt_mode", "DIRECTLY_PROMPTING"),
                            "input": "RETRY_FUTURE_EXCEPTION",
                            "response": f"RETRY_FUTURE_EXCEPTION: {str(e)}",
                            "answer": -1,
                            "groundtruth": -1,
                            "correctness": -1
                        }
                    })
        except Exception as e:
            print(f"{Colors.RED}[ERROR]{Colors.RESET} as_completed loop exception: {str(e)}")
            
        # Check if there are any unfinished tasks
        remaining_futures = [f for f in futures if not f.done()]
        if remaining_futures:
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} Found {len(remaining_futures)} unfinished retry tasks, trying to force process...")
            
            for i, future in enumerate(futures):
                if not future.done():
                    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Retry task {i+1} is still running, trying to get result...")
                    try:
                        result = future.result(timeout=30)  # Give the last 30 seconds chance
                        retry_results.append(result)
                        completed_count += 1
                        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Retry task {i+1} finally completed")
                    except Exception as fe:
                        print(f"{Colors.RED}[ERROR]{Colors.RESET} Retry task {i+1} finally failed: {str(fe)}")
                        completed_count += 1
                        # Create default result for failed tasks
                        retry_results.append({
                            'idx': i,
                            'icustay_id': f"RETRY_TIMEOUT_{i}",
                            'patient_id': f"RETRY_TIMEOUT_{i}",
                            'probability': -1,
                            'prediction': -1,
                            'groundtruth': -1,
                            'experiment_log': {
                                "patient_id": f"RETRY_TIMEOUT_{i}",
                                "icustay_id": f"RETRY_TIMEOUT_{i}",
                                "model": model_config["display_name"],
                                "prompt_mode": model_config.get("prompt_mode", "DIRECTLY_PROMPTING"),
                                "input": "RETRY_TIMEOUT_OR_EXCEPTION",
                                "response": f"RETRY_TIMEOUT_OR_EXCEPTION: {str(fe)}",
                                "answer": -1,
                                "groundtruth": -1,
                                "correctness": -1
                            }
                        })
        
        # Final status report
        total_elapsed = time.time() - start_time
        successful_tasks = len([r for r in retry_results if r.get('probability', -1) != -1])
        failed_tasks = len(retry_results) - successful_tasks
        
        print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Retry task completion summary:")
        print(f"Retry task count: {total_futures}")
        print(f"Processed tasks: {completed_count}")
        print(f"Collected results: {len(retry_results)}")
        print(f"Successful tasks: {successful_tasks}")
        print(f"Failed tasks: {failed_tasks}")
        print(f"Total time: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)")
        
        if len(retry_results) < total_futures:
            missing_count = total_futures - len(retry_results)
            print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Missing {missing_count} retry results, creating default results")
            
            # Create default results for missing tasks
            existing_indices = {r['idx'] for r in retry_results}
            for i in range(total_futures):
                if i not in existing_indices:
                    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Creating default result for retry task {i+1}")
                    retry_results.append({
                        'idx': i,
                        'icustay_id': f"RETRY_MISSING_{i}",
                        'patient_id': f"RETRY_MISSING_{i}",
                        'probability': -1,
                        'prediction': -1,
                        'groundtruth': -1,
                        'experiment_log': {
                            "patient_id": f"RETRY_MISSING_{i}",
                            "icustay_id": f"RETRY_MISSING_{i}",
                            "model": model_config["display_name"],
                            "prompt_mode": model_config.get("prompt_mode", "DIRECTLY_PROMPTING"),
                            "input": "RETRY_MISSING_RESULT",
                            "response": "RETRY_MISSING_RESULT",
                            "answer": -1,
                            "groundtruth": -1,
                            "correctness": -1
                        }
                    })
    
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Retry completed, obtained {len(retry_results)} results")
    return retry_results

def update_results_files_batch_deprecated(
    retry_results: List[Dict],
    existing_logs: List[Dict],
    csv_file_path: str,
    json_file_path: str,
    label: str
) -> None:
    """
    Update result files, merge retry results into existing results
    """
    print(f"\nStart updating result files...")
    
    # Input data integrity check
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Check the quality of the retry results...")
    valid_retry_count = 0
    placeholder_retry_count = 0
    
    for result in retry_results:
        patient_id = result.get('patient_id', '')
        if is_placeholder_id(str(patient_id)):
            placeholder_retry_count += 1
        else:
            valid_retry_count += 1
    
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Retry result statistics: valid ID={valid_retry_count}, placeholder ID={placeholder_retry_count}")
    
    if placeholder_retry_count > 0:
        print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Warning: there are still {placeholder_retry_count} placeholder IDs in the retry results")
        print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} These placeholders may indicate abnormal situations during the retry process")
    
    # 1. Update JSON log file - uniform data type processing and placeholder processing
    # Create a mapping from patient_id to log (patient_id -> log)
    existing_logs_dict = {}
    json_placeholder_count = 0
    
    for log in existing_logs:
        if 'patient_id' in log:
            # Use safe conversion function
            original_pid = log['patient_id']
            converted_id, is_placeholder, original_index = safe_convert_to_patient_id(original_pid)
            
            if converted_id:
                # Update patient_id in the log to the standard format (if possible)
                if is_placeholder:
                    json_placeholder_count += 1
                    # If it is a placeholder but the real ID is recovered, update it
                    if not is_placeholder_id(converted_id):
                        log['patient_id'] = converted_id
                        if 'icustay_id' in log:
                            log['icustay_id'] = converted_id
                        print(f"{Colors.WHITE}[INFO]{Colors.RESET} JSON log repair: {original_pid} -> {converted_id}")
                else:
                    # Ensure data type consistency
                    log['patient_id'] = converted_id
                    if 'icustay_id' in log:
                        log['icustay_id'] = converted_id
                
                existing_logs_dict[converted_id] = log
    
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Existing JSON record processing: total={len(existing_logs_dict)}, placeholder={json_placeholder_count}")
    
    # Update or add logs using retry results
    updated_count = 0
    added_count = 0
    retry_placeholder_count = 0
    
    for result in retry_results:
        # Process patient_id in retry results
        original_pid = result['patient_id']
        converted_id, is_placeholder, original_index = safe_convert_to_patient_id(original_pid)
        
        if not converted_id:
            print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Skip invalid retry results: {original_pid}")
            continue
        
        if is_placeholder:
            retry_placeholder_count += 1
        
        new_log = result['experiment_log'].copy()
        
        # Ensure patient_id consistency in the new log
        new_log['patient_id'] = converted_id
        new_log['icustay_id'] = converted_id
        
        if converted_id in existing_logs_dict:
            # Update existing log
            existing_logs_dict[converted_id] = new_log
            updated_count += 1
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} Update JSON record: {converted_id}")
        else:
            # Add new log
            existing_logs_dict[converted_id] = new_log
            added_count += 1
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} Add new JSON record: {converted_id}")
    
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Retry result processing: update={updated_count}, add={added_count}, placeholder={retry_placeholder_count}")
    
    # Safe save JSON file (add backup mechanism)
    try:
        # If the original file exists, create a backup first
        if os.path.exists(json_file_path):
            backup_json = json_file_path + '.backup'
            import shutil
            shutil.copy2(json_file_path, backup_json)
            print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} JSON backup created: {backup_json}")
        
        # Save the updated JSON file
        updated_logs = list(existing_logs_dict.values())
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(updated_logs, f, ensure_ascii=False, indent=2)
        
        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} JSON file updated: updated {updated_count} records, added {added_count} records, total {len(updated_logs)} records")
    except Exception as e:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} JSON file update failed: {str(e)}")
        # If the backup exists, try to restore
        backup_json = json_file_path + '.backup'
        if os.path.exists(backup_json):
            shutil.copy2(backup_json, json_file_path)
            print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} JSON file restored from backup")
        raise
    
    # 2. Update CSV result file - completely fix the problem of duplicate addition and placeholder processing
    try:
        # Detailed check of the existing CSV file status
        existing_csv_dict = {}
        csv_backup_created = False
        csv_placeholder_count = 0
        
        if os.path.exists(csv_file_path):
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} Loading existing CSV file...")
            # Create CSV backup first
            backup_csv = csv_file_path + '.backup'
            import shutil
            shutil.copy2(csv_file_path, backup_csv)
            csv_backup_created = True
            print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} CSV backup created: {backup_csv}")
            
            # Load existing CSV data, explicitly specify data types
            existing_csv = pd.read_csv(csv_file_path, dtype={
                'icustay_id': str,
                'patient_id': str
                # Other column types will be determined in subsequent processing
            })
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} Existing CSV status: {len(existing_csv)} rows, columns: {list(existing_csv.columns)}")
            
            # Use safe conversion function to process CSV data
            if 'patient_id' in existing_csv.columns:
                # Ensure the key column is a string type
                existing_csv['patient_id'] = existing_csv['patient_id'].astype(str)
                if 'icustay_id' in existing_csv.columns:
                    existing_csv['icustay_id'] = existing_csv['icustay_id'].astype(str)
                
                seen_patient_ids = set()
                duplicate_entries = []
                
                for idx, row in existing_csv.iterrows():
                    original_pid = row['patient_id']
                    converted_id, is_placeholder, original_index = safe_convert_to_patient_id(original_pid)
                    
                    if converted_id:
                        if is_placeholder:
                            csv_placeholder_count += 1
                        
                        if converted_id in seen_patient_ids:
                            duplicate_entries.append((idx, converted_id))
                            print(f"{Colors.RED}[ERROR]{Colors.RESET} Detected duplicate patient_id: {converted_id} (row {idx})")
                        else:
                            seen_patient_ids.add(converted_id)
                            row_dict = row.to_dict()
                            row_dict['patient_id'] = str(converted_id)  # Ensure string type
                            existing_csv_dict[converted_id] = row_dict
                
                if duplicate_entries:
                    print(f"\n{Colors.RED}[ERROR]{Colors.RESET} CSV duplicate check: detected {len(duplicate_entries)} duplicate patient_id!")
                    for idx, pid in duplicate_entries:
                        print(f"- row {idx}: patient_id {pid}")
                    print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Skipped duplicate records, only the first occurrence is retained")
                
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} Loaded CSV record count (after deduplication): {len(existing_csv_dict)}, placeholder: {csv_placeholder_count}")
                
            # Compatible with old format, if there is no patient_id column but there is an icustay_id column
            elif 'icustay_id' in existing_csv.columns:
                print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Detected old format CSV, converting...")
                # Ensure the icustay_id column is a string type
                existing_csv['icustay_id'] = existing_csv['icustay_id'].astype(str)
                
                for _, row in existing_csv.iterrows():
                    original_id = row['icustay_id']
                    converted_id, is_placeholder, original_index = safe_convert_to_patient_id(original_id)
                    
                    if converted_id:
                        if is_placeholder:
                            csv_placeholder_count += 1
                        row_dict = row.to_dict()
                        row_dict['patient_id'] = str(converted_id)  # Add patient_id, ensure string type
                        existing_csv_dict[converted_id] = row_dict
            else:
                print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} CSV file has neither patient_id column nor icustay_id column")
        else:
            print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} CSV file does not exist, will create a new file")
        
        # Replace existing data with retry results, handle placeholder cases
        csv_updated_count = 0
        csv_added_count = 0
        csv_retry_placeholder_count = 0
        
        pred_col = f'{label.lower()}_prediction'
        prob_col = f'{label.lower()}_probability'
        
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Start processing {len(retry_results)} retry results...")
        
        for result in retry_results:
            # Process patient_id in retry results
            original_pid = result['patient_id']
            converted_id, is_placeholder, original_index = safe_convert_to_patient_id(original_pid)
            
            if not converted_id:
                print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Skip invalid retry results: {original_pid}")
                continue
            
            if is_placeholder:
                csv_retry_placeholder_count += 1
            
            # Try to get the real ICUSTAY_ID (if it is a placeholder)
            icustay_id = converted_id
            
            # Create a CSV row, explicitly convert all values to string type
            csv_row = {
                'icustay_id': str(icustay_id),
                'patient_id': str(converted_id),
                pred_col: int(result['prediction']),
                prob_col: float(result['probability']),
                'ground_truth': int(result['groundtruth'])
            }
            
            if converted_id in existing_csv_dict:
                # Here it is replaced, not added
                existing_csv_dict[converted_id] = csv_row
                csv_updated_count += 1
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} Replace CSV record: {converted_id}")
            else:
                # Only add new records when patient_id does not exist
                existing_csv_dict[converted_id] = csv_row
                csv_added_count += 1
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} Add new CSV record: {converted_id}")
        
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Retry result CSV processing: update={csv_updated_count}, add={csv_added_count}, placeholder={csv_retry_placeholder_count}")
        
        # Safe save the updated CSV file
        updated_csv_data = list(existing_csv_dict.values())
        if not updated_csv_data:
            print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} The updated CSV data is empty!")
            return
            
        updated_csv_df = pd.DataFrame(updated_csv_data)
        
        # Final verification should not have duplicate patient_id
        if 'patient_id' in updated_csv_df.columns:
            # Ensure patient_id is a string type to avoid type warnings
            updated_csv_df['patient_id'] = updated_csv_df['patient_id'].astype(str)
            if 'icustay_id' in updated_csv_df.columns:
                updated_csv_df['icustay_id'] = updated_csv_df['icustay_id'].astype(str)
                
            unique_patient_ids = updated_csv_df['patient_id'].nunique()
            total_rows = len(updated_csv_df)
            if unique_patient_ids != total_rows:
                print(f"{Colors.RED}[ERROR]{Colors.RESET} Serious error: duplicate patient_id appears in CSV after retry operation!")
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} Unique ID count: {unique_patient_ids}, total rows: {total_rows}")
                
                # Display duplicate patient_id
                duplicate_ids = updated_csv_df[updated_csv_df.duplicated(subset=['patient_id'], keep=False)]['patient_id'].unique()
                print(f"{Colors.RED}[ERROR]{Colors.RESET} Duplicate patient_id: {list(duplicate_ids)}")
                
                # Temporary emergency repair, but needs investigation
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} Execute emergency deduplication (keep the last occurrence)...")
                updated_csv_df = updated_csv_df.drop_duplicates(subset=['patient_id'], keep='last')
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} After emergency deduplication, the number of rows: {len(updated_csv_df)}")
                print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Please check the retry logic, it should not produce duplicates!")
            else:
                print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Verification passed: no duplicate patient_id")
                
            # Count the placeholder situation in the final CSV
            final_placeholder_count = 0
            final_valid_count = 0
            for pid in updated_csv_df['patient_id']:
                if is_placeholder_id(str(pid)):
                    final_placeholder_count += 1
                else:
                    final_valid_count += 1
            
            print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Final CSV data quality: valid ID={final_valid_count}, placeholder ID={final_placeholder_count}")
            if final_placeholder_count > 0:
                placeholder_rate = final_placeholder_count / len(updated_csv_df) * 100
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} Placeholder ratio: {placeholder_rate:.1f}%")
        
        # Ensure column order
        column_order = ['icustay_id', 'patient_id', pred_col, prob_col, 'ground_truth']
        # Ensure all columns exist
        for col in column_order:
            if col not in updated_csv_df.columns:
                updated_csv_df[col] = None
        
        # Verify data integrity
        if len(updated_csv_df) == 0:
            raise ValueError(f"{Colors.RED}[ERROR]{Colors.RESET} The updated CSV data frame is empty")
        if 'patient_id' not in updated_csv_df.columns:
            raise ValueError(f"{Colors.RED}[ERROR]{Colors.RESET} The updated CSV is missing the patient_id column")
        
        # Sort by patient_id to ensure consistency (handle placeholders)
        def sort_key(pid):
            """Custom sort key, numeric ID first, placeholder last"""
            try:
                return (0, float(pid))  # Numeric ID
            except:
                return (1, pid)  # Placeholder ID
        
        updated_csv_df['_sort_key'] = updated_csv_df['patient_id'].apply(sort_key)
        updated_csv_df = updated_csv_df.sort_values('_sort_key').drop('_sort_key', axis=1).reset_index(drop=True)
        
        # Save CSV file
        updated_csv_df = updated_csv_df[column_order]
        updated_csv_df.to_csv(csv_file_path, index=False)
        
        # Verify the saved file, explicitly specify data types
        verification_df = pd.read_csv(csv_file_path, dtype={
            'icustay_id': str,
            'patient_id': str
            # Other column types remain default
        })
        if len(verification_df) != len(updated_csv_df):
            raise ValueError(f"{Colors.RED}[ERROR]{Colors.RESET} CSV file save verification failed: expected {len(updated_csv_df)} rows, actual {len(verification_df)} rows")
        
        # Final verification should not have duplicates
        final_unique_count = verification_df['patient_id'].nunique()
        final_total_count = len(verification_df)
        if final_unique_count != final_total_count:
            raise ValueError(f"{Colors.RED}[ERROR]{Colors.RESET} CSV save verification failed: duplicate patient_id detected, unique ID count {final_unique_count}, total rows {final_total_count}")
        
        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} CSV file updated: updated {csv_updated_count} rows, added {csv_added_count} rows, total {len(updated_csv_df)} rows")
        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Final verification: unique patient_id count = total rows = {final_unique_count} ✓")
        
        # If everything is normal, delete the backup file
        if csv_backup_created:
            backup_csv = csv_file_path + '.backup'
            if os.path.exists(backup_csv):
                os.remove(backup_csv)
                print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} CSV backup file deleted")
                
    except Exception as e:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} CSV file update failed: {str(e)}")
        # If the backup exists, try to restore
        backup_csv = csv_file_path + '.backup'
        if csv_backup_created and os.path.exists(backup_csv):
            shutil.copy2(backup_csv, csv_file_path)
            print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} CSV file restored from backup")
        raise

def import_required_functions_manual():
    """Import necessary functions dynamically in manual mode"""
    global predict_single_patient, load_feature_descriptions, load_feature_lists, validate_label, evaluate_predictions, load_llm_cd_features_for_label, load_cd_optimized_features_for_label, load_cd_filtered_features_for_label, confirm_cd_optimized_config_with_mapping, build_optimized_file_path, parse_optimized_features_file, convert_features_desc_to_codes
    
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Manual mode: dynamically import necessary functions...")
    
    try:
        prediction = import_prediction_module()
        predict_single_patient = prediction.predict_single_patient
        load_feature_descriptions = prediction.load_feature_descriptions
        load_feature_lists = prediction.load_feature_lists
        validate_label = prediction.validate_label
        evaluate_predictions = prediction.evaluate_predictions
        load_llm_cd_features_for_label = prediction.load_llm_cd_features_for_label  # New LLM causal feature loading function
        load_cd_optimized_features_for_label = prediction.load_cd_optimized_features_for_label  # New CD optimized feature loading function
        load_cd_filtered_features_for_label = prediction.load_cd_filtered_features_for_label  # New CD filtered feature loading function
        confirm_cd_optimized_config_with_mapping = prediction.confirm_cd_optimized_config_with_mapping  # New CD optimized configuration confirmation function
        build_optimized_file_path = prediction.build_optimized_file_path  # New file path building function
        parse_optimized_features_file = prediction.parse_optimized_features_file  # New feature file parsing function
        convert_features_desc_to_codes = prediction.convert_features_desc_to_codes  # New feature mapping function
        
        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Successfully imported necessary functions from prediction.py")
        return True
    except Exception as e:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} Function import failed: {str(e)}")
        return False

def find_existing_result_files(script_dir: str, model_config: Dict, label: str, prompt_mode: str, version_prefix: str = '') -> tuple:
    """
    Find existing result files, if not found, create a new file name
    Return: (csv_file, json_file, txt_file, unique_id, is_existing)
    """
    results_dir = os.path.join(script_dir, 'results')
    
    # Generate file name pattern
    model_short_name = model_config["display_name"].replace(".", "_").replace("-", "_")
    prompt_mode_short = prompt_mode.replace("_", "").lower()
    
    # Search existing files (matching pattern: {version_prefix}{label}_*_{model_short_name}_{prompt_mode_short}_*.{ext})
    pattern_base = f'{version_prefix}{label}_*_{model_short_name}_{prompt_mode_short}_*'
    
    existing_csv = None
    existing_json = None
    existing_txt = None
    existing_unique_id = None
    
    if os.path.exists(results_dir):
        for filename in os.listdir(results_dir):
            if filename.startswith(f'{version_prefix}{label}_predict_results_{model_short_name}_{prompt_mode_short}_') and filename.endswith('.csv'):
                existing_csv = os.path.join(results_dir, filename)
                # Extract unique_id
                existing_unique_id = filename.split('_')[-1].replace('.csv', '')
            elif filename.startswith(f'{version_prefix}{label}_experiment_logs_{model_short_name}_{prompt_mode_short}_') and filename.endswith('.json'):
                existing_json = os.path.join(results_dir, filename)
            elif filename.startswith(f'{version_prefix}{label}_metrics_{model_short_name}_{prompt_mode_short}_') and filename.endswith('.txt'):
                existing_txt = os.path.join(results_dir, filename)
    
    # If existing files are found, use the existing unique_id
    if existing_csv and existing_json and existing_unique_id:
        csv_file = existing_csv
        json_file = existing_json
        txt_file = existing_txt if existing_txt else os.path.join(results_dir, f'{version_prefix}{label}_metrics_{model_short_name}_{prompt_mode_short}_{existing_unique_id}.txt')
        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Existing result files found, will update existing file (unique_id: {existing_unique_id})")
        return csv_file, json_file, txt_file, existing_unique_id, True
    else:
        # Generate new unique_id and file name
        unique_id = str(uuid.uuid4())[:8]
        os.makedirs(results_dir, exist_ok=True)
        csv_file = os.path.join(results_dir, f'{version_prefix}{label}_predict_results_{model_short_name}_{prompt_mode_short}_{unique_id}.csv')
        json_file = os.path.join(results_dir, f'{version_prefix}{label}_experiment_logs_{model_short_name}_{prompt_mode_short}_{unique_id}.json')
        txt_file = os.path.join(results_dir, f'{version_prefix}{label}_metrics_{model_short_name}_{prompt_mode_short}_{unique_id}.txt')
        print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} No existing result files found, will create a new file (unique_id: {unique_id})")
        return csv_file, json_file, txt_file, unique_id, False

def main():
    """Main function - supports automatic and manual modes"""
    print("="*60)
    print("Retry failed prediction tasks")
    print("="*60)
    
    if MANUAL_MODE:
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Running mode: manual mode")
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Use specified historical experiment files for retry")
        return main_manual_mode()
    else:
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Running mode: automatic mode") 
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Use the current configuration of prediction.py")
        return main_auto_mode()

def main_manual_mode():
    """Manual mode main function"""
    print("\n" + "="*50)
    print("Manual mode execution process")
    print("="*50)
    
    try:
        # 1. Validate configuration file
        print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Step 1: validate manual configuration")
        if not validate_manual_config():
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Configuration validation failed, please check the file path")
            return False
        
        # 2. Get file paths
        input_csv, csv_file, json_file, txt_file = get_manual_file_paths()
        
        # 3. Extract configuration information
        print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Step 2: extract configuration information")
        
        # Extract model configuration from JSON file
        extracted_model_config, _, extracted_prompt_mode = extract_config_from_json(json_file)
        
        # Extract label from file name
        extracted_label = extract_label_from_filename(csv_file)
        
        # Use manual configuration override settings (if any)
        raw_model_config = MANUAL_CONFIG["override_model_config"] or extracted_model_config
        label = MANUAL_CONFIG["override_label"] or extracted_label
        prompt_mode = MANUAL_CONFIG["override_prompt_mode"] or extracted_prompt_mode
        
        # Apply parameter filtering logic to manual configuration
        if raw_model_config:
            model_config = filter_model_params(raw_model_config)
            print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Manual configuration parameter filtering completed")
        else:
            model_config = raw_model_config
        
        if not model_config or not label or not prompt_mode:
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Unable to extract complete configuration information")
            print(f"Model configuration: {f'{Colors.GREEN}[SUCCESS]{Colors.RESET}' if model_config else f'{Colors.RED}[ERROR]{Colors.RESET}'}")
            print(f"Label: {f'{Colors.GREEN}[SUCCESS]{Colors.RESET}' if label else f'{Colors.RED}[ERROR]{Colors.RESET}'}")
            print(f"Prompt mode: {f'{Colors.GREEN}[SUCCESS]{Colors.RESET}' if prompt_mode else f'{Colors.RED}[ERROR]{Colors.RESET}'}")
            return False
        
        # 3. Configuration consistency check
        print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Step 3: configuration consistency check")
        try:
            is_match, reports = validate_manual_config_against_files(MANUAL_CONFIG, csv_file, json_file, txt_file)
            
            # Display check reports
            for report in reports:
                print(report)
            
            print("\n" + "="*50)
            
            if not is_match:
                print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Configuration mismatch detected!")
                print("\nPlease choose:")
                print("1. Correct the manual configuration and re-run")
                print("2. Ignore the warning and continue execution")
                print("3. Exit the program")
                
                while True:
                    choice = input("\nPlease enter your choice (1/2/3): ").strip()
                    if choice == '1':
                        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Please correct the configuration and re-run the script")
                        return False
                    elif choice == '2':
                        print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Ignore the warning and continue execution...")
                        break
                    elif choice == '3':
                        print(f"{Colors.WHITE}[INFO]{Colors.RESET} User chose to exit")
                        return False
                    else:
                        print("Please enter a valid choice (1/2/3)")
            else:
                print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} All configuration checks passed!")
                
        except Exception as e:
            print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Configuration check error: {str(e)}, please check and confirm")
        
        # 4. Dynamically import necessary functions
        print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Step 4: import necessary functions")
        if not import_required_functions_manual():
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Function import failed")
            return False
        
        # 5. Display configuration information
        print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Step 5: display final configuration information")
        print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Manual mode configuration information:")
        print(f"Input file: {input_csv}")
        print(f"CSV result file: {csv_file}")
        print(f"JSON log file: {json_file}")
        print(f"TXT metrics file: {txt_file}")
        print(f"Prediction label: {label}")
        print(f"Prompt mode: {prompt_mode}")
        print(f"Using model: {model_config['display_name']}")
        
        # Generate manual mode task specific identifier
        import hashlib
        manual_task_signature = f"{os.path.basename(csv_file)}_{label}_{model_config['display_name']}_manual"
        manual_unique_id = hashlib.md5(manual_task_signature.encode()).hexdigest()[:8]
        
        # Execute retry workflow
        return execute_retry_workflow(
            input_csv=input_csv,
            csv_file=csv_file, 
            json_file=json_file,
            txt_file=txt_file,
            model_config=model_config,
            label=label,
            prompt_mode=prompt_mode,
            unique_id=manual_unique_id,
            mode="manual"
        )
        
    except Exception as e:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} Manual mode execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main_auto_mode():
    """Automatic mode main function (original logic)"""
    print("\n" + "="*50)
    print("Automatic mode execution process") 
    print("="*50)
    
    try:
        # Configuration file path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_filename = 'your_data_file.csv'  # Keep consistent with prediction.py, here modify the data file
        input_csv = os.path.join(script_dir, data_filename)
        
        # Automatically detect version prefix based on data file name
        if 'MIMIC3' in data_filename:
            version_prefix = '3_'
        elif 'MIMIC4' in data_filename:
            version_prefix = '4_'
        else:
            version_prefix = ''  # If not recognized, no prefix is added
        
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Detected data file: {data_filename}")
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Version prefix: {version_prefix if version_prefix else 'No prefix'}")
        
        # Automatic mode also adds version consistency check
        input_version = None
        if 'MIMIC3' or 'MIMICIII' in data_filename:
            input_version = '3'
        elif 'MIMIC4' or 'MIMICIV' in data_filename:
            input_version = '4'
        
        if input_version and version_prefix and not version_prefix.startswith(input_version):
            print(f"\n{Colors.RED}[ERROR]{Colors.RESET} Automatic mode configuration error: version mismatch!")
            print(f"Input file: {data_filename} (MIMIC{input_version})")
            print(f"Generated version prefix: {version_prefix}")
            print(f"Please check data_filename configuration")
            return False
        
        # Get model configuration and apply parameter filtering
        raw_model_config = MODEL_CONFIG
        model_config = filter_model_params(raw_model_config)
        label = model_config.get('label', 'DIEINHOSPITAL')
        prompt_mode = model_config.get('prompt_mode', 'DIRECTLY_PROMPTING')
        
        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Automatic mode parameter filtering completed")
        
        # Find existing result files or create new file name
        csv_file, json_file, txt_file, unique_id, is_existing = find_existing_result_files(
            script_dir, model_config, label, prompt_mode, version_prefix
        )
        
        print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Automatic mode configuration information:")
        print(f"Input file: {input_csv}")
        print(f"CSV result file: {csv_file}")
        print(f"JSON log file: {json_file}")
        print(f"Prediction label: {label}")
        print(f"Prompt mode: {prompt_mode}")
        print(f"Using model: {model_config['display_name']}")
        print(f"Unique identifier: {unique_id}")
        
        # Execute retry workflow
        return execute_retry_workflow(
            input_csv=input_csv,
            csv_file=csv_file,
            json_file=json_file, 
            txt_file=txt_file,
            model_config=model_config,
            label=label,
            prompt_mode=prompt_mode,
            unique_id=unique_id,
            mode="automatic"
        )
        
    except Exception as e:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} Automatic mode execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def execute_retry_workflow(input_csv: str, csv_file: str, json_file: str, txt_file: str, 
                          model_config: Dict, label: str, prompt_mode: str, unique_id: str, mode: str):
    """Execute real-time retry workflow"""
    try:
        # 1. Load data
        print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Step 1: load data")
        input_data = load_input_data(input_csv)
        validate_label(input_data, label)
        
        existing_logs = load_existing_results(json_file)
        existing_csv = load_existing_csv_results(csv_file)
        
        # 2. Find tasks that need to be retried
        print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Step 2: analyze tasks that need to be retried")
        retry_patient_ids = find_failed_tasks(input_data, existing_logs, existing_csv)
        
        # Note: The case where no retry is needed is now handled in the user confirmation part
        
        # Check for abnormal retry task counts
        total_input_records = len(input_data)
        retry_count = len(retry_patient_ids)
        retry_percentage = (retry_count / total_input_records) * 100 if total_input_records > 0 else 0
        
        print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Retry task count check:")
        print(f"Total input records: {total_input_records}")
        print(f"Number of tasks to retry: {retry_count}")
        print(f"Retry percentage: {retry_percentage:.1f}%")
        
        # Check for abnormal retry task counts
        if retry_count >= 600:
            print(f"\n{Colors.RED}[ERROR]{Colors.RESET} Configuration error: abnormal retry task counts!")
            print(f"Need to retry {retry_count} tasks, this may indicate configuration errors!")
            print(f"\nPossible reasons:")
            print(f"1. input_csv file does not match the result file (different experiments/models)")
            print(f"2. Result file is empty or corrupted")
            print(f"3. Data type conversion issues causing incorrect record matching")
            print(f"\nPlease check:")
            print(f"1. Confirm that input_csv matches the result file from the same experiment")
            print(f"2. Check if there are valid prediction records in the result file")
            print(f"3. Verify that the patient_id format is consistent")
            print(f"\nIf you really need to retry a large number of tasks, please confirm that the configuration is correct and manually delete this check.")
            return False
        
        if retry_percentage > 80:
            print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Retry ratio too high ({retry_percentage:.1f}%), please confirm that the configuration is correct")
            print(f"Please check if the input file matches the result file")
        elif retry_percentage > 50:
            print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Retry ratio too high ({retry_percentage:.1f}%), this may be a normal API failure recovery")
        else:
            print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Retry ratio normal ({retry_percentage:.1f}%)")
        
        # User confirmation function
        print(f"\n{'='*60}")
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Retry task confirmation")
        print(f"{'='*60}")
        
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Retry task statistics:")
        print(f"Input data file: {os.path.basename(input_csv)}")
        print(f"Total input records: {total_input_records}")
        print(f"Number of tasks to retry: {retry_count}")
        
        print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Configuration information:")
        print(f"Model: {model_config['display_name']} ({model_config.get('model_name', 'Unknown')})")
        print(f"Prediction label: {label}")
        print(f"Prompt mode: {prompt_mode}")
        print(f"Running mode: {mode} mode")
        
        print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Target file:")
        print(f"CSV result: {os.path.basename(csv_file)}")
        print(f"JSON log: {os.path.basename(json_file)}")
        print(f"TXT metrics: {os.path.basename(txt_file)}")
        
        if retry_count > 0:
            print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Important reminder:")
            print(f"• Real-time update mode, save immediately after each task is completed")
            print(f"• Resume from the last checkpoint, continue running from the last checkpoint if interrupted")
            print(f"• Use new results to overwrite existing failed records")
            print(f"• If you make a mistake, you can use the backup file (automatically created in the results directory before starting)")
            
            if retry_percentage > 50:
                print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Retry ratio too high, please confirm that the configuration is correct!")
        
        print(f"\n" + "="*60)
        
        # Request user confirmation
        try:
            if retry_count > 0:
                user_input = input(f"{Colors.WHITE}[INFO]{Colors.RESET} Confirm to execute retry tasks? (Enter 'y' or 'yes' to confirm, any other input to cancel): ").strip().lower()
                
                if user_input in ['y', 'yes']:
                    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} User confirmed, starting to execute retry tasks...")
                else:
                    print(f"{Colors.RED}[ERROR]{Colors.RESET} User cancelled, exiting retry process")
                    return True
            else:
                user_input = input(f"{Colors.WHITE}[INFO]{Colors.RESET} No retry tasks, whether to recalculate performance metrics? (Enter 'y' or 'yes' to confirm, any other input to skip): ").strip().lower()
                
                if user_input not in ['y', 'yes']:
                    print(f"{Colors.WHITE}[INFO]{Colors.RESET} User skipped recalculating performance metrics, task completed")
                    return True
                else:
                    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} User confirmed, starting to recalculate performance metrics...")
                    # Execute metric recalculation
                    print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Recalculating performance metrics and checking if updates are needed...")
                    updated_metrics = recalculate_and_update_metrics(
                        csv_file=csv_file,
                        txt_file=txt_file,
                        label=label,
                        model_config=model_config,
                        prompt_mode=prompt_mode,
                        unique_id=unique_id,
                        mode=mode
                    )
                    
                    if updated_metrics:
                        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Performance metrics have been recalculated and updated")
                    else:
                        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Performance metrics verified, no updates needed")
                    
                    return True
        except KeyboardInterrupt:
            print(f"\n\n{Colors.RED}[ERROR]{Colors.RESET} User interrupted operation (Ctrl+C), exiting retry process")
            return True
        except Exception as e:
            print(f"\n{Colors.RED}[ERROR]{Colors.RESET} Input processing exception: {str(e)}, default to cancel operation")
            return True
        
        print(f"\n{'='*60}")
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Starting to execute the confirmed operation...")
        print(f"{'='*60}")
        
        # 3. Load model dependencies
        print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Step 3: load model dependencies")
        feature_descriptions = load_feature_descriptions()
        feature_lists = load_feature_lists()
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Loaded {len(feature_descriptions)} feature descriptions")
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Loaded feature lists: Diag({len(feature_lists['Diag'])}), Proc({len(feature_lists['Proc'])}), Med({len(feature_lists['Med'])}), TS({len(feature_lists['TS'])})")
        
        # Preload LLM_CD_FEATURES mode feature configuration
        corl_features = None
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Debug: current prompt_mode = '{prompt_mode}'")
        if prompt_mode == "LLM_CD_FEATURES":
            print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Step 3.1: preload LLM causal feature configuration")
            try:
                # Load LLM causal features
                llm_features_desc, config_file_path = load_llm_cd_features_for_label(model_config, label)
                
                if not llm_features_desc:
                    print(f"{Colors.RED}[ERROR]{Colors.RESET} LLM causal feature configuration for label '{label}' is empty")
                    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Configuration file: {config_file_path}")
                    return False
                
                print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Successfully loaded {len(llm_features_desc)} LLM causal features for label '{label}'")
                
                # Execute reverse mapping to convert feature descriptions to feature codes (using imported function)
                allowed_feature_codes, missing_features = convert_features_desc_to_codes(llm_features_desc, feature_descriptions)
                
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} Feature description->code conversion:")
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} LLM features: {len(llm_features_desc)}")
                print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Conversion successful: {len(allowed_feature_codes)}")
                if missing_features:
                    print(f"{Colors.RED}[ERROR]{Colors.RESET} No mapping found: {len(missing_features)}")
                
                corl_features = list(allowed_feature_codes)
                print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} LLM causal feature configuration completed, {len(corl_features)} feature codes")
                
            except Exception as e:
                print(f"{Colors.RED}[ERROR]{Colors.RESET} Failed to load LLM causal feature configuration: {str(e)}")
                print(f"Program terminated")
                return False
        
        elif prompt_mode == "CD_FEATURES_OPTIMIZED":
            print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Step 3.1: preload CD algorithm optimized feature configuration")
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} Debug: entering CD_FEATURES_OPTIMIZED preload branch")
            try:
                cd_algorithm = model_config.get("cd_algorithm", None)
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} Debug: cd_algorithm = '{cd_algorithm}'")
                if not cd_algorithm:
                    print(f"{Colors.RED}[ERROR]{Colors.RESET} Error: CD_FEATURES_OPTIMIZED mode requires specifying a CD algorithm")
                    print(f"Please add 'cd_algorithm' parameter in model_config")
                    return False
                
                # Load CD algorithm optimized features
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} Debug: starting to call load_cd_optimized_features_for_label")
                optimized_features_desc, config_file_path = load_cd_optimized_features_for_label(cd_algorithm, model_config, label)
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} Debug: load_cd_optimized_features_for_label returned: {len(optimized_features_desc) if optimized_features_desc else 0} features")
                
                if not optimized_features_desc:
                    print(f"{Colors.RED}[ERROR]{Colors.RESET} Error: {cd_algorithm} optimized feature configuration for label '{label}' is empty")
                    print(f"Configuration file: {config_file_path}")
                    return False
                
                print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Successfully loaded {len(optimized_features_desc)} {cd_algorithm} optimized features for label '{label}'")
                
                # Execute reverse mapping to convert feature descriptions to feature codes (simplified version, no user confirmation needed)
                allowed_feature_codes, missing_features = convert_features_desc_to_codes(optimized_features_desc, feature_descriptions)
                
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} Feature description->code conversion:")
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} {cd_algorithm} features: {len(optimized_features_desc)}")
                print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Conversion successful: {len(allowed_feature_codes)}")
                if missing_features:
                    print(f"{Colors.RED}[ERROR]{Colors.RESET} No mapping found: {len(missing_features)}")
                
                corl_features = list(allowed_feature_codes)
                print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} {cd_algorithm} optimized feature configuration completed, {len(corl_features)} feature codes")
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} Debug: corl_features first 10 = {corl_features[:10] if corl_features else 'None'}")
                
            except Exception as e:
                print(f"{Colors.RED}[ERROR]{Colors.RESET} Failed to load {cd_algorithm if 'cd_algorithm' in locals() else 'CD'} algorithm optimized feature configuration: {str(e)}")
                print(f"Program terminated, not allowed to revert to original feature list")
                import traceback
                traceback.print_exc()
                return False
        
        elif prompt_mode in ["CD_FILTERED", "CORL_FILTERED", "DirectLiNGAM_FILTERED"]:
            print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Step 3.1: preload {prompt_mode} feature configuration")
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} Debug: entering {prompt_mode} preload branch")
            try:
                # Determine CD algorithm type
                if prompt_mode == "CD_FILTERED":
                    cd_algorithm = model_config.get("cd_algorithm", "CORL")
                elif prompt_mode == "CORL_FILTERED":
                    cd_algorithm = "CORL"
                elif prompt_mode == "DirectLiNGAM_FILTERED":
                    cd_algorithm = "DirectLiNGAM"
                else:
                    cd_algorithm = "CORL"  # Default value
                
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} Debug: using CD algorithm = '{cd_algorithm}'")
                
                # Load CD algorithm features
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} Debug: starting to call load_cd_filtered_features_for_label")
                corl_features = load_cd_filtered_features_for_label(cd_algorithm, label)
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} Debug: load_cd_filtered_features_for_label returned: {len(corl_features) if corl_features else 0} features")
                
                if not corl_features:
                    print(f"{Colors.RED}[ERROR]{Colors.RESET} {cd_algorithm} feature configuration for label '{label}' is empty")
                    print(f"Please check configuration file: {cd_algorithm}_F.txt")
                    return False
                
                print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Successfully loaded {len(corl_features)} {cd_algorithm} features for label '{label}'")
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} Debug: corl_features first 10 = {corl_features[:10] if corl_features else 'None'}")
                
            except Exception as e:
                print(f"{Colors.RED}[ERROR]{Colors.RESET} Failed to load {cd_algorithm if 'cd_algorithm' in locals() else 'CD'} algorithm features configuration: {str(e)}")
                print(f"Program terminated")
                import traceback
                traceback.print_exc()
                return False
        
        else:
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} Debug: non-special mode ({prompt_mode}), skipping feature preload")
        
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Debug: preload completed, corl_features = {len(corl_features) if corl_features else 'None'}")
        
        # Prepare task-specific progress file path
        # Use task-specific checkpoint files to avoid conflicts between different tasks
        import hashlib
        task_signature = f"{os.path.basename(csv_file)}_{label}_{model_config.get('model_name', 'unknown')}"
        task_hash = hashlib.md5(task_signature.encode()).hexdigest()[:8]
        progress_file = os.path.join(os.path.dirname(csv_file), f"retry_progress_{task_hash}.json")
        
        # Execute real-time retry
        print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Step 4: execute real-time retry (supports checkpointing)")
        # Ensure results directory exists
        results_dir = os.path.dirname(csv_file)
        os.makedirs(results_dir, exist_ok=True)
        
        # Fix bug: ensure prompt_mode is correctly passed to model_config
        model_config["prompt_mode"] = prompt_mode
        print(f"Added prompt_mode to model_config: {prompt_mode}")
        
        total_tasks, successful_tasks, failed_tasks = retry_failed_predictions_realtime(
            input_data=input_data,
            retry_patient_ids=retry_patient_ids,
            model_config=model_config,
            feature_descriptions=feature_descriptions,
            feature_lists=feature_lists,
            label=label,
            csv_file_path=csv_file,
            json_file_path=json_file,
            progress_file_path=progress_file,
            max_workers=3,  # Use fewer workers during retry to avoid overload
            corl_features=corl_features  # Pass LLM causal feature configuration
        )
        
        if total_tasks == 0:
            print(f"{Colors.RED}[ERROR]{Colors.RESET} No retryable tasks found")
            return False
        
        # 6. Re-evaluate overall performance
        print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Step 5: re-evaluate overall performance")
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Real-time retry result statistics:")
        print(f"Total tasks: {total_tasks}")
        print(f"Successful tasks: {successful_tasks}")
        print(f"Failed tasks: {failed_tasks}")
        print(f"Success rate: {successful_tasks/total_tasks*100:.1f}%")
        
        # Reload updated CSV for complete evaluation, explicitly specify data types
        updated_csv = pd.read_csv(csv_file, dtype={
            'icustay_id': str,
            'patient_id': str
            # Other column types will be determined at runtime
        })
        
        if 'patient_id' in updated_csv.columns:
            pred_col = f'{label.lower()}_prediction'
            prob_col = f'{label.lower()}_probability'
            
            # Filter out successful predictions
            valid_mask = (updated_csv[pred_col] != -1) & (updated_csv[prob_col] != -1)
            valid_data = updated_csv[valid_mask]
            
            if len(valid_data) > 0:
                y_true = valid_data['ground_truth'].values
                y_pred = valid_data[pred_col].values
                y_pred_proba = valid_data[prob_col].values
                
                metrics = evaluate_predictions(y_true, y_pred, y_pred_proba)
                
                print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Final performance evaluation:")
                print("-"*40)
                print(f"Total tasks:        {len(updated_csv)}")
                print(f"Successful predictions:      {len(valid_data)}")
                print(f"Failed tasks:      {len(updated_csv) - len(valid_data)}")
                print(f"Success rate:          {len(valid_data)/len(updated_csv)*100:.1f}%")
                print("-"*40)
                print(f"F1%:             {metrics['F1%']:>8.2f}%")
                print(f"AUROC:           {metrics['AUROC']:>8.4f}")
                print(f"AUPRC:           {metrics['AUPRC']:>8.4f}")
                print("-"*40)

                # Save metrics to txt file
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.write(f"Model performance evaluation after real-time retry ({mode} mode)\n")
                    f.write(f"="*50 + "\n")
                    f.write(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Run mode: {mode} mode (real-time update + checkpointing)\n")
                    f.write(f"Unique identifier: {unique_id}\n")
                    f.write(f"Model: {model_config['display_name']} ({model_config.get('model_name', 'Unknown')})\n")
                    f.write(f"Prediction label: {label}\n")
                    f.write(f"Prompt mode: {prompt_mode}\n")
                    f.write(f"\nRetry task statistics:\n")
                    f.write(f"Retry total tasks: {total_tasks}\n")
                    f.write(f"Retry successful tasks: {successful_tasks}\n")
                    f.write(f"Retry failed tasks: {failed_tasks}\n")
                    f.write(f"Retry success rate: {successful_tasks/total_tasks*100:.1f}%\n")
                    f.write(f"\nFinal task statistics:\n")
                    f.write(f"Total tasks: {len(updated_csv)}\n")
                    f.write(f"Successful predictions: {len(valid_data)}\n")
                    f.write(f"Failed tasks: {len(updated_csv) - len(valid_data)}\n")
                    f.write(f"Success rate: {len(valid_data)/len(updated_csv)*100:.1f}%\n")
                    f.write(f"\nPerformance metrics:\n")
                    f.write(f"F1%: {metrics['F1%']:.2f}%\n")
                    f.write(f"AUROC: {metrics['AUROC']:.4f}\n")
                    f.write(f"AUPRC: {metrics['AUPRC']:.4f}\n")
                    f.write(f"\nFile information:\n")
                    f.write(f"CSV result file: {os.path.basename(csv_file)}\n")
                    f.write(f"JSON log file: {os.path.basename(json_file)}\n")
                    f.write(f"Metric file: {os.path.basename(txt_file)}\n")
                    f.write(f"Progress file: {os.path.basename(progress_file)}\n")

                print(f"Saved performance metrics to: {txt_file}")
            else:
                print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} No valid predictions for performance evaluation")

                # Even if there are no valid predictions, save basic information to txt file, but still save the metrics
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.write(f"Model performance evaluation after real-time retry ({mode} mode)\n")
                    f.write(f"="*50 + "\n")
                    f.write(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Run mode: {mode} mode (real-time update + checkpointing)\n")
                    f.write(f"Unique identifier: {unique_id}\n")
                    f.write(f"Model: {model_config['display_name']} ({model_config.get('model_name', 'Unknown')})\n")
                    f.write(f"Prediction label: {label}\n")
                    f.write(f"Prompt mode: {prompt_mode}\n")
                    f.write(f"\nRetry task statistics:\n")
                    f.write(f"Retry total tasks: {total_tasks}\n")
                    f.write(f"Retry successful tasks: {successful_tasks}\n")
                    f.write(f"Retry failed tasks: {failed_tasks}\n")
                    f.write(f"Retry success rate: {successful_tasks/total_tasks*100:.1f}%\n")
                    f.write(f"\nFinal task statistics:\n")
                    f.write(f"Total tasks: {len(updated_csv)}\n")
                    f.write(f"Successful predictions: 0\n")
                    f.write(f"Failed tasks: {len(updated_csv)}\n")
                    f.write(f"Success rate: 0.0%\n")
                    f.write(f"\nPerformance metrics:\n")
                    f.write(f"F1%: cannot be calculated (all tasks failed)\n")
                    f.write(f"AUROC: cannot be calculated (all tasks failed)\n")
                    f.write(f"AUPRC: cannot be calculated (all tasks failed)\n")
                    f.write(f"\nFile information:\n")
                    f.write(f"CSV result file: {os.path.basename(csv_file)}\n")
                    f.write(f"JSON log file: {os.path.basename(json_file)}\n")
                    f.write(f"Metric file: {os.path.basename(txt_file)}\n")
                    f.write(f"Progress file: {os.path.basename(progress_file)}\n")

                print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Saved basic information to: {txt_file}")
        
        print(f"\n{Colors.GREEN}[SUCCESS]{Colors.RESET} {mode} mode real-time retry completed!")
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Updated files:")
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} CSV result file: {csv_file}")
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} JSON log file: {json_file}")
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Metric file: {txt_file}")
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Progress file: {progress_file} {'(Cleaned)' if successful_tasks == total_tasks else '(Retained)'}")
        
        if failed_tasks == 0:
            print(f"\n{Colors.GREEN}[SUCCESS]{Colors.RESET} All retry tasks completed successfully!")
        else:
            print(f"\n{Colors.YELLOW}[WARNING]{Colors.RESET} {failed_tasks} tasks failed, can re-run script to continue retry")
        
        return True
        
    except Exception as e:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} Error occurred during {mode} mode retry: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def retry_failed_predictions_realtime(
    input_data: pd.DataFrame,
    retry_patient_ids: List[str],
    model_config: Dict,
    feature_descriptions: Dict,
    feature_lists: Dict,
    label: str,
    csv_file_path: str,
    json_file_path: str,
    progress_file_path: str,
    max_workers: int = 10,
    corl_features: Optional[List[str]] = None
) -> Tuple[int, int, int]:
    """
    Real-time retry failed prediction tasks, supports checkpointing
    Return: (total_tasks, successful_tasks, failed_tasks)
    """
    print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Starting real-time retry mode...")
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Total tasks: {len(retry_patient_ids)}")
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Concurrent workers: {max_workers}")
    
    # Verify feature configuration for special modes
    prompt_mode = model_config.get("prompt_mode", "DIRECTLY_PROMPTING")
    if prompt_mode in ["LLM_CD_FEATURES", "CD_FEATURES_OPTIMIZED", "CORL_FILTERED", "DirectLiNGAM_FILTERED", "CD_FILTERED"]:
        if not corl_features:
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Fatal error: feature configuration is empty in {prompt_mode} mode!")
            print(f"Pre-load step failed, program cannot continue execution")
            print(f"Please check if the configuration file exists and contains the configuration for label '{label}'")
            raise ValueError(f"{prompt_mode} mode requires valid feature configuration, but configuration is empty")
        else:
            print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} {prompt_mode} mode feature configuration verified, contains {len(corl_features)} features")
    
    if not retry_patient_ids:
        print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} No tasks to retry")
        return 0, 0, 0
    
    # Create file backup
    csv_backup = create_file_backup(csv_file_path)
    json_backup = create_file_backup(json_file_path)
    
    # Prepare task metadata for verifying checkpointing data correctness
    task_metadata = {
        'csv_file': os.path.basename(csv_file_path),
        'json_file': os.path.basename(json_file_path),
        'label': label,
        'model_name': model_config.get('model_name', 'unknown'),
        'total_retry_ids': len(retry_patient_ids)  # Used to verify task size
    }
    
    # Load checkpointing progress (including task verification)
    completed_ids, previously_failed_ids = load_retry_progress(progress_file_path, task_metadata)
    
    # Filter out tasks that really need to be retried
    remaining_tasks = []
    for patient_id in retry_patient_ids:
        if patient_id not in completed_ids:
            remaining_tasks.append(patient_id)
    
    if completed_ids:
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Checkpoint: skip {len(completed_ids)} completed tasks")
    
    if not remaining_tasks:
        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} All tasks completed, no need to continue retry")
        return len(retry_patient_ids), len(completed_ids), len(previously_failed_ids)
    
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Tasks to process this time: {len(remaining_tasks)}")
    
    # Prepare data for retry tasks
    retry_tasks = []
    retry_patient_id_set = set(remaining_tasks)
    
    for idx, row in input_data.iterrows():
        # Handle data types uniformly, consistent with find_failed_tasks
        icustay_id = str(int(float(row['ICUSTAY_ID'])))
        patient_id = icustay_id
        if patient_id in retry_patient_id_set:
            groundtruth = int(row[label])
            retry_tasks.append((idx, row, groundtruth))
    
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Matched data: {len(retry_tasks)} tasks")
    
    if not retry_tasks:
        print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} No matching retry task data found")
        return len(retry_patient_ids), len(completed_ids), len(previously_failed_ids)
    
    # Correct statistics: separate checkpointing data and current processing count
    # Checkpoint completed and failed counts
    checkpoint_completed_count = len(completed_ids)
    checkpoint_failed_count = len(previously_failed_ids)
    
    # Counter for current processing (starting from 0)
    current_batch_completed = 0
    current_batch_failed = 0
    
    # Maintain ID set for saving progress
    current_completed_set = completed_ids.copy()
    current_failed_set = previously_failed_ids.copy()
    
    # Real-time process each task
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Starting real-time retry processing")
    print(f"{'='*60}")
    
    # Use thread pool but process results in real-time
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {}
        for task_data in retry_tasks:
            future = executor.submit(
                predict_single_patient, 
                task_data, 
                model_config, 
                feature_descriptions, 
                label, 
                feature_lists,
                corl_features  # Pass LLM causal feature configuration
            )
            future_to_task[future] = task_data
        
        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Submitted {len(future_to_task)} parallel tasks")
        
        # Real-time process completed tasks
        task_timeout = 120  # Single task timeout 2 minutes
        
        for future in as_completed(future_to_task.keys()):
            task_data = future_to_task[future]
            task_idx, task_row, task_groundtruth = task_data
            patient_id = str(int(float(task_row['ICUSTAY_ID'])))
            
            try:
                # Get prediction result
                result = future.result(timeout=task_timeout)
                
                if result and result.get('probability', -1) != -1:
                    # Task successful, update file immediately
                    print(f"\n{Colors.GREEN}[SUCCESS]{Colors.RESET} Processed successful task: {patient_id}")
                    
                    # Update CSV file
                    csv_success = update_single_csv_record(
                        csv_file_path=csv_file_path,
                        patient_id=patient_id,
                        prediction_data=result,
                        label=label
                    )
                    
                    # Update JSON file
                    json_success = update_single_json_record(
                        json_file_path=json_file_path,
                        patient_id=patient_id,
                        experiment_log=result['experiment_log']
                    )
                    
                    if csv_success and json_success:
                        current_batch_completed += 1
                        current_completed_set.add(patient_id)
                        # Remove from failed set (if previously failed)
                        current_failed_set.discard(patient_id)
                        
                        # Save progress (including task metadata)
                        save_retry_progress(progress_file_path, current_completed_set, current_failed_set, task_metadata)
                        
                        elapsed = time.time() - start_time
                        total_completed = checkpoint_completed_count + current_batch_completed
                        total_failed = checkpoint_failed_count + current_batch_failed
                        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} File updated successfully")
                        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Progress: successful={total_completed}, failed={total_failed}, time={elapsed:.1f}s")
                    else:
                        print(f"{Colors.RED}[ERROR]{Colors.RESET} File update failed, task marked as failed")
                        current_batch_failed += 1
                        current_failed_set.add(patient_id)
                        save_retry_progress(progress_file_path, current_completed_set, current_failed_set, task_metadata)
                
                else:
                    # Task prediction failed
                    print(f"\n{Colors.RED}[ERROR]{Colors.RESET} Prediction failed: {patient_id}")
                    current_batch_failed += 1
                    current_failed_set.add(patient_id)
                    save_retry_progress(progress_file_path, current_completed_set, current_failed_set, task_metadata)
                    
                    elapsed = time.time() - start_time
                    total_completed = checkpoint_completed_count + current_batch_completed
                    total_failed = checkpoint_failed_count + current_batch_failed
                    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Progress: successful={total_completed}, failed={total_failed}, time={elapsed:.1f}s")
                    
            except Exception as e:
                # Task exception
                print(f"\n{Colors.RED}[ERROR]{Colors.RESET} Task exception: {patient_id}")
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} {str(e)}")
                current_batch_failed += 1
                current_failed_set.add(patient_id)
                save_retry_progress(progress_file_path, current_completed_set, current_failed_set, task_metadata)
                
                elapsed = time.time() - start_time
                total_completed = checkpoint_completed_count + current_batch_completed
                total_failed = checkpoint_failed_count + current_batch_failed
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} Progress: successful={total_completed}, failed={total_failed}, time={elapsed:.1f}s")
    
    # Correct final statistics: use correct base to calculate success rate
    total_elapsed = time.time() - start_time
    total_tasks = len(retry_patient_ids)  # Total tasks to retry this time
    successful_tasks = current_batch_completed  # Total successful tasks this time
    failed_tasks = current_batch_failed  # Total failed tasks this time
    
    # Calculate success rate for this retry
    if total_tasks > 0:
        batch_success_rate = (successful_tasks / total_tasks) * 100
    else:
        batch_success_rate = 0.0
    
    print(f"\n{'='*60}")
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Real-time retry summary")
    print(f"{'='*60}")
    print(f"Task statistics:")
    print(f"Total tasks: {total_tasks}")
    print(f"Successful tasks: {successful_tasks}")
    print(f"Failed tasks: {failed_tasks}")
    print(f"Success rate: {batch_success_rate:.1f}%")
    
    # 🆕 Display checkpointing statistics
    if checkpoint_completed_count > 0 or checkpoint_failed_count > 0:
        print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Checkpoint statistics:")
        print(f"Checkpoint completed: {checkpoint_completed_count}")
        print(f"Checkpoint failed: {checkpoint_failed_count}")
        total_cumulative = checkpoint_completed_count + checkpoint_failed_count + successful_tasks + failed_tasks
        print(f"Total cumulative: {total_cumulative}")
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    
    # If there are failed tasks, display details
    if current_failed_set:
        print(f"\n{Colors.RED}[ERROR]{Colors.RESET} Failed task list (first 10):")
        failed_list = list(current_failed_set)[:10]
        for i, failed_id in enumerate(failed_list, 1):
            print(f"{i}. {failed_id}")
        if len(current_failed_set) > 10:
            print(f"... {len(current_failed_set) - 10} more failed tasks")
    
    # Clean up progress file (if all completed)
    if failed_tasks == 0:
        try:
            if os.path.exists(progress_file_path):
                os.remove(progress_file_path)
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} Progress file cleaned")
        except:
            pass
    
    return total_tasks, successful_tasks, failed_tasks

def parse_existing_metrics(txt_file_path: str) -> Dict:
    """Parse performance metrics from existing TXT file"""
    existing_metrics = {}
    
    if not os.path.exists(txt_file_path):
        return existing_metrics
    
    try:
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse key metrics
        import re
        
        # Parse F1%
        f1_match = re.search(r'F1%:\s*([\d.]+)%', content)
        if f1_match:
            existing_metrics['F1%'] = float(f1_match.group(1))
        
        # Parse AUROC
        auroc_match = re.search(r'AUROC:\s*([\d.]+)', content)
        if auroc_match:
            existing_metrics['AUROC'] = float(auroc_match.group(1))
        
        # Parse AUPRC
        auprc_match = re.search(r'AUPRC:\s*([\d.]+)', content)
        if auprc_match:
            existing_metrics['AUPRC'] = float(auprc_match.group(1))
        
        # Parse task statistics
        total_match = re.search(r'Total tasks:\s*(\d+)', content)
        if total_match:
            existing_metrics['total_tasks'] = int(total_match.group(1))
        
        success_match = re.search(r'Successful predictions:\s*(\d+)', content)
        if success_match:
            existing_metrics['successful_tasks'] = int(success_match.group(1))
        
        failed_match = re.search(r'Failed tasks:\s*(\d+)', content)
        if failed_match:
            existing_metrics['failed_tasks'] = int(failed_match.group(1))
        
        success_rate_match = re.search(r'Success rate:\s*([\d.]+)%', content)
        if success_rate_match:
            existing_metrics['success_rate'] = float(success_rate_match.group(1))
        
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Parsed existing metrics: F1%={existing_metrics.get('F1%', 'N/A')}, "
              f"AUROC={existing_metrics.get('AUROC', 'N/A')}, "
              f"AUPRC={existing_metrics.get('AUPRC', 'N/A')}")
        
    except Exception as e:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} Failed to parse existing metrics file: {str(e)}")
    
    return existing_metrics

def recalculate_and_update_metrics(
    csv_file: str,
    txt_file: str,
    label: str,
    model_config: Dict,
    prompt_mode: str,
    unique_id: str,
    mode: str
) -> bool:
    """
    Recalculate performance metrics and compare with existing metrics, update TXT file if different
    Return: True if update, False if no update
    """
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Starting to recalculate performance metrics...")
    
    # 1. Parse existing metrics
    existing_metrics = parse_existing_metrics(txt_file)
    
    # 2. Recalculate metrics from CSV
    if not os.path.exists(csv_file):
        print(f"{Colors.RED}[ERROR]{Colors.RESET} CSV file does not exist, cannot recalculate metrics: {csv_file}")
        return False
    
    try:
        # Explicitly specify data types to avoid type inference issues
        df = pd.read_csv(csv_file, dtype={
            'icustay_id': str,
            'patient_id': str
            # Other column types will be determined in subsequent processing
        })
        
        # Ensure key columns are string types
        if 'patient_id' in df.columns:
            df['patient_id'] = df['patient_id'].astype(str)
        if 'icustay_id' in df.columns:
            df['icustay_id'] = df['icustay_id'].astype(str)
            
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Loaded CSV data: {len(df)} rows")
        
        if 'patient_id' not in df.columns:
            print(f"{Colors.RED}[ERROR]{Colors.RESET} CSV file missing patient_id column")
            return False
        
        pred_col = f'{label.lower()}_prediction'
        prob_col = f'{label.lower()}_probability'
        
        if pred_col not in df.columns or prob_col not in df.columns or 'ground_truth' not in df.columns:
            print(f"{Colors.RED}[ERROR]{Colors.RESET} CSV file missing necessary columns: {pred_col}, {prob_col}, ground_truth")
            return False
        
        # Filter out successful predictions
        valid_mask = (df[pred_col] != -1) & (df[prob_col] != -1) & (df[pred_col].notna()) & (df[prob_col].notna())
        valid_data = df[valid_mask]
        
        total_tasks = len(df)
        successful_tasks = len(valid_data)
        failed_tasks = total_tasks - successful_tasks
        success_rate = (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Task statistics: total={total_tasks}, successful={successful_tasks}, failed={failed_tasks}, success rate={success_rate:.1f}%")
        
        # Calculate performance metrics
        new_metrics = {}
        
        if successful_tasks > 0:
            y_true = valid_data['ground_truth'].values
            y_pred = valid_data[pred_col].values
            y_pred_proba = valid_data[prob_col].values
            
            # Ensure no missing values
            valid_indices = ~(pd.isna(y_true) | pd.isna(y_pred) | pd.isna(y_pred_proba))
            y_true = y_true[valid_indices]
            y_pred = y_pred[valid_indices]
            y_pred_proba = y_pred_proba[valid_indices]
            
            if len(y_true) > 0:
                metrics = evaluate_predictions(y_true, y_pred, y_pred_proba)
                new_metrics = {
                    'F1%': metrics['F1%'],
                    'AUROC': metrics['AUROC'],
                    'AUPRC': metrics['AUPRC'],
                    'total_tasks': total_tasks,
                    'successful_tasks': successful_tasks,
                    'failed_tasks': failed_tasks,
                    'success_rate': success_rate
                }
                
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} Recalculated metrics: F1%={new_metrics['F1%']:.2f}, "
                      f"AUROC={new_metrics['AUROC']:.4f}, "
                      f"AUPRC={new_metrics['AUPRC']:.4f}")
            else:
                print(f"{Colors.RED}[ERROR]{Colors.RESET} No valid data after filtering for metric calculation")
                return False
        else:
            print(f"{Colors.RED}[ERROR]{Colors.RESET} No successful predictions, cannot calculate performance metrics")
            return False
        
        # 3. Compare metrics for differences
        needs_update = False
        tolerance = 1e-6  # Floating point comparison tolerance
        
        for key in ['F1%', 'AUROC', 'AUPRC', 'total_tasks', 'successful_tasks', 'failed_tasks', 'success_rate']:
            if key in existing_metrics and key in new_metrics:
                old_value = existing_metrics[key]
                new_value = new_metrics[key]
                
                if key in ['F1%', 'AUROC', 'AUPRC', 'success_rate']:
                    # Floating point comparison
                    if abs(old_value - new_value) > tolerance:
                        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Metric difference {key}: {old_value} -> {new_value}")
                        needs_update = True
                else:
                    # Integer comparison
                    if old_value != new_value:
                        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Statistical difference {key}: {old_value} -> {new_value}")
                        needs_update = True
            elif key in new_metrics:
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} New metric {key}: {new_metrics[key]}")
                needs_update = True
        
        # 4. If update is needed, rewrite TXT file
        if needs_update:
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} Detected metric differences, updating TXT file...")
            
            # Create backup
            backup_path = create_file_backup(txt_file)
            
            try:
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.write(f"Recalculated model performance evaluation results ({mode} mode)\n")
                    f.write(f"="*50 + "\n")
                    f.write(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Run mode: {mode} mode (metrics recalculated)\n")
                    f.write(f"Unique identifier: {unique_id}\n")
                    f.write(f"Model: {model_config['display_name']} ({model_config.get('model_name', 'Unknown')})\n")
                    f.write(f"Prediction label: {label}\n")
                    f.write(f"Prompt mode: {prompt_mode}\n")
                    f.write(f"\nTask statistics:\n")
                    f.write(f"Total tasks: {new_metrics['total_tasks']}\n")
                    f.write(f"Successful predictions: {new_metrics['successful_tasks']}\n")
                    f.write(f"Failed tasks: {new_metrics['failed_tasks']}\n")
                    f.write(f"Success rate: {new_metrics['success_rate']:.1f}%\n")
                    f.write(f"\nPerformance metrics:\n")
                    f.write(f"F1%: {new_metrics['F1%']:.2f}%\n")
                    f.write(f"AUROC: {new_metrics['AUROC']:.4f}\n")
                    f.write(f"AUPRC: {new_metrics['AUPRC']:.4f}\n")
                    f.write(f"\nFile information:\n")
                    f.write(f"CSV result file: {os.path.basename(csv_file)}\n")
                    f.write(f"JSON log file: {os.path.basename(csv_file.replace('predict_results', 'experiment_logs').replace('.csv', '.json'))}\n")
                    f.write(f"Metrics file: {os.path.basename(txt_file)}\n")
                    f.write(f"\nNote: This file has been updated based on the recalculated metrics\n")
                
                print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} TXT file updated: {txt_file}")
                
                # Delete backup (if successful)
                if backup_path and os.path.exists(backup_path):
                    try:
                        os.remove(backup_path)
                    except:
                        pass
                
                return True
                
            except Exception as e:
                print(f"{Colors.RED}[ERROR]{Colors.RESET} Failed to update TXT file: {str(e)}")
                # Restore backup
                restore_from_backup(txt_file, backup_path)
                return False
        else:
            print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Metrics consistent, no update needed")
            return False
        
    except Exception as e:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} Error occurred while recalculating metrics: {str(e)}")
        return False

def parse_filename_config(filename: str) -> Dict[str, Optional[str]]:
    """
    Parse configuration information from filename
    Supports two formats:
    1. Standard format: {version_prefix}{label}_predict_results_{model}_{prompt}{cd_algorithm}_{id}.csv
    2. CD_FEATURES_OPTIMIZED format: {version_prefix}{label}_{cd_algorithm}_optimized_{model}_{prompt}_{id}.csv
    
    Args:
        filename (str): filename
        
    Returns:
        Dict[str, Optional[str]]: parsed configuration information
    """
    config = {
        "label": None,
        "model": None, 
        "prompt_mode": None,
        "cd_algorithm": None,
        "version_prefix": "",
        "unique_id": None
    }
    
    # Remove file extension
    basename = os.path.splitext(filename)[0]
    
    try:
        # Detect version prefix
        version_match = re.match(r'^(\d+_)', basename)
        if version_match:
            config["version_prefix"] = version_match.group(1)
            basename = basename[len(config["version_prefix"]):]
        
        # Extract unique ID (last 8 characters)
        if len(basename) > 8 and basename[-9] == '_':
            config["unique_id"] = basename[-8:]
            basename = basename[:-9]
        
        # Try to parse CD_FEATURES_OPTIMIZED format
        cd_optimized_pattern = r'^(.+?)_(.+?)_optimized_(.+?)_(.+)$'
        cd_match = re.match(cd_optimized_pattern, basename)
        
        if cd_match and 'optimized' in basename:
            # CD_FEATURES_OPTIMIZED format: {label}_{cd_algorithm}_optimized_{model}_{prompt}
            config["label"] = cd_match.group(1)
            cd_algo_raw = cd_match.group(2)
            config["model"] = cd_match.group(3)
            config["prompt_mode"] = cd_match.group(4).upper()
            
            # Standardize CD algorithm name
            cd_algo_mapping = {
                'directlingam': 'DirectLiNGAM',
                'corl': 'CORL', 
                'ges': 'GES'
            }
            config["cd_algorithm"] = cd_algo_mapping.get(cd_algo_raw.lower(), cd_algo_raw.upper())
            
            # Restore prompt_mode format
            prompt_mapping = {
                'CDFEATURESOPTIMIZED': 'CD_FEATURES_OPTIMIZED'
            }
            config["prompt_mode"] = prompt_mapping.get(config["prompt_mode"], config["prompt_mode"])
        
        else:
            # Try to parse standard format
            standard_pattern = r'^(.+?)_predict_results_(.+?)_(.+)$'
            std_match = re.match(standard_pattern, basename)
            
            if std_match:
                config["label"] = std_match.group(1)
                config["model"] = std_match.group(2)
                prompt_and_cd = std_match.group(3)
                
                # Parse prompt mode and CD algorithm
                # Supported formats: cdfiltered_ges, directlyprompting, corlfiltered_corl, etc.
                cd_algorithms = ['_ges', '_corl', '_directlingam']
                cd_found = None
                
                for cd_suffix in cd_algorithms:
                    if prompt_and_cd.endswith(cd_suffix):
                        cd_algo_raw = cd_suffix[1:]
                        # Standardize CD algorithm name
                        cd_algo_mapping = {
                            'directlingam': 'DirectLiNGAM',
                            'corl': 'CORL',
                            'ges': 'GES'
                        }
                        config["cd_algorithm"] = cd_algo_mapping.get(cd_algo_raw.lower(), cd_algo_raw.upper())
                        config["prompt_mode"] = prompt_and_cd[:-len(cd_suffix)]
                        cd_found = True
                        break
                
                if not cd_found:
                    config["prompt_mode"] = prompt_and_cd
                
                # Restore prompt_mode format
                prompt_mapping = {
                    'DIRECTLYPROMPTING': 'DIRECTLY_PROMPTING',
                    'CHAINOFTHOUGHT': 'CHAIN_OF_THOUGHT',
                    'SELFREFLECTION': 'SELF_REFLECTION',
                    'ROLEPLAYING': 'ROLE_PLAYING',
                    'INCONTEXTLEARNING': 'IN_CONTEXT_LEARNING',
                    'CSVDIRECT': 'CSV_DIRECT',
                    'CSVRAW': 'CSV_RAW',
                    'JSONSTRUCTURED': 'JSON_STRUCTURED',
                    'LATEXTABLE': 'LATEX_TABLE',
                    'NATURALLANGUAGE': 'NATURAL_LANGUAGE',
                    'CORLFILTERED': 'CORL_FILTERED',
                    'DIRECTLINGAMFILTERED': 'DirectLiNGAM_FILTERED',
                    'CDFILTERED': 'CD_FILTERED',
                    'LLMCDFEATURES': 'LLM_CD_FEATURES'
                }
                
                if config["prompt_mode"]:
                    config["prompt_mode"] = prompt_mapping.get(config["prompt_mode"].upper(), config["prompt_mode"].upper())
                
                # Special handling: infer cd_algorithm from prompt_mode
                if config["prompt_mode"] == 'CORL_FILTERED' and not config["cd_algorithm"]:
                    config["cd_algorithm"] = 'CORL'
                elif config["prompt_mode"] == 'DirectLiNGAM_FILTERED' and not config["cd_algorithm"]:
                    config["cd_algorithm"] = 'DirectLiNGAM'
    
    except Exception as e:
        print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Failed to parse filename: {filename}, error: {str(e)}")
    
    return config

def normalize_model_name(display_name: str) -> str:
    """
    Standardize display_name to filename format
    Apply the same conversion rules as prediction.py
    """
    return display_name.replace(".", "_").replace("-", "_")

def validate_manual_config_against_files(manual_config: Dict, csv_file: str, json_file: str, txt_file: str) -> Tuple[bool, List[str]]:
    """
    Validate manual configuration against file information - use loose inclusion matching
    
    Args:
        manual_config (Dict): manual configuration
        csv_file (str): CSV file path  
        json_file (str): JSON file path
        txt_file (str): TXT file path
        
    Returns:
        Tuple[bool, List[str]]: (whether matched, check report list)
    """
    reports = []
    all_match = True
    
    csv_filename = os.path.basename(csv_file)
    filename_lower = csv_filename.lower()
    
    def normalize_for_comparison(text):
        """Normalize text for fuzzy comparison: lowercase, remove separators"""
        if not text:
            return ""
        return text.lower().replace("-", "").replace("_", "").replace(".", "").replace(" ", "")
    
    def fuzzy_contains_check(config_value, filename, field_name):
        """Fuzzy inclusion check"""
        if not config_value:
            return True, f"{Colors.GREEN}[SUCCESS]{Colors.RESET} {field_name}: not configured, skip check"
        
        config_normalized = normalize_for_comparison(str(config_value))
        filename_normalized = normalize_for_comparison(filename)
        
        # Check if config value appears in filename
        if config_normalized in filename_normalized:
            return True, f"{Colors.GREEN}[SUCCESS]{Colors.RESET} {field_name} matched: '{config_value}' found in filename"
        
        # For compound words, try token matching
        config_parts = [part for part in [config_value.replace("-", " ").replace("_", " ")] if part]
        for part in str(config_value).replace("-", " ").replace("_", " ").split():
            part_normalized = normalize_for_comparison(part)
            if len(part_normalized) >= 3 and part_normalized in filename_normalized:  # Only check parts with 3 or more characters
                return True, f"{Colors.GREEN}[SUCCESS]{Colors.RESET} {field_name} partial match: '{part}' (from '{config_value}') found in filename"
        
        return False, f"{Colors.RED}[ERROR]{Colors.RESET} {field_name} not matched: '{config_value}' not found in filename"
    
    reports.append("Loose configuration check results (inclusion matching)")
    reports.append("=" * 50)
    reports.append(f"Filename: {csv_filename}")
    reports.append("")
    
    # Check label
    override_label = manual_config.get("override_label")
    if override_label:
        match_result, message = fuzzy_contains_check(override_label, filename_lower, "Label")
        reports.append(message)
        if not match_result:
            all_match = False
            reports.append(f"Suggestion: check if label '{override_label}' matches the file")
    
    # Check model
    model_config = manual_config.get("override_model_config", {})
    manual_display_name = model_config.get("display_name")
    manual_model_name = model_config.get("model_name")
    
    model_match_found = False
    if manual_display_name:
        match_result, message = fuzzy_contains_check(manual_display_name, filename_lower, "Model display name")
        reports.append(message)
        if match_result:
            model_match_found = True
    
    if manual_model_name and not model_match_found:
        match_result, message = fuzzy_contains_check(manual_model_name, filename_lower, "Model name")
        reports.append(message)
        if match_result:
            model_match_found = True
    
    if (manual_display_name or manual_model_name) and not model_match_found:
        all_match = False
        reports.append(f"Suggestion: check if model configuration matches the file")
        reports.append(f"Model configuration: display_name='{manual_display_name}', model_name='{manual_model_name}'")
    
    # Check CD algorithm
    manual_cd_algorithm = model_config.get("cd_algorithm")
    if manual_cd_algorithm:
        match_result, message = fuzzy_contains_check(manual_cd_algorithm, filename_lower, "CD algorithm")
        reports.append(message)
        if not match_result:
            all_match = False
            reports.append(f"Suggestion: check if CD algorithm '{manual_cd_algorithm}' matches the file")
    
    # Check prompt mode (if specified)
    override_prompt_mode = manual_config.get("override_prompt_mode")
    if override_prompt_mode:
        # For prompt mode, check keywords
        prompt_keywords = []
        if "CD_FEATURES_OPTIMIZED" in override_prompt_mode:
            prompt_keywords.extend(["optimized", "cdfeaturesoptimized"])
        elif "CD_FILTERED" in override_prompt_mode:
            prompt_keywords.extend(["filtered", "cdfiltered"])
        elif "DIRECTLY_PROMPTING" in override_prompt_mode:
            prompt_keywords.extend(["directly", "directlyprompting"])
        
        prompt_match_found = False
        for keyword in prompt_keywords:
            match_result, _ = fuzzy_contains_check(keyword, filename_lower, "")
            if match_result:
                prompt_match_found = True
                break
        
        if prompt_match_found:
            reports.append(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Prompt mode matched: '{override_prompt_mode}' related keywords found in filename")
        else:
            reports.append(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Prompt mode check: '{override_prompt_mode}' related keywords not found in filename")
            # For prompt mode mismatch, not set as severe error, just warning
    
    # Additional file consistency check
    reports.append("")
    reports.append(f"{Colors.WHITE}[INFO]{Colors.RESET} File consistency check:")
    
    # Check if all files exist and have consistent IDs
    base_names = []
    file_paths = [csv_file, json_file, txt_file]
    file_types = ["CSV", "JSON", "TXT"]
    
    for i, (file_path, file_type) in enumerate(zip(file_paths, file_types)):
        if os.path.exists(file_path):
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            base_names.append(base_name)
            reports.append(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} {file_type} file exists: {os.path.basename(file_path)}")
        else:
            reports.append(f"{Colors.RED}[ERROR]{Colors.RESET} {file_type} file does not exist: {os.path.basename(file_path)}")
            all_match = False
    
    # Check if file names have consistent IDs (extract last 8 characters as unique ID)
    if len(base_names) >= 2:
        unique_ids = []
        for base_name in base_names:
            # Extract last 8 characters as ID
            if len(base_name) >= 8:
                unique_id = base_name[-8:]
                unique_ids.append(unique_id)
        
        if len(set(unique_ids)) == 1:
            reports.append(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} File ID consistent: {unique_ids[0]}")
        else:
            reports.append(f"{Colors.RED}[ERROR]{Colors.RESET} File ID inconsistent: {unique_ids}")
            reports.append(f"This may indicate files from different experiments")
    
    # Summary
    reports.append("")
    if all_match:
        reports.append(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Configuration check passed! File matches configuration")
        reports.append("Loose matching mode: focus on inclusion of key fields")
    else:
        reports.append(f"{Colors.RED}[ERROR]{Colors.RESET} Found configuration mismatch issues")
        reports.append("Loose matching mode: may need to adjust configuration or confirm file selection")
        reports.append("If you confirm the file selection is correct, you can choose to ignore the warning and continue execution")
    
    return all_match, reports

if __name__ == "__main__":
    main() 