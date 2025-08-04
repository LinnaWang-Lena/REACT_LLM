#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature optimization script

Function: Call large model API to optimize causal feature list
- Support configuration of model_name, labels, features file
- All features are mapped to readable descriptions through features-desc.csv
- Call API and reuse configuration from prediction.py
"""

import os
import json
import sys
import glob
import uuid
import time
import traceback
import re
from datetime import datetime

# Import existing functions from prediction.py
from prediction import (
    query_model, 
    load_feature_descriptions, 
    load_feature_lists
)

# Import outcome description function from causal feature generation script
from generate_causal_features import get_outcome_descriptions

# ANSI color definition class
class Colors:
    """Academic terminal output color definition"""
    RED = '\033[91m'      # Error
    GREEN = '\033[92m'    # Success
    YELLOW = '\033[93m'   # Warning
    WHITE = '\033[97m'    # Info
    RESET = '\033[0m'     # Reset

# Optimization configuration
OPTIMIZATION_CONFIG = {
    # Model configuration (reuse MODEL_CONFIG structure from prediction.py)
    "model_config": {
        "api_type": "openai",  # "openai" or "dashscope"
        "model_name": "o3-mini",  # Choose model here
        "display_name": "GPT-o3 mini",
        
        # OpenAI configuration
        "openai_config": {
            "api_key": "your_api_key",  # Replace with your API key
            "api_base": 'your_api_base',
        },
        
        # DashScope configuration (if using Aliyun model)
        "dashscope_config": {
            "api_key": "your_api_key",
        },
        
        # Generation parameter configuration
        "generation_params": {
            "temperature": 0.0,      # Lower temperature to ensure more stable output
            "top_p": 1.0,
            "max_tokens": 5000       # Significantly increase token limit to avoid response truncation
        }
    },
    
    # Feature file selection
    "features_file": "GES_F.txt",  # Optional: "CORL_F.txt", "DirectLiNGAM_F.txt", "GES_F.txt"
    
    # Target label list (supports multiple labels)
    "target_labels": [
        "DIEINHOSPITAL",
        "Readmission_30", 
        "Multiple_ICUs",
        "sepsis_all", 
        "FirstICU24_AKI_ALL",
        "LOS_Hospital", 
        "ICU_within_12hr_of_admit"
    ],
    
    # Experiment configuration
    "experiment_count": 3,  # Repeat experiment count
    
    # Output configuration
    "output_dir": "optimization_results_o3-mini",
    
    # Quality control configuration
    "strict_mode": True,              # Strict mode: require 100% success rate to generate final result
    "min_success_rate": 1.0,          # Minimum success rate threshold (1.0 = 100%)
    "allow_partial_consensus": False  # Whether to allow partial success generation of consensus
}

def get_all_available_feature_descriptions():
    """
    Get complete 262 feature description list (excluding 4 basic features)
    
    Returns:
        tuple: (feature description list, original feature name list)
    """
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Loading complete feature list (excluding basic features)...")
    
    # 1. Load 262 original feature names (excluding 4 basic features) from selected_features.txt
    feature_lists = load_feature_lists()
    original_names = []
    # Skip basic features: ['GENDER', 'ADMISSION_TYPE', 'FIRST_CAREUNIT', 'AGE']
    original_names.extend(feature_lists['Diag'])     # 62
    original_names.extend(feature_lists['Proc'])     # 27  
    original_names.extend(feature_lists['Med'])      # 55
    original_names.extend(feature_lists['TS'])       # 115
    
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Loaded {len(original_names)} original feature names (excluding 4 basic features)")
    
    # 2. Load feature description mapping table
    feature_descriptions = load_feature_descriptions()
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Loaded {len(feature_descriptions)} feature description mappings")
    
    # 3. Map original feature names to descriptions
    feature_descriptions_list = []
    missing_descriptions = []
    
    for feature_name in original_names:
        description = feature_descriptions.get(feature_name, None)
        if description:
            # Remove extra spaces and use description
            feature_descriptions_list.append(description.strip())
        else:
            # If no description is found, use original feature name and record
            feature_descriptions_list.append(feature_name)
            missing_descriptions.append(feature_name)
    
    if missing_descriptions:
        print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} {len(missing_descriptions)} features have no description mapping, using original names")
        print(f"Missing description features: {missing_descriptions[:10]}{'...' if len(missing_descriptions) > 10 else ''}")
    
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Feature mapping completed: {len(feature_descriptions_list)} feature descriptions")
    return feature_descriptions_list, original_names

def get_target_feature_descriptions(features_file, label):
    """
    Get target feature description list for specified label
    
    Args:
        features_file (str): feature file name ("CORL_F.txt" or "DirectLiNGAM_F.txt")
        label (str): target label name
        
    Returns:
        tuple: (target feature description list, original feature name list)
    """
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Loading features for label '{label}' in {features_file}...")
    
    # 1. Read feature file to get original feature names
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, features_file)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{Colors.RED}[ERROR]{Colors.RESET} Feature file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Execute Python code to get feature list
    namespace = {}
    exec(content, namespace)
    target_feature_names = namespace.get(label, [])
    
    if not target_feature_names:
        raise ValueError(f"{Colors.RED}[ERROR]{Colors.RESET} No features found for label '{label}' in {features_file}")
    
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Found {len(target_feature_names)} target features")
    
    # 2. Load feature description mapping table
    feature_descriptions = load_feature_descriptions()
    
    # 3. Map original feature names to descriptions
    target_feature_descriptions = []
    missing_descriptions = []
    
    for feature_name in target_feature_names:
        description = feature_descriptions.get(feature_name, None)
        if description:
            target_feature_descriptions.append(description.strip())
        else:
            target_feature_descriptions.append(feature_name)
            missing_descriptions.append(feature_name)
    
    if missing_descriptions:
        print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} {len(missing_descriptions)} target features have no description mapping")
        print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Missing description features: {missing_descriptions}")
    
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Target feature mapping completed: {len(target_feature_descriptions)} feature descriptions")
    return target_feature_descriptions, target_feature_names

class OptimizationLogger:
    """
    Optimization process logger
    """
    def __init__(self, session_dir, target_label, experiment_num=None):
        """
        Initialize logger
        
        Args:
            session_dir (str): session directory path
            target_label (str): target label
            experiment_num (int, optional): experiment number
        """
        self.session_dir = session_dir
        self.target_label = target_label
        self.experiment_num = experiment_num
        
        # Create logs subdirectory
        self.logs_dir = os.path.join(session_dir, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Define log file path
        exp_suffix = f"_exp{experiment_num}" if experiment_num else ""
        self.api_log = os.path.join(self.logs_dir, f"api_requests{exp_suffix}.log")
        self.processing_log = os.path.join(self.logs_dir, f"processing{exp_suffix}.log")
        self.error_log = os.path.join(self.logs_dir, f"errors{exp_suffix}.log")
        
        # Create failed response save directory
        self.failed_responses_dir = os.path.join(self.logs_dir, "failed_responses")
        os.makedirs(self.failed_responses_dir, exist_ok=True)
    
    def log_api_request(self, prompt, response, metadata=None):
        """Log API request and response"""
        timestamp = datetime.now().isoformat()
        with open(self.api_log, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Time: {timestamp}\n")
            f.write(f"Label: {self.target_label}\n")
            if self.experiment_num:
                f.write(f"Experiment: {self.experiment_num}\n")
            if metadata:
                f.write(f"Metadata: {json.dumps(metadata, ensure_ascii=False, indent=2)}\n")
            f.write(f"Prompt length: {len(prompt)} characters\n")
            f.write(f"Prompt content:\n{prompt}\n")
            f.write(f"\n{'-'*40} Response {'-'*40}\n")
            f.write(f"Response length: {len(response)} characters\n")
            f.write(f"Response content:\n{response}\n")
            f.write(f"{'='*80}\n\n")
    
    def log_processing(self, message, level="INFO"):
        """Log processing process"""
        timestamp = datetime.now().isoformat()
        with open(self.processing_log, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] [{level}] {message}\n")
    
    def log_error(self, error_message, exception=None, context=None):
        """Log error information"""
        timestamp = datetime.now().isoformat()
        with open(self.error_log, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Time: {timestamp}\n")
            f.write(f"Label: {self.target_label}\n")
            if self.experiment_num:
                f.write(f"Experiment: {self.experiment_num}\n")
            f.write(f"Error: {error_message}\n")
            if context:
                f.write(f"Context: {json.dumps(context, ensure_ascii=False, indent=2)}\n")
            if exception:
                f.write(f"Exception details:\n{traceback.format_exc()}\n")
            f.write(f"{'='*80}\n\n")
    
    def save_failed_response(self, response, error_message, context=None):
        """Save failed API response"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_suffix = f"_exp{self.experiment_num}" if self.experiment_num else ""
        filename = f"failed_response_{self.target_label}{exp_suffix}_{timestamp}.json"
        filepath = os.path.join(self.failed_responses_dir, filename)
        
        failed_data = {
            "timestamp": datetime.now().isoformat(),
            "target_label": self.target_label,
            "experiment_num": self.experiment_num,
            "error_message": error_message,
            "response_length": len(response),
            "raw_response": response,
            "context": context or {}
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(failed_data, f, ensure_ascii=False, indent=2)
        
        return filepath

def validate_optimization_result(result, target_label):
    """
    Validate the completeness and validity of the optimization result
    
    Args:
        result (dict): optimization result
        target_label (str): target label
        
    Returns:
        tuple: (whether valid, error message)
    """
    if "error" in result:
        return False, f"Result contains error: {result.get('error', 'Unknown error')}"
    
    if target_label not in result:
        return False, f"Missing result data for target label '{target_label}'"
    
    label_data = result[target_label]
    if not isinstance(label_data, dict):
        return False, f"Data format error for label '{target_label}' (should be a dictionary)"
    
    # Check required fields
    required_fields = ["added_features", "removed_features"]
    for field in required_fields:
        if field not in label_data:
            return False, f"Missing required field '{field}'"
        
        if not isinstance(label_data[field], list):
            return False, f"Field '{field}' should be a list format"
    
    # Validate the completeness of feature items
    for field_name in ["added_features", "removed_features"]:
        features = label_data[field_name]
        for i, feature_item in enumerate(features):
            if not isinstance(feature_item, dict):
                return False, f"{field_name}[{i}] should be a dictionary format"
            
            if "feature" not in feature_item:
                return False, f"{field_name}[{i}] missing 'feature' field"
            
            if "reason" not in feature_item:
                return False, f"{field_name}[{i}] missing 'reason' field"
            
            # Check that the field value is not empty
            if not feature_item["feature"] or not feature_item["feature"].strip():
                return False, f"{field_name}[{i}] 'feature' field is empty"
            
            if not feature_item["reason"] or not feature_item["reason"].strip():
                return False, f"{field_name}[{i}] 'reason' field is empty"
    
    return True, "Completeness validation passed"

def calculate_optimized_features(original_features, added_features, removed_features):
    """
    Automatically calculate the optimized feature set based on original features and add/remove operations
    
    Args:
        original_features (list): original feature description list
        added_features (list): new feature list [{"feature": "...", "reason": "..."}]
        removed_features (list): deleted feature list [{"feature": "...", "reason": "..."}]
        
    Returns:
        tuple: (optimized feature list, validation information dictionary)
    """
    # Extract feature names
    added_feature_names = [item["feature"] for item in added_features if isinstance(item, dict) and "feature" in item]
    removed_feature_names = [item["feature"] for item in removed_features if isinstance(item, dict) and "feature" in item]
    
    # Convert to set for operation
    original_set = set(original_features)
    added_set = set(added_feature_names)
    removed_set = set(removed_feature_names)
    
    # Validate that the deleted features exist in the original features
    invalid_removals = removed_set - original_set
    
    # Validate that the added features already exist in the original features
    duplicate_additions = added_set & original_set
    
    # Calculate optimized feature set
    optimized_set = original_set.copy()
    optimized_set -= removed_set  # Remove specified features
    optimized_set |= added_set    # Add new features
    
    # Generate validation information
    validation_info = {
        "original_count": len(original_features),
        "added_count": len(added_feature_names),
        "removed_count": len(removed_feature_names),
        "optimized_count": len(optimized_set),
        "invalid_removals": list(invalid_removals),
        "duplicate_additions": list(duplicate_additions),
        "net_change": len(optimized_set) - len(original_features)
    }
    
    return list(optimized_set), validation_info

def fix_json_syntax_errors(json_text, logger=None):
    """
    Advanced JSON syntax error fixer
    
    Special handling for common JSON errors in qwen3 and other models:
    - Missing comma separator
    - Unescaped quotes in strings
    - Incomplete JSON structure
    - Field value truncation issues
    
    Args:
        json_text (str): JSON text to be fixed
        logger (OptimizationLogger, optional): Logger
        
    Returns:
        tuple: (fixed JSON string, repair steps information)
    """
    if logger:
        logger.log_processing("Starting advanced JSON syntax repair", "INFO")
    
    original_text = json_text
    repair_steps = []
    
    # Repair strategy 1: Add missing comma separators
    try:
        # Handle cases where }{ is missing a comma
        fixed_text = re.sub(r'}\s*{', '},{', json_text)
        
        # Handle cases where ] [ is missing a comma
        fixed_text = re.sub(r']\s*\[', '],[', fixed_text)
        
        # Handle cases where object properties are missing a comma
        # Match "value" \n "key": pattern
        fixed_text = re.sub(r'"\s*\n\s*"([^"]+)":', '",\n  "\\1":', fixed_text)
        
        if fixed_text != json_text:
            repair_steps.append("Repair 1: Add missing comma separators")
            json_text = fixed_text
            
            # Try to parse the repaired JSON
            try:
                json.loads(json_text)
                repair_steps.append("Repair 1: Successfully parsed after comma repair")
                if logger:
                    logger.log_processing("Repair strategy 1: Comma problem fixed", "INFO")
                return json_text, repair_steps
            except json.JSONDecodeError:
                repair_steps.append("Repair 1: Still needs further processing after comma repair")
        
    except Exception as e:
        repair_steps.append(f"Repair 1: Error in comma repair - {str(e)}")
    
    # Repair strategy 2: Escape quotes in strings
    try:
        # Special handling for qwen3 model errors: quotes in reason field
        # Handle quotes conflict in "Clinical explanation: ..." pattern
        
        def fix_reason_quotes(text):
            """Special handling for quotes in reason field"""
            lines = text.split('\n')
            fixed_lines = []
            
            i = 0
            while i < len(lines):
                line = lines[i]
                
                # Find the line where the reason field starts
                if '"reason"' in line and 'Clinical explanation:' in line:
                    # Find the reason field, check if it spans multiple lines or has quote issues
                    reason_start = line.find('"reason"')
                    colon_pos = line.find(':', reason_start)
                    
                    if colon_pos != -1:
                        # Extract the value part of reason
                        value_start = line.find('"', colon_pos)
                        if value_start != -1:
                            value_start += 1  # Skip the starting quote
                            
                            # Check if the value ends on the same line
                            remaining_text = line[value_start:]
                            quote_count = remaining_text.count('"')
                            
                            if quote_count == 1:
                                # Normal case, one line complete
                                fixed_lines.append(line)
                            elif quote_count > 1:
                                # There are multiple quotes in the line, need to escape
                                # Find the last quote as the end
                                last_quote = remaining_text.rfind('"')
                                content = remaining_text[:last_quote]
                                # Escape internal quotes
                                escaped_content = content.replace('"', '\\"')
                                fixed_line = line[:value_start] + escaped_content + '"'
                                fixed_lines.append(fixed_line)
                            else:
                                # quote_count == 0, means the value spans multiple lines
                                # Collect cross-line content
                                content_lines = [remaining_text]
                                j = i + 1
                                
                                while j < len(lines):
                                    next_line = lines[j]
                                    content_lines.append(next_line)
                                    
                                    if '"' in next_line:
                                        # Find the end quote
                                        break
                                    j += 1
                                
                                # Merge content and escape quotes
                                full_content = ''.join(content_lines)
                                if '"' in full_content:
                                    end_quote_pos = full_content.rfind('"')
                                    actual_content = full_content[:end_quote_pos]
                                    escaped_content = actual_content.replace('"', '\\"')
                                    
                                    # Generate the fixed single line
                                    fixed_line = line[:value_start] + escaped_content + '"'
                                    fixed_lines.append(fixed_line)
                                    
                                    # Skip processed lines
                                    i = j
                                else:
                                    # No end quote found, add one
                                    escaped_content = full_content.replace('"', '\\"')
                                    fixed_line = line[:value_start] + escaped_content + '"'
                                    fixed_lines.append(fixed_line)
                                    i = j
                        else:
                            fixed_lines.append(line)
                    else:
                        fixed_lines.append(line)
                else:
                    fixed_lines.append(line)
                
                i += 1
            
            return '\n'.join(fixed_lines)
        
        # First apply basic quote repair
        pattern = r'"reason"\s*:\s*"([^"]*)"([^"]*)"([^"]*)"'
        
        def fix_quotes(match):
            full_content = match.group(1) + '"' + match.group(2) + '"' + match.group(3)
            # Escape internal quotes
            escaped_content = full_content.replace('"', '\\"')
            return f'"reason": "{escaped_content}"'
        
        fixed_text = re.sub(pattern, fix_quotes, json_text)
        
        # Apply specialized reason field repair
        reason_fixed = fix_reason_quotes(fixed_text)
        
        # Check the repair effect of each stage
        if fixed_text != json_text:
            json_text = fixed_text
            repair_steps.append("Repair 2a: Basic quote escape repair")
        
        if reason_fixed != json_text:
            json_text = reason_fixed
            repair_steps.append("Repair 2b: Special reason field repair")
            
            # Try to parse
            try:
                json.loads(json_text)
                repair_steps.append("Repair 2: Successfully parsed after quote repair")
                if logger:
                    logger.log_processing("Repair strategy 2: Quotes fixed", "INFO")
                return json_text, repair_steps
            except json.JSONDecodeError:
                repair_steps.append("Repair 2: Still needs further processing after quote repair")
        
    except Exception as e:
        repair_steps.append(f"Repair 2: Error in quote repair - {str(e)}")
    
    # Repair strategy 3: Complete incomplete JSON structure
    try:
        lines = json_text.split('\n')
        brace_count = 0
        bracket_count = 0
        in_string = False
        escape_next = False
        
        for char in json_text:
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
                
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                elif char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
        
        # Complete missing end symbols
        completions = []
        
        # Complete objects
        while brace_count > 0:
            completions.append('}')
            brace_count -= 1
        
        # Complete arrays
        while bracket_count > 0:
            completions.append(']')
            bracket_count -= 1
        
        if completions:
            json_text = json_text.rstrip() + ''.join(completions)
            repair_steps.append(f"Repair 3: Complete JSON structure - added {''.join(completions)}")
            
            # Try to parse
            try:
                json.loads(json_text)
                repair_steps.append("Repair 3: Successfully parsed after structure completion")
                if logger:
                    logger.log_processing("Repair strategy 3: JSON structure completed", "INFO")
                return json_text, repair_steps
            except json.JSONDecodeError:
                repair_steps.append("Repair 3: Still needs further processing after structure completion")
    
    except Exception as e:
        repair_steps.append(f"Repair 3: Error in structure completion - {str(e)}")
    
    # Repair strategy 4: Handle truncated field values
    try:
        # Find the last field that may be truncated
        lines = json_text.split('\n')
        
        # Find incomplete fields from the end
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()
            
            # If the line starts with a quote but doesn't end correctly
            if line.startswith('"') and line.count('"') % 2 == 1:
                # It may be a truncated field, try to complete
                if not line.endswith('"'):
                    lines[i] = line + '"'
                    repair_steps.append("Repair 4a: Complete truncated field values")
                break
            
            # If the line doesn't end correctly (missing comma or end symbol)
            if line and not line.endswith((',', '}', ']', '"')):
                if '"reason"' in line:
                    # This may be a truncated reason field
                    lines[i] = line + '"'
                    repair_steps.append("Repair 4b: Complete truncated reason field")
                break
        
        fixed_text = '\n'.join(lines)
        
        if fixed_text != json_text:
            json_text = fixed_text
            
            # Try to parse
            try:
                json.loads(json_text)
                repair_steps.append("Repair 4: Successfully parsed after truncation repair")
                if logger:
                    logger.log_processing("Repair strategy 4: Truncation problem fixed", "INFO")
                return json_text, repair_steps
            except json.JSONDecodeError:
                repair_steps.append("Repair 4: Still needs further processing after truncation repair")
    
    except Exception as e:
        repair_steps.append(f"Repair 4: Error in truncation repair - {str(e)}")
    
    # Repair strategy 5: Remove extra content and standardize format
    try:
        # Remove extra commas
        fixed_text = re.sub(r',(\s*[}\]])', r'\1', json_text)
        
        # Remove extra commas at the end of lines
        fixed_text = re.sub(r',\s*\n\s*[}\]]', '\n}', fixed_text)
        
        # Standardize quote format
        fixed_text = fixed_text.replace('"', '"').replace('"', '"')
        
        if fixed_text != json_text:
            json_text = fixed_text
            repair_steps.append("Repair 5: Remove extra content and standardize format")
            
            # Try to parse
            try:
                json.loads(json_text)
                repair_steps.append("Repair 5: Successfully parsed after format standardization")
                if logger:
                    logger.log_processing("Repair strategy 5: Format standardized", "INFO")
                return json_text, repair_steps
            except json.JSONDecodeError:
                repair_steps.append("Repair 5: Still needs further processing after format standardization")
        
    except Exception as e:
        repair_steps.append(f"Repair 5: Error in format standardization - {str(e)}")
    
    # All repair strategies failed
    repair_steps.append("All repair strategies failed: All repair strategies failed")
    if logger:
        logger.log_processing("All JSON repair strategies failed", "ERROR")
        logger.log_processing(f"Repair steps: {'; '.join(repair_steps)}", "ERROR")
    
    return original_text, repair_steps

def clean_and_extract_json(response_text, logger=None):
    """
    Smart extraction and cleaning of JSON content, supporting multiple model response formats
    
    Formats to be processed:
    - Markdown code block wrapping (```json ... ```)
    - Extra text before and after the response
    - Incomplete JSON endings
    - Format differences between models
    - JSON syntax errors (new)
    
    Args:
        response_text (str): Original API response text
        logger (OptimizationLogger, optional): Logger
        
    Returns:
        tuple: (Cleaned JSON string, cleaning steps information)
    """
    if logger:
        logger.log_processing("Starting JSON cleaning and extraction process", "INFO")
    
    original_text = response_text
    cleaning_steps = []
    
    # Strategy 1: Direct try to parse
    try:
        json.loads(response_text)
        cleaning_steps.append("Strategy 1: Original response parsed successfully")
        if logger:
            logger.log_processing("Strategy 1: Original response can be parsed directly", "INFO")
        return response_text, cleaning_steps
    except json.JSONDecodeError:
        cleaning_steps.append("Strategy 1: Original response parsing failed, try to clean")
        
    # Strategy 2: Remove markdown code block markers
    # Process ```json ... ``` format
    markdown_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    markdown_match = re.search(markdown_pattern, response_text, re.DOTALL)
    
    if markdown_match:
        response_text = markdown_match.group(1).strip()
        cleaning_steps.append("Strategy 2: Remove markdown code block markers")
        if logger:
            logger.log_processing("Strategy 2: Removed markdown code block markers", "INFO")
        
        try:
            json.loads(response_text)
            cleaning_steps.append("Strategy 2: Parsed successfully after removing markdown")
            return response_text, cleaning_steps
        except json.JSONDecodeError:
            cleaning_steps.append("Strategy 2: Parsing failed after removing markdown")
    
    # Strategy 3: Find JSON object boundaries
    # Find the first { and the last }
    first_brace = response_text.find('{')
    last_brace = response_text.rfind('}')
    
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        extracted_json = response_text[first_brace:last_brace + 1]
        cleaning_steps.append("Strategy 3: Extract JSON object boundaries")
        if logger:
            logger.log_processing(f"Strategy 3: Extract JSON boundaries [{first_brace}:{last_brace+1}]", "INFO")
        
        try:
            json.loads(extracted_json)
            cleaning_steps.append("Strategy 3: Parsed successfully after extracting boundaries")
            return extracted_json, cleaning_steps
        except json.JSONDecodeError:
            cleaning_steps.append("Strategy 3: Parsing failed after extracting boundaries")
            # Try to fix the extracted JSON syntax
            response_text = extracted_json
    
    # Strategy 4: Advanced JSON syntax repair (new)
    if response_text != original_text or True:  # Always try to fix
        fixed_json, repair_steps = fix_json_syntax_errors(response_text, logger)
        
        if repair_steps:
            cleaning_steps.extend(repair_steps)
            
            # If the repair is successful, return the repaired JSON
            if any("Parsed successfully" in step for step in repair_steps):
                cleaning_steps.append("Strategy 4: JSON syntax repair successful")
                return fixed_json, cleaning_steps
            else:
                cleaning_steps.append("Strategy 4: JSON syntax repair failed, continue with other strategies")
                response_text = fixed_json  # Use the repaired version to continue
    
    # Strategy 5: Find complete JSON line by line
    lines = response_text.split('\n')
    json_start = -1
    json_end = -1
    brace_count = 0
    
    for i, line in enumerate(lines):
        for char in line:
            if char == '{':
                if json_start == -1:
                    json_start = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and json_start != -1:
                    json_end = i
                    break
        if json_end != -1:
            break
    
    if json_start != -1 and json_end != -1:
        json_lines = lines[json_start:json_end + 1]
        extracted_json = '\n'.join(json_lines)
        cleaning_steps.append("Strategy 5: Find complete JSON line by line")
        if logger:
            logger.log_processing(f"Strategy 5: Extract JSON [line {json_start}:{json_end+1}]", "INFO")
        
        try:
            json.loads(extracted_json)
            cleaning_steps.append("Strategy 5: Parsed successfully after line by line extraction")
            return extracted_json, cleaning_steps
        except json.JSONDecodeError:
            cleaning_steps.append("Strategy 5: Parsing failed after line by line extraction")
    
    # Strategy 6: Regex pattern extraction of JSON
    # Match standard JSON object structure
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    json_matches = re.findall(json_pattern, response_text, re.DOTALL)
    
    for i, match in enumerate(json_matches):
        try:
            json.loads(match)
            cleaning_steps.append(f"Strategy 6: Regex extraction of the {i+1}th JSON successfully")
            if logger:
                logger.log_processing(f"Strategy 6: Regex extraction of the {i+1}th JSON successfully", "INFO")
            return match, cleaning_steps
        except json.JSONDecodeError:
            continue
    
    # Strategy 7: Try to fix common JSON format problems
    # Remove trailing commas, fix quotes, etc.
    cleaned_text = response_text.strip()
    
    # Remove trailing extra commas
    cleaned_text = re.sub(r',(\s*[}\]])', r'\1', cleaned_text)
    
    # Try to fix quote problems
    cleaned_text = cleaned_text.replace('"', '"').replace('"', '"')
    
    cleaning_steps.append("Strategy 7: Try to fix common JSON format problems")
    
    try:
        json.loads(cleaned_text)
        cleaning_steps.append("Strategy 7: Parsed successfully after format repair")
        if logger:
            logger.log_processing("Strategy 7: Format repair successful", "INFO")
        return cleaned_text, cleaning_steps
    except json.JSONDecodeError:
        cleaning_steps.append("Strategy 7: Parsing failed after format repair")
    
    # All strategies failed
    cleaning_steps.append("All strategies failed: All cleaning strategies failed")
    if logger:
        logger.log_processing("All JSON cleaning strategies failed", "ERROR")
        logger.log_processing(f"Original response length: {len(original_text)}", "ERROR")
        logger.log_processing(f"Cleaning steps: {'; '.join(cleaning_steps)}", "ERROR")
    
    return original_text, cleaning_steps

def build_optimization_prompt(algorithm_name, target_label, target_feature_descriptions, all_feature_descriptions):
    """
    Build English prompt for feature optimization
    
    Args:
        algorithm_name (str): Algorithm name (CORL or DirectLiNGAM)
        target_label (str): Target label name
        target_feature_descriptions (list): Target feature description list
        all_feature_descriptions (list): Complete feature description list
        
    Returns:
        str: Built prompt
    """
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Building {algorithm_name} algorithm {target_label} optimization prompt...")
    
    # Format feature list to list format
    all_features_str = str(all_feature_descriptions)
    target_features_str = str(target_feature_descriptions)
    
    # Get the clinical description of the target label
    outcome_descriptions = get_outcome_descriptions()
    target_description = outcome_descriptions.get(target_label, target_label)
    
    prompt = f"""You are a clinical expert with extensive experience in intensive care medicine. Your task is to optimize the causal feature sets for various clinical prognostic outcomes generated by the {algorithm_name} causal discovery algorithm, based on established clinical knowledge and clinical experience.

1. All Available Features ({len(all_feature_descriptions)} features):
{all_features_str}

2. Feature Sets to be Optimized:
{target_features_str}

3. Your Task and Output Requirements:

Target Outcome Clinical Context:
{target_label}: {target_description}

Please consider the specific pathophysiology and clinical risk factors associated with {target_label} when making your optimization decisions.

Complete the following tasks:
1. Add features that you believe are clinically causally related to {target_label} but were missed.
2. Remove features that you believe are not clinically causally related to {target_label}.
3. Return ONLY the changes (additions and removals) in the following JSON format. You must provide clear, concise clinical explanations for each decision.

{{
  "{target_label}": {{
    "added_features": [
      {{
        "feature": "Suggested_Addition_Feature_Description",
        "reason": "Clinical explanation: Why this feature is crucial for predicting this outcome, even though the algorithm didn't select it."
      }}
    ],
    "removed_features": [
      {{
        "feature": "Suggested_Removal_Feature_Description", 
        "reason": "Clinical explanation: Why this feature should be removed (e.g., redundancy, weak correlation, target leakage, etc.)."
      }}
    ]
  }}
}}

Please ensure your response is valid JSON format and includes clinical reasoning for all modifications."""

    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Prompt built, total length: {len(prompt)} characters")
    print(prompt)
    return prompt

def optimize_features_for_label(model_config, features_file, target_label, session_dir=None, experiment_num=None):
    """
    Optimize features for a specified label (enhanced version: includes logging, retry mechanism, and automatic calculation of optimized feature sets)
    
    Args:
        model_config (dict): Model configuration
        features_file (str): Feature file name
        target_label (str): Target label
        session_dir (str, optional): Session directory path, for logging
        experiment_num (int, optional): Experiment number
        
    Returns:
        dict: Optimization result
    """
    print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Starting to optimize features for label '{target_label}'")
    
    # Initialize logger
    logger = None
    if session_dir:
        logger = OptimizationLogger(session_dir, target_label, experiment_num)
        logger.log_processing(f"Starting optimization experiment - algorithm: {features_file}, model: {model_config['model_name']}")
    
    try:
        # 1. Get algorithm name
        algorithm_name = features_file.split('_')[0]  # CORL_F.txt → CORL
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Algorithm: {algorithm_name}")
        if logger:
            logger.log_processing(f"Algorithm name: {algorithm_name}")
        
        # 2. Get complete feature description pool
        all_feature_descriptions, all_feature_names = get_all_available_feature_descriptions()
        if logger:
            logger.log_processing(f"Loading available features: {len(all_feature_descriptions)}")
        
        # 3. Get target feature description
        target_feature_descriptions, target_feature_names = get_target_feature_descriptions(features_file, target_label)
        if logger:
            logger.log_processing(f"Loading target features: {len(target_feature_descriptions)}")
        
        # 4. Build prompt
        prompt = build_optimization_prompt(algorithm_name, target_label, target_feature_descriptions, all_feature_descriptions)
        if logger:
            logger.log_processing(f"Build prompt completed, length: {len(prompt)} characters")
        
        # 5. API call with retry mechanism
        print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Calling {model_config['model_name']} API...")
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Prompt length: {len(prompt)} characters")
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Target feature number: {len(target_feature_descriptions)}")
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Available feature number: {len(all_feature_descriptions)}")
        
        max_retries = 3
        response = None
        last_error = None
        
        for attempt in range(max_retries):
            try:
                if logger:
                    logger.log_processing(f"API call attempt {attempt + 1}/{max_retries}")
                
                response = query_model(prompt, model_config)
                print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} API call successful, response length: {len(response)} characters")
                
                if logger:
                    logger.log_api_request(prompt, response, {
                        "attempt": attempt + 1,
                        "prompt_length": len(prompt),
                        "response_length": len(response)
                    })
                    logger.log_processing(f"API call successful, response length: {len(response)} characters")
                
                break  # If successful, break the retry loop
                
            except Exception as api_error:
                last_error = api_error
                print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} API call failed (attempt {attempt + 1}/{max_retries}): {str(api_error)}")
                
                if logger:
                    logger.log_error(f"API call failed (attempt {attempt + 1}/{max_retries})", 
                                   exception=api_error, 
                                   context={"prompt_length": len(prompt)})
                
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # Incremental waiting time: 2, 4, 6 seconds
                    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
        
        if response is None:
            error_msg = f"API call failed, retried {max_retries} times"
            if logger:
                logger.log_error(error_msg, exception=last_error)
            
            return {
                "error": error_msg,
                "last_error": str(last_error) if last_error else "Unknown error",
                "metadata": {
                    "algorithm": algorithm_name,
                    "features_file": features_file,
                    "target_label": target_label,
                    "model_name": model_config["model_name"],
                    "timestamp": datetime.now().isoformat(),
                    "status": "api_failed",
                    "retry_attempts": max_retries
                }
            }
        
        # 6. Smart JSON parsing (supports multiple model formats)
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Cleaning and parsing JSON response...")
        if logger:
            logger.log_processing("Starting smart JSON parsing process")
        
        # Use smart JSON cleaner
        cleaned_json, cleaning_steps = clean_and_extract_json(response, logger)
        
        # Record cleaning process
        if logger:
            logger.log_processing(f"JSON cleaning steps: {'; '.join(cleaning_steps)}")
        
        # Try to parse the cleaned JSON
        try:
            result = json.loads(cleaned_json)
            print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} JSON parsing successful")
            if logger:
                logger.log_processing(f"JSON parsing successful, cleaning strategy effective: {cleaning_steps[-1] if cleaning_steps else 'direct'}")
            
            # 7. Validate result completeness
            is_valid, validation_error = validate_optimization_result(result, target_label)
            
            if not is_valid:
                error_msg = f"Completeness validation failed: {validation_error}"
                print(f"{Colors.RED}[ERROR]{Colors.RESET} {error_msg}")
                
                if logger:
                    logger.log_error(error_msg, context={
                        "validation_error": validation_error,
                        "response_length": len(response),
                        "cleaned_response_length": len(cleaned_json),
                        "cleaning_steps": cleaning_steps,
                        "parsed_result_keys": list(result.keys()) if isinstance(result, dict) else "not_dict"
                    })
                    failed_file = logger.save_failed_response(response, error_msg, {
                        "validation_error": validation_error,
                        "parsed_result": result,
                        "cleaning_steps": cleaning_steps
                    })
                    logger.log_processing(f"Completeness validation failed response saved: {failed_file}")
                
                return {
                    "error": "Completeness validation failed",
                    "validation_error": validation_error,
                    "raw_response": response,
                    "cleaned_response": cleaned_json,
                    "parsed_result": result,
                    "cleaning_steps": cleaning_steps,
                    "metadata": {
                        "algorithm": algorithm_name,
                        "features_file": features_file,
                        "target_label": target_label,
                        "model_name": model_config["model_name"],
                        "timestamp": datetime.now().isoformat(),
                        "status": "validation_failed",
                        "experiment_num": experiment_num,
                        "response_lengths": {
                            "original": len(response),
                            "cleaned": len(cleaned_json)
                        }
                    }
                }
            
            print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Completeness validation passed")
            if logger:
                logger.log_processing("Completeness validation passed, all required fields are complete")
            
            # 8. Automatically calculate optimized feature set
            if target_label in result:
                added_features = result[target_label].get("added_features", [])
                removed_features = result[target_label].get("removed_features", [])
                
                # Calculate optimized feature set
                optimized_features, validation_info = calculate_optimized_features(
                    target_feature_descriptions, added_features, removed_features
                )
                
                # Add to result
                result[target_label]["optimized_feature_set"] = optimized_features
                result[target_label]["validation_info"] = validation_info
                
                if logger:
                    logger.log_processing(f"Automatic calculation of optimized feature set completed: {validation_info}")
                
                # Print validation information
                if validation_info["invalid_removals"]:
                    print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Invalid deletion (not in original features): {validation_info['invalid_removals']}")
                if validation_info["duplicate_additions"]:
                    print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Duplicate addition (already in original features): {validation_info['duplicate_additions']}")
                
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} Feature change: {validation_info['original_count']} → {validation_info['optimized_count']} "
                      f"(net change: {validation_info['net_change']:+d})")
            
            # 9. Add enhanced metadata (including validation and cleaning information)
            result["metadata"] = {
                "algorithm": algorithm_name,
                "features_file": features_file,
                "target_label": target_label,
                "model_name": model_config["model_name"],
                "timestamp": datetime.now().isoformat(),
                "original_target_features_count": len(target_feature_names),
                "available_features_count": len(all_feature_names),
                "prompt_length": len(prompt),
                "response_length": len(response),
                "experiment_num": experiment_num,
                "status": "success",
                "processing_version": "enhanced_v2.2_with_validation",
                "features_calculated_automatically": True,
                "json_cleaning_applied": True,
                "json_cleaning_steps": cleaning_steps,
                "cleaned_response_length": len(cleaned_json),
                "content_validation_passed": True,
                "validation_checks": [
                    "JSON syntax correctness",
                    "Target label existence", 
                    "Required field completeness",
                    "Feature item format correctness",
                    "Field value non-empty validation"
                ]
            }
            
            return result
            
        except json.JSONDecodeError as e:
            error_msg = f"Smart JSON parsing still failed: {str(e)}"
            print(f"{Colors.RED}[ERROR]{Colors.RESET} {error_msg}")
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} Original response: {response[:300]}...")
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} Cleaned response: {cleaned_json[:300]}...")
            
            # Save detailed failure information
            if logger:
                failed_file = logger.save_failed_response(response, error_msg, {
                    "json_error": str(e),
                    "original_response_preview": response[:1000],
                    "cleaned_response_preview": cleaned_json[:1000],
                    "cleaning_steps": cleaning_steps,
                    "model_name": model_config["model_name"]
                })
                logger.log_error(error_msg, exception=e, context={
                    "original_response_length": len(response),
                    "cleaned_response_length": len(cleaned_json),
                    "cleaning_steps": cleaning_steps,
                    "failed_response_saved": failed_file
                })
            
            # Return detailed failure information
            return {
                "error": "Smart JSON parsing failed",
                "json_error": str(e),
                "raw_response": response,
                "cleaned_response": cleaned_json,
                "cleaning_steps": cleaning_steps,
                "metadata": {
                    "algorithm": algorithm_name,
                    "features_file": features_file,
                    "target_label": target_label,
                    "model_name": model_config["model_name"],
                    "timestamp": datetime.now().isoformat(),
                    "status": "smart_json_parse_failed",
                    "experiment_num": experiment_num,
                    "original_response_length": len(response),
                    "cleaned_response_length": len(cleaned_json),
                    "cleaning_attempts": len(cleaning_steps)
                }
            }
            
    except Exception as e:
        error_msg = f"Optimization process error: {str(e)}"
        print(f"{Colors.RED}[ERROR]{Colors.RESET} {error_msg}")
        
        if logger:
            logger.log_error(error_msg, exception=e)
        
        traceback.print_exc()
        
        return {
            "error": "Optimization process failed",
            "error_message": str(e),
            "metadata": {
                "features_file": features_file,
                "target_label": target_label,
                "model_name": model_config.get("model_name", "unknown"),
                "timestamp": datetime.now().isoformat(),
                "status": "process_error",
                "experiment_num": experiment_num
            }
        }

def save_optimization_result(result, output_dir, features_file, model_name, target_label, experiment_num=None, session_timestamp=None):
    """
    Save optimization result to file
    
    Args:
        result (dict): optimization result
        output_dir (str): output directory
        features_file (str): feature file name
        model_name (str): model name
        target_label (str): target label
        experiment_num (int, optional): experiment number
        session_timestamp (str, optional): session timestamp, used for uniform folder name
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate file name
    if session_timestamp is None:
        session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    algorithm_name = features_file.split('_')[0]
    clean_model_name = model_name.replace(" ", "_").replace("-", "_")
    
    # If there is an experiment number, add it to the file name
    if experiment_num is not None:
        output_file = f"{algorithm_name}_{clean_model_name}_{target_label}_optimization_exp{experiment_num}_{session_timestamp}.json"
    else:
        output_file = f"{algorithm_name}_{clean_model_name}_{target_label}_optimization_{session_timestamp}.json"
    
    save_path = os.path.join(output_dir, output_file)
    
    # Save result
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Result saved: {output_file}")
    return save_path

def generate_session_summary(all_results, config, session_dir, original_features, session_timestamp, target_label):
    """
    Generate session summary file
    
    Args:
        all_results (list): all experiment results list
        config (dict): optimization configuration
        session_dir (str): session directory path
        original_features (list): original feature list
        session_timestamp (str): session timestamp
        target_label (str): target label name
    """
    algorithm_name = config['features_file'].split('_')[0]
    
    # Generate summary file name
    summary_file = os.path.join(session_dir, f"session_summary_{session_timestamp}.txt")
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        # Write title and basic information
        f.write("="*80 + "\n")
        f.write("Feature optimization session summary report\n")
        f.write("="*80 + "\n\n")
        
        # Configuration information
        f.write("Configuration information:\n")
        f.write("-"*50 + "\n")
        f.write(f"Algorithm: {algorithm_name}\n")
        f.write(f"Feature file: {config['features_file']}\n")
        f.write(f"Model: {config['model_config']['model_name']}\n")
        f.write(f"Target label: {target_label}\n")
        f.write(f"Experiment count: {config['experiment_count']}\n")
        f.write(f"Session time: {session_timestamp}\n")
        f.write(f"Temperature setting: {config['model_config']['generation_params']['temperature']}\n\n")
        
        # Original feature information
        f.write(f"{Colors.WHITE}[INFO]{Colors.RESET} Original feature set:\n")
        f.write("-"*50 + "\n")
        f.write(f"Original feature count: {len(original_features)}\n")
        f.write("Original feature list:\n")
        for i, feature in enumerate(original_features, 1):
            f.write(f"  {i:2d}. {feature}\n")
        f.write("\n")
        
        # Analyze the result of each experiment
        successful_experiments = []
        failed_experiments = []
        
        for i, result in enumerate(all_results, 1):
            if "error" in result:
                failed_experiments.append(i)
            else:
                successful_experiments.append(result)
        
        f.write(f"{Colors.WHITE}[INFO]{Colors.RESET} Experiment result statistics:\n")
        f.write("-"*50 + "\n")
        f.write(f"Total experiment count: {len(all_results)}\n")
        f.write(f"Successful experiments: {len(successful_experiments)}\n")
        f.write(f"Failed experiments: {len(failed_experiments)}\n")
        if failed_experiments:
            f.write(f"Failed experiment numbers: {', '.join(map(str, failed_experiments))}\n")
        f.write(f"Success rate: {len(successful_experiments)/len(all_results)*100:.1f}%\n\n")
        
        if successful_experiments:
            # Summarize all successful experiments' feature changes
            all_added_features = []
            all_removed_features = []
            optimized_feature_sets = []
            
            for result in successful_experiments:
                if target_label in result:
                    label_result = result[target_label]
                    
                    # Collect optimized feature sets
                    optimized_set = label_result.get("optimized_feature_set", [])
                    optimized_feature_sets.append(optimized_set)
                    
                    # Collect new features
                    added_features = label_result.get("added_features", [])
                    for feature_info in added_features:
                        all_added_features.append({
                            "feature": feature_info.get("feature", ""),
                            "reason": feature_info.get("reason", ""),
                            "experiment": len(optimized_feature_sets)
                        })
                    
                    # Collect deleted features
                    removed_features = label_result.get("removed_features", [])
                    for feature_info in removed_features:
                        all_removed_features.append({
                            "feature": feature_info.get("feature", ""),
                            "reason": feature_info.get("reason", ""),
                            "experiment": len(optimized_feature_sets)
                        })
            
            # Write feature change summary
            f.write(f"{Colors.WHITE}[INFO]{Colors.RESET} Feature change summary:\n")
            f.write("-"*50 + "\n")
            f.write(f"New feature count: {len(all_added_features)}\n")
            f.write(f"Deleted feature count: {len(all_removed_features)}\n\n")
            
            # New feature details (if any)
            if all_added_features:
                f.write("New feature details:\n")
                f.write("-"*30 + "\n")
                for i, feature_info in enumerate(all_added_features, 1):
                    f.write(f"{i:2d}. [Experiment {feature_info['experiment']}] {feature_info['feature']}\n")
                    f.write(f"    Reason: {feature_info['reason']}\n\n")
            else:
                f.write("New feature: None\n\n")
            
            # Deleted feature details (if any)
            if all_removed_features:
                f.write("Deleted feature details:\n")
                f.write("-"*30 + "\n")
                for i, feature_info in enumerate(all_removed_features, 1):
                    f.write(f"{i:2d}. [Experiment {feature_info['experiment']}] {feature_info['feature']}\n")
                    f.write(f"    Reason: {feature_info['reason']}\n\n")
            else:
                f.write("Deleted feature: None\n\n")
            
            # Optimized feature set statistics
            f.write(f"{Colors.WHITE}[INFO]{Colors.RESET} Optimized feature set statistics:\n")
            f.write("-"*50 + "\n")
            for i, optimized_set in enumerate(optimized_feature_sets, 1):
                f.write(f"Experiment {i} - Optimized feature count: {len(optimized_set)}\n")
            
            if optimized_feature_sets:
                avg_features = sum(len(s) for s in optimized_feature_sets) / len(optimized_feature_sets)
                f.write(f"Average optimized feature count: {avg_features:.1f}\n")
                f.write(f"Original feature count: {len(original_features)}\n")
                f.write(f"Average change: {avg_features - len(original_features):+.1f}\n\n")
            
            # Feature frequency statistics (if there are multiple experiments)
            if len(optimized_feature_sets) > 1:
                f.write("Feature frequency statistics:\n")
                f.write("-"*50 + "\n")
                
                # Count how many times each feature appears in each experiment
                feature_counts = {}
                for optimized_set in optimized_feature_sets:
                    for feature in optimized_set:
                        feature_counts[feature] = feature_counts.get(feature, 0) + 1
                
                # Sort by frequency
                sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
                
                f.write(f"Total feature count (unique): {len(sorted_features)}\n")
                f.write("Feature frequency (high to low):\n")
                for feature, count in sorted_features:
                    percentage = count / len(optimized_feature_sets) * 100
                    f.write(f"  {feature} - {count}/{len(optimized_feature_sets)} times ({percentage:.0f}%)\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Report generated successfully\n")
        f.write("="*80 + "\n")
    
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Session summary saved: {os.path.basename(summary_file)}")
    return summary_file

def generate_consensus_features(all_results, config, session_dir, session_timestamp, target_label):
    """
    Generate consensus feature set based on multiple experiments (supports strict mode quality control)
    
    Args:
        all_results (list): all experiment results list
        config (dict): optimization configuration
        session_dir (str): session directory path
        session_timestamp (str): session timestamp
        target_label (str): target label name
        
    Returns:
        str: consensus file path, return None if not met
    """
    algorithm_name = config['features_file'].split('_')[0]
    clean_model_name = config['model_config']['model_name'].replace(" ", "_").replace("-", "_")
    
    print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Generating consensus feature set...")
    
    # 1. Collect all successful experiments' optimized feature sets
    optimized_feature_sets = []
    successful_experiments = []
    total_experiments = len(all_results)
    
    for i, result in enumerate(all_results, 1):
        if "error" not in result and target_label in result:
            optimized_set = result[target_label].get("optimized_feature_set", [])
            if optimized_set:  # Ensure feature set is not empty
                optimized_feature_sets.append(optimized_set)
                successful_experiments.append(i)
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} Experiment {i}: {len(optimized_set)} features")
    
    successful_count = len(optimized_feature_sets)
    success_rate = successful_count / total_experiments if total_experiments > 0 else 0
    
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Experiment success statistics: {successful_count}/{total_experiments} ({success_rate*100:.1f}%)")
    
    # 2. Strict mode quality control check
    strict_mode = config.get("strict_mode", False)
    min_success_rate = config.get("min_success_rate", 1.0)
    allow_partial_consensus = config.get("allow_partial_consensus", True)
    
    if strict_mode:
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Strict mode enabled (requires success rate ≥ {min_success_rate*100:.0f}%)")
        
        if success_rate < min_success_rate:
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Quality control failed: Success rate {success_rate*100:.1f}% < required {min_success_rate*100:.0f}%")
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} Actual success: {successful_count} experiments")
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} Required success: {int(total_experiments * min_success_rate)} experiments")
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} Gap: {int(total_experiments * min_success_rate) - successful_count} experiments")
            print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Skipping consensus generation for {target_label}")
            
            # Generate failure explanation file
            failure_file = os.path.join(session_dir, f"consensus_generation_failed_{session_timestamp}.txt")
            with open(failure_file, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write(f"Consensus generation failed report - {target_label}\n")
                f.write("="*80 + "\n\n")
                f.write(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Quality control check failed\n\n")
                f.write(f"Configuration requirements:\n")
                f.write(f"Strict mode: {strict_mode}\n")
                f.write(f"Minimum success rate: {min_success_rate*100:.0f}%\n")
                f.write(f"Allow partial consensus: {allow_partial_consensus}\n\n")
                f.write(f"Actual results:\n")
                f.write(f"Total experiments: {total_experiments}\n")
                f.write(f"Successful experiments: {successful_count}\n")
                f.write(f"Actual success rate: {success_rate*100:.1f}%\n")
                f.write(f"Required success: {int(total_experiments * min_success_rate)} experiments\n")
                f.write(f"Gap: {int(total_experiments * min_success_rate) - successful_count} experiments\n\n")
                f.write(f"Suggestions:\n")
                f.write(f"  1. Check error logs of failed experiments\n")
                f.write(f"  2. Adjust model parameters or prompt\n")
                f.write(f"  3. Re-run failed experiments\n")
                f.write(f"  4. Or set strict_mode=False to allow partial success\n")
                f.write("\n" + "="*80 + "\n")
            
            print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Failure report saved: {os.path.basename(failure_file)}")
            return None
        else:
            print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Quality control passed: Success rate {success_rate*100:.1f}% ≥ required {min_success_rate*100:.0f}%")
    
    # 3. Check minimum experiment number requirements
    min_experiments_for_consensus = 2
    if successful_count < min_experiments_for_consensus:
        print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Insufficient successful experiments ({successful_count}<{min_experiments_for_consensus}), cannot generate reliable consensus")
        return None
    
    # 2. Count feature frequency
    feature_counts = {}
    for feature_set in optimized_feature_sets:
        for feature in feature_set:
            feature_counts[feature] = feature_counts.get(feature, 0) + 1
    
    # 3. Generate consensus feature set based on different thresholds
    consensus_threshold = 2  # At least 2 times
    consensus_features = [
        feature for feature, count in feature_counts.items() 
        if count >= consensus_threshold
    ]
    
    # 4. Generate detailed consensus analysis
    # Sort by frequency
    sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Classify features
    high_consensus = [(f, c) for f, c in sorted_features if c >= consensus_threshold]
    low_consensus = [(f, c) for f, c in sorted_features if c < consensus_threshold]
    
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Consensus analysis completed:")
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Total feature count (unique): {len(feature_counts)}")
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} High consensus features (≥{consensus_threshold} times): {len(high_consensus)}")
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Low consensus features (<{consensus_threshold} times): {len(low_consensus)}")
    
    # 5. Build consensus result
    consensus_result = {
        target_label: {
            "consensus_feature_set": consensus_features,
            "consensus_threshold": consensus_threshold,
            "total_successful_experiments": len(optimized_feature_sets),
            "successful_experiment_numbers": successful_experiments,
            "feature_frequency_analysis": {
                "high_consensus_features": high_consensus,
                "low_consensus_features": low_consensus,
                "all_feature_counts": feature_counts
            },
            "consensus_statistics": {
                "total_unique_features": len(feature_counts),
                "consensus_features_count": len(consensus_features),
                "consensus_rate": len(consensus_features) / len(feature_counts) * 100 if feature_counts else 0
            }
        },
        "metadata": {
            "algorithm": algorithm_name,
            "features_file": config['features_file'],
            "target_label": target_label,
            "model_name": config['model_config']['model_name'],
            "timestamp": datetime.now().isoformat(),
            "session_timestamp": session_timestamp,
            "generation_method": "consensus_from_multiple_experiments",
            "consensus_threshold": consensus_threshold,
            "total_experiments": len(all_results),
            "successful_experiments": len(optimized_feature_sets)
        }
    }
    
    # 6. Save consensus file
    consensus_file = f"{algorithm_name}_{clean_model_name}_{target_label}_consensus_{session_timestamp}.json"
    consensus_path = os.path.join(session_dir, consensus_file)
    
    with open(consensus_path, 'w', encoding='utf-8') as f:
        json.dump(consensus_result, f, ensure_ascii=False, indent=2)
    
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Consensus result saved: {consensus_file}")
    
    # 7. Generate consensus feature list file
    consensus_txt_file = f"{algorithm_name}_{clean_model_name}_{target_label}_consensus_{session_timestamp}.txt"
    consensus_txt_path = os.path.join(session_dir, consensus_txt_file)
    
    with open(consensus_txt_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"Consensus feature set - {target_label}\n")
        f.write("="*80 + "\n\n")
        
        f.write("Consensus configuration:\n")
        f.write("-"*50 + "\n")
        f.write(f"Algorithm: {algorithm_name}\n")
        f.write(f"Model: {config['model_config']['model_name']}\n")
        f.write(f"Target label: {target_label}\n")
        f.write(f"Consensus threshold: ≥{consensus_threshold} times\n")
        f.write(f"Successful experiments: {len(optimized_feature_sets)}\n")
        f.write(f"Session time: {session_timestamp}\n\n")
        
        f.write(f"{Colors.WHITE}[INFO]{Colors.RESET} Consensus statistics:\n")
        f.write("-"*50 + "\n")
        f.write(f"Total feature count (unique): {len(feature_counts)}\n")
        f.write(f"Consensus feature count: {len(consensus_features)}\n")
        f.write(f"Consensus ratio: {len(consensus_features) / len(feature_counts) * 100:.1f}%\n\n")
        
        f.write(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Consensus feature list:\n")
        f.write("-"*50 + "\n")
        for i, feature in enumerate(consensus_features, 1):
            count = feature_counts[feature]
            percentage = count / len(optimized_feature_sets) * 100
            f.write(f"{i:3d}. {feature} - {count}/{len(optimized_feature_sets)} times ({percentage:.0f}%)\n")
        
        if low_consensus:
            f.write(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Features not reaching consensus (<{consensus_threshold} times):\n")
            f.write("-"*50 + "\n")
            for feature, count in low_consensus:
                percentage = count / len(optimized_feature_sets) * 100
                f.write(f"   {feature} - {count}/{len(optimized_feature_sets)} times ({percentage:.0f}%)\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Consensus analysis completed\n")
        f.write("="*80 + "\n")
    
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Consensus feature list saved: {consensus_txt_file}")
    
    return consensus_path

def generate_multi_label_summary(all_label_results, config, main_output_dir, session_timestamp):
    """
    Generate overall summary report for multi-label optimization
    
    Args:
        all_label_results (dict): all label results statistics
        config (dict): optimization configuration
        main_output_dir (str): main output directory path
        session_timestamp (str): session timestamp
        
    Returns:
        str: summary report file path
    """
    algorithm_name = config['features_file'].split('_')[0]
    
    summary_file = os.path.join(main_output_dir, f"multi_label_optimization_summary_{session_timestamp}.txt")
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        # Write title and basic information
        f.write("="*80 + "\n")
        f.write("Multi-label feature optimization summary report\n")
        f.write("="*80 + "\n\n")
        
        # Configuration information
        f.write(f"{Colors.WHITE}[INFO]{Colors.RESET} Configuration information:\n")
        f.write("-"*50 + "\n")
        f.write(f"Algorithm: {algorithm_name}\n")
        f.write(f"Features file: {config['features_file']}\n")
        f.write(f"Model: {config['model_config']['model_name']}\n")
        f.write(f"Number of labels: {len(config['target_labels'])}\n")
        f.write(f"Number of experiments per label: {config['experiment_count']}\n")
        f.write(f"Total experiments: {len(config['target_labels']) * config['experiment_count']}\n")
        f.write(f"Available features: 262 (excluding 4 basic features) \n")
        f.write(f"Start time: {session_timestamp}\n\n")
        
        # Label processing statistics
        f.write(f"{Colors.WHITE}[INFO]{Colors.RESET} Label processing statistics:\n")
        f.write("-"*50 + "\n")
        
        total_successful_labels = 0
        total_successful_experiments = 0
        total_experiments = 0
        
        for label, results in all_label_results.items():
            successful_experiments = results['successful_experiments']
            total_experiments_for_label = results['total_experiments']
            success_rate = successful_experiments / total_experiments_for_label * 100 if total_experiments_for_label > 0 else 0
            
            status_text = f"{Colors.GREEN}[SUCCESS]{Colors.RESET}" if successful_experiments > 0 else f"{Colors.RED}[ERROR]{Colors.RESET}"
            f.write(f"{status_text} {label}: {successful_experiments}/{total_experiments_for_label} successful ({success_rate:.1f}%)\n")
            
            if successful_experiments > 0:
                total_successful_labels += 1
            total_successful_experiments += successful_experiments
            total_experiments += total_experiments_for_label
        
        f.write(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Overall statistics:\n")
        f.write("-"*50 + "\n")
        f.write(f"Successful labels: {total_successful_labels}/{len(config['target_labels'])} ({total_successful_labels/len(config['target_labels'])*100:.1f}%)\n")
        f.write(f"Successful experiments: {total_successful_experiments}/{total_experiments} ({total_successful_experiments/total_experiments*100:.1f}%)\n")
        
        # Strict mode quality control report
        strict_mode = config.get("strict_mode", False)
        min_success_rate = config.get("min_success_rate", 1.0)
        
        f.write(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Quality control settings:\n")
        f.write("-"*50 + "\n")
        f.write(f"Strict mode: {'Enabled' if strict_mode else 'Disabled'}\n")
        f.write(f"Minimum success rate: {min_success_rate*100:.0f}%\n")
        
        if strict_mode:
            # Check which labels meet strict mode requirements
            compliant_labels = []
            non_compliant_labels = []
            
            for label, results in all_label_results.items():
                success_rate = results['successful_experiments'] / results['total_experiments'] if results['total_experiments'] > 0 else 0
                if success_rate >= min_success_rate:
                    compliant_labels.append(label)
                else:
                    non_compliant_labels.append(label)
            
            f.write(f"Labels meeting quality requirements: {len(compliant_labels)}/{len(config['target_labels'])} ({len(compliant_labels)/len(config['target_labels'])*100:.1f}%)\n")
            
            if compliant_labels:
                f.write(f"  {Colors.GREEN}[SUCCESS]{Colors.RESET} Compliant labels: {', '.join(compliant_labels)}\n")
            
            if non_compliant_labels:
                f.write(f"  {Colors.RED}[ERROR]{Colors.RESET} Non-compliant labels: {', '.join(non_compliant_labels)}\n")
                f.write(f"     Consensus and final summary for these labels will be skipped\n")
            
            overall_quality_status = "Passed" if len(non_compliant_labels) == 0 else "Failed"
            f.write(f"Overall quality status: {overall_quality_status}\n")
        
        # Detailed result statistics
        f.write(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Detailed result statistics:\n")
        f.write("-"*50 + "\n")
        
        for label, results in all_label_results.items():
            f.write(f"\n[{label}]:\n")
            f.write(f"   Successful experiments: {results['successful_experiments']}/{results['total_experiments']}\n")
            f.write(f"   Session folder: {results['session_folder']}\n")
            f.write(f"   Summary report: {results.get('summary_file', 'Not generated')}\n")
            f.write(f"  Consensus: {results.get('consensus_file', 'Not generated')}\n")
            
            if results['successful_experiments'] > 0:
                f.write(f"   Status: {Colors.GREEN}[SUCCESS]{Colors.RESET} Success\n")
            else:
                f.write(f"   Status: {Colors.RED}[ERROR]{Colors.RESET} All experiments failed\n")
        
        # Run statistics
        f.write(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Run statistics:\n")
        f.write("-"*50 + "\n")
        f.write(f"Number of labels processed: {len(all_label_results)}\n")
        f.write(f"Number of session folders generated: {len(all_label_results)}\n")
        f.write(f"Number of experiment files generated: {total_successful_experiments}\n")
        
        # Get outcome descriptions
        outcome_descriptions = get_outcome_descriptions()
        f.write(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Details of processed labels:\n")
        f.write("-"*50 + "\n")
        for i, label in enumerate(config['target_labels'], 1):
            description = outcome_descriptions.get(label, label)
            f.write(f"{i:2d}. {label}: {description}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Multi-label optimization summary completed\n")
        f.write("="*80 + "\n")
    
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Multi-label summary report saved: {os.path.basename(summary_file)}")
    return summary_file

def generate_consensus_summary_txt(main_output_dir, config, session_timestamp, all_label_results):
    """
    Generate consensus feature summary txt file, support strict mode quality control
    
    Args:
        main_output_dir (str): main output directory (e.g. optimization_results/CORL_o3_mini/)
        config (dict): optimization configuration
        session_timestamp (str): session timestamp
        all_label_results (dict): all label results statistics
        
    Returns:
        str: generated txt file path, return None if not met
    """
    algorithm_name = config['features_file'].split('_')[0]
    clean_model_name = config['model_config']['model_name'].replace(" ", "_").replace("-", "_")
    
    print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Generating consensus feature summary txt file...")
    
    # Strict mode quality control check
    strict_mode = config.get("strict_mode", False)
    min_success_rate = config.get("min_success_rate", 1.0)
    
    if strict_mode:
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Strict mode check: verifying if all labels meet quality requirements...")
        
        # Check if each label has a complete consensus
        incomplete_labels = []
        total_labels = len(config['target_labels'])
        experiment_count = config['experiment_count']
        
        for target_label in config['target_labels']:
            if target_label in all_label_results:
                successful_experiments = all_label_results[target_label]['successful_experiments']
                total_experiments = all_label_results[target_label]['total_experiments']
                success_rate = successful_experiments / total_experiments if total_experiments > 0 else 0
                
                if success_rate < min_success_rate:
                    incomplete_labels.append({
                        'label': target_label,
                        'success_rate': success_rate,
                        'successful': successful_experiments,
                        'total': total_experiments
                    })
            else:
                # Label not processed
                incomplete_labels.append({
                    'label': target_label,
                    'success_rate': 0.0,
                    'successful': 0,
                    'total': experiment_count
                })
        
        if incomplete_labels:
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Strict mode check failed: {len(incomplete_labels)}/{total_labels} labels did not meet requirements")
            print(f"    Requirements: each label success rate ≥ {min_success_rate*100:.0f}%")
            
            for label_info in incomplete_labels:
                print(f"   {Colors.RED}[ERROR]{Colors.RESET} {label_info['label']}: {label_info['successful']}/{label_info['total']} "
                      f"({label_info['success_rate']*100:.1f}%)")
            
            # Generate detailed failure report
            failure_file = os.path.join(main_output_dir, f"consensus_summary_generation_failed_{session_timestamp}.txt")
            with open(failure_file, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("Consensus feature summary generation failed report\n")
                f.write("="*80 + "\n\n")
                f.write(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Strict mode quality control check failed\n\n")
                f.write(f"Configuration requirements:\n")
                f.write(f"   Strict mode: {strict_mode}\n")
                f.write(f"   Minimum success rate: {min_success_rate*100:.0f}%\n")
                f.write(f"   Number of labels processed: {total_labels}\n")
                f.write(f"   Number of experiments per label: {experiment_count}\n\n")
                f.write(f"Details of non-compliant labels:\n")
                f.write("-"*50 + "\n")
                
                for label_info in incomplete_labels:
                    f.write(f"{Colors.RED}[ERROR]{Colors.RESET} {label_info['label']}:\n")
                    f.write(f"    Successful experiments: {label_info['successful']}/{label_info['total']}\n")
                    f.write(f"    Success rate: {label_info['success_rate']*100:.1f}%\n")
                    f.write(f"    Required successful experiments: {int(label_info['total'] * min_success_rate)}\n")
                    f.write(f"    Difference: {int(label_info['total'] * min_success_rate) - label_info['successful']}\n\n")
                
                f.write(f"Resolution suggestions:\n")
                f.write(f"  1. Re-run failed experiments until all labels meet the required success rate\n")
                f.write(f"  2. Check the error logs and reasons for failed experiments\n")
                f.write(f"  3. Adjust model parameters, prompts, or retry mechanisms\n")
                f.write(f"  4. Set strict_mode=False to allow partial success generation of summaries\n")
                f.write(f"  5. Or lower the min_success_rate threshold\n")
                f.write("\n" + "="*80 + "\n")
            
            print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Skipping consensus summary file generation")
            print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Detailed failure report saved: {os.path.basename(failure_file)}")
            return None
        else:
            print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Strict mode check passed: all {total_labels} labels meet the {min_success_rate*100:.0f}% success rate requirement")
    
    # Collect consensus features for all labels
    all_consensus_features = {}
    successful_labels = []
    failed_labels = []
    
    for target_label in config['target_labels']:
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Processing label: {target_label}")
        
        try:
            # Find the folder for this label (matching {label}_timestamp format)
            label_folders = [d for d in os.listdir(main_output_dir) 
                           if os.path.isdir(os.path.join(main_output_dir, d)) and d.startswith(target_label + "_")]
            
            if not label_folders:
                print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} No session folder found for {target_label}")
                failed_labels.append(target_label)
                continue
            
            # Use the latest folder (sorted by timestamp, take the last one)
            label_folder = sorted(label_folders)[-1]
            label_folder_path = os.path.join(main_output_dir, label_folder)
            
            # Find consensus JSON files in this folder
            consensus_pattern = os.path.join(label_folder_path, f"*{target_label}_consensus_*.json")
            consensus_files = glob.glob(consensus_pattern)
            
            if not consensus_files:
                print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} No consensus file found in {label_folder}")
                failed_labels.append(target_label)
                continue
            
            # Use the found consensus file (if there are multiple, take the latest)
            consensus_file = sorted(consensus_files)[-1]
            
            # Read consensus JSON file
            with open(consensus_file, 'r', encoding='utf-8') as f:
                consensus_data = json.load(f)
            
            # Extract consensus feature set
            if target_label in consensus_data:
                consensus_features = consensus_data[target_label].get("consensus_feature_set", [])
                
                if consensus_features:
                    all_consensus_features[target_label] = consensus_features
                    successful_labels.append(target_label)
                    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} {target_label}: {len(consensus_features)} consensus features")
                else:
                    # Empty consensus feature set
                    all_consensus_features[target_label] = []
                    successful_labels.append(target_label)
                    print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} {target_label}: consensus feature set is empty")
            else:
                print(f"{Colors.RED}[ERROR]{Colors.RESET} {target_label} data not found in consensus file")
                failed_labels.append(target_label)
                
        except Exception as e:
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Error processing {target_label}: {str(e)}")
            failed_labels.append(target_label)
    
    # Generate summary txt file
    if successful_labels:
        consensus_txt_file = f"{algorithm_name}_{clean_model_name}_consensus.txt"
        consensus_txt_path = os.path.join(main_output_dir, consensus_txt_file)
        
        with open(consensus_txt_path, 'w', encoding='utf-8') as f:
            # Output labels in the order specified in config
            for target_label in config['target_labels']:
                if target_label in all_consensus_features:
                    consensus_features = all_consensus_features[target_label]
                    
                    # Convert feature descriptions back to original feature names (if needed, keep the description format here)
                    features_str = str(consensus_features).replace("'", "'")  # Uniform quote format
                    
                    f.write(f"{target_label} = {features_str} \n\n")
                else:
                    # For failed labels, output an empty list
                    f.write(f"{target_label} = [] \n\n")
        
        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Consensus summary file generated: {consensus_txt_file}")
        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Successfully processed labels: {len(successful_labels)}/{len(config['target_labels'])}")
        
        if successful_labels:
            print(f"    Success labels: {', '.join(successful_labels)}")
        if failed_labels:
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Failed labels: {', '.join(failed_labels)}")
        
        # Generate statistics
        total_consensus_features = sum(len(features) for features in all_consensus_features.values())
        avg_features_per_label = total_consensus_features / len(successful_labels) if successful_labels else 0
        
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Total consensus features: {total_consensus_features}")
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Average features per label: {avg_features_per_label:.1f}")
        
        return consensus_txt_path
    
    else:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} No successful labels, cannot generate consensus summary file")
        return None

def print_optimization_summary(result, target_label):
    """
    Print optimization result summary
    
    Args:
        result (dict): optimization result
        target_label (str): target label
    """
    print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} {target_label} optimization result summary:")
    print(f"{'='*50}")
    
    if "error" in result:
        error_type = result.get("metadata", {}).get("status", "unknown_error")
        
        if error_type == "validation_failed":
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Status: validation failed")
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Error type: {result['error']}")
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Validation error: {result.get('validation_error', 'Unknown validation error')}")
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Reason: API response incomplete or format abnormal")
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Original response length: {result.get('metadata', {}).get('response_lengths', {}).get('original', 'N/A')}, "
                  f"Cleaned response length: {result.get('metadata', {}).get('response_lengths', {}).get('cleaned', 'N/A')}")
        elif error_type == "smart_json_parse_failed":
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Status: JSON parsing failed")
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Error: {result['error']}")
            print(f"{Colors.RED}[ERROR]{Colors.RESET} JSON error: {result.get('json_error', 'Unknown JSON error')}")
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Number of cleaning steps: {len(result.get('cleaning_steps', []))}")
        else:
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Status: failed")
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Error: {result['error']}")
        return
    
    if target_label in result:
        label_result = result[target_label]
        
        optimized_count = len(label_result.get("optimized_feature_set", []))
        added_count = len(label_result.get("added_features", []))
        removed_count = len(label_result.get("removed_features", []))
        
        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Status: success")
        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Optimized feature count: {optimized_count}")
        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Added feature count: {added_count}")
        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Removed feature count: {removed_count}")
        
        if added_count > 0:
            print(f"\n{Colors.GREEN}[SUCCESS]{Colors.RESET} Added features:")
            for feature in label_result["added_features"]:
                print(f"   - {feature['feature']}")
        
        if removed_count > 0:
            print(f"\n{Colors.RED}[ERROR]{Colors.RESET} Removed features:")
            for feature in label_result["removed_features"]:
                print(f"   - {feature['feature']}")
    else:
        print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Status: response format abnormal")
        print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Result not found for label '{target_label}'")

def validate_config():
    """
    Validate the validity of the configuration
    """
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Validating configuration...")
    
    # Check if the feature file exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    features_file_path = os.path.join(script_dir, OPTIMIZATION_CONFIG["features_file"])
    
    if not os.path.exists(features_file_path):
        raise FileNotFoundError(f"{Colors.RED}[ERROR]{Colors.RESET} Feature file not found: {features_file_path}")
    
    # Check if the required files exist
    required_files = ["selected_features.txt", "features-desc.csv"]
    for file_name in required_files:
        file_path = os.path.join(script_dir, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{Colors.RED}[ERROR]{Colors.RESET} Required file not found: {file_path}")
    
    # Check if the labels exist in the feature file
    with open(features_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    namespace = {}
    exec(content, namespace)
    
    target_labels = OPTIMIZATION_CONFIG["target_labels"]
    if not isinstance(target_labels, list) or not target_labels:
        raise ValueError(f"{Colors.RED}[ERROR]{Colors.RESET} target_labels must be a non-empty list")
    
    missing_labels = []
    for target_label in target_labels:
        if target_label not in namespace:
            missing_labels.append(target_label)
    
    if missing_labels:
        raise ValueError(f"{Colors.RED}[ERROR]{Colors.RESET} The following labels do not exist in {OPTIMIZATION_CONFIG['features_file']}: {missing_labels}")
    
    # Validate the experiment count configuration
    experiment_count = OPTIMIZATION_CONFIG.get("experiment_count", 1)
    if not isinstance(experiment_count, int) or experiment_count < 1:
        raise ValueError(f"{Colors.RED}[ERROR]{Colors.RESET} Experiment count must be a positive integer, current value: {experiment_count}")
    
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Configuration validation passed")

def main():
    """
    Main function - supports multi-label batch processing
    """
    print("Multi-label feature optimization script")
    print("="*70)
    
    try:
        # 1. Validate configuration
        validate_config()
        
        # 2. Display configuration information
        target_labels = OPTIMIZATION_CONFIG['target_labels']
        experiment_count = OPTIMIZATION_CONFIG['experiment_count']
        algorithm_name = OPTIMIZATION_CONFIG['features_file'].split('_')[0]
        clean_model_name = OPTIMIZATION_CONFIG['model_config']['model_name'].replace(" ", "_").replace("-", "_")
        
        print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Configuration information:")
        print(f"Feature file: {OPTIMIZATION_CONFIG['features_file']}")
        print(f"Model: {OPTIMIZATION_CONFIG['model_config']['model_name']}")
        print(f"Target label count: {len(target_labels)}")
        print(f"Target labels: {', '.join(target_labels)}")
        print(f"Experiments per label: {experiment_count}")
        print(f"Total experiments: {len(target_labels) * experiment_count}")
        
        # Dynamically get the number of available features
        try:
            all_feature_descriptions, _ = get_all_available_feature_descriptions()
            available_feature_count = len(all_feature_descriptions)
            print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Available feature count: {available_feature_count}")
        except Exception as e:
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Failed to get available feature count: {str(e)}")
            
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Output directory: {OPTIMIZATION_CONFIG['output_dir']}")
        
        # Quality control configuration information
        strict_mode = OPTIMIZATION_CONFIG.get("strict_mode", False)
        min_success_rate = OPTIMIZATION_CONFIG.get("min_success_rate", 1.0)
        allow_partial_consensus = OPTIMIZATION_CONFIG.get("allow_partial_consensus", True)
        
        print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Quality control configuration:")
        print(f"- Strict mode: {'Enabled' if strict_mode else 'Disabled'}")
        print(f"- Minimum success rate: {min_success_rate*100:.0f}%")
        print(f"- Allow partial consensus: {'Yes' if allow_partial_consensus else 'No'}")
        
        if strict_mode:
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} Strict mode description:")
            print(f"- Only labels with success rate ≥ {min_success_rate*100:.0f}% will generate consensus")
            print(f"- Only when all labels meet the requirements will the final summary be generated")
            print(f"- Ensure the high quality and reliability of experimental results")
        
        # 3. Confirm execution
        total_experiments = len(target_labels) * experiment_count
        
        confirm = input(f"\nContinue with multi-label optimization? (y/N): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print(f"{Colors.RED}[ERROR]{Colors.RESET} User cancelled operation")
            return
        
        # 4. Create main output folder
        session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Main folder structure: optimization_results/CORL_qwen3_8b_a1b2c3d4/
        # Add uuid to ensure folder name uniqueness, avoid duplicates
        unique_id = str(uuid.uuid4())[:8]
        main_folder_name = f"{algorithm_name}_{clean_model_name}_{unique_id}"
        main_output_dir = os.path.join(OPTIMIZATION_CONFIG['output_dir'], main_folder_name)
        os.makedirs(main_output_dir, exist_ok=True)
        
        print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Created main output folder: {main_folder_name}")
        
        # 5. Initialize multi-label processing statistics
        all_label_results = {}
        total_successful_labels = 0
        total_successful_experiments = 0
        
        # 6. Loop through each label
        for label_index, target_label in enumerate(target_labels, 1):
            print(f"\n{'='*70}")
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} Label progress: {label_index}/{len(target_labels)} ({target_label})")
            print(f"{'='*70}")
            
            # Create a dedicated timestamp and session folder for each label
            label_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_folder_name = f"{target_label}_{label_timestamp}"
            session_dir = os.path.join(main_output_dir, session_folder_name)
            os.makedirs(session_dir, exist_ok=True)
            
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} Created label session folder: {session_folder_name}")
            
            # Preload the original feature information for this label
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} Loading original feature information for {target_label}...")
            try:
                original_feature_descriptions, _ = get_target_feature_descriptions(
                    OPTIMIZATION_CONFIG['features_file'], 
                    target_label
                )
            except Exception as e:
                print(f"{Colors.RED}[ERROR]{Colors.RESET} Failed to load features for {target_label}: {str(e)}")
                all_label_results[target_label] = {
                    'successful_experiments': 0,
                    'total_experiments': experiment_count,
                    'session_folder': session_folder_name,
                    'error': str(e)
                }
                continue
            
            # Execute multiple experiments for this label
            successful_optimizations = 0
            all_results = []  # Collect all experiments for this label
            
            for experiment_num in range(1, experiment_count + 1):
                print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Experiment progress: {experiment_num}/{experiment_count}")
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} Current label: {target_label}")
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} Overall progress: Label {label_index}/{len(target_labels)}, Experiment {experiment_num}/{experiment_count}")
                
                # Execute optimization (using enhanced version: logging + retry mechanism + automatic feature calculation)
                result = optimize_features_for_label(
                    model_config=OPTIMIZATION_CONFIG["model_config"],
                    features_file=OPTIMIZATION_CONFIG["features_file"],
                    target_label=target_label,
                    session_dir=session_dir,
                    experiment_num=experiment_num
                )
                
                # Add experiment information to the metadata
                if "metadata" in result:
                    result["metadata"]["experiment_number"] = experiment_num
                    result["metadata"]["total_experiments"] = experiment_count
                    result["metadata"]["session_folder"] = session_folder_name
                    result["metadata"]["label_index"] = label_index
                    result["metadata"]["total_labels"] = len(target_labels)
                
                # Save results to the session folder for this label
                save_path = save_optimization_result(
                    result=result,
                    output_dir=session_dir,
                    features_file=OPTIMIZATION_CONFIG["features_file"],
                    model_name=OPTIMIZATION_CONFIG["model_config"]["model_name"],
                    target_label=target_label,
                    experiment_num=experiment_num,
                    session_timestamp=label_timestamp
                )
                
                # Collect results for summary
                all_results.append(result)
                
                # Print summary
                print_optimization_summary(result, target_label)
                
                if "error" not in result:
                    successful_optimizations += 1
                    total_successful_experiments += 1
                
                # If not the last experiment, add a gap
                if experiment_num < experiment_count:
                    print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Preparing to start the next experiment...")
            
            # 7. Generate session summary report for this label
            print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Generating summary report for {target_label}...")
            summary_file = generate_session_summary(
                all_results=all_results,
                config=OPTIMIZATION_CONFIG,
                session_dir=session_dir,
                original_features=original_feature_descriptions,
                session_timestamp=label_timestamp,
                target_label=target_label
            )
            
            # 8. Generate consensus feature set for this label
            print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Generating consensus feature set for {target_label}...")
            consensus_file = generate_consensus_features(
                all_results=all_results,
                config=OPTIMIZATION_CONFIG,
                session_dir=session_dir,
                session_timestamp=label_timestamp,
                target_label=target_label
            )
            
            # Record the processing result for this label
            if successful_optimizations > 0:
                total_successful_labels += 1
            
            all_label_results[target_label] = {
                'successful_experiments': successful_optimizations,
                'total_experiments': experiment_count,
                'session_folder': session_folder_name,
                'summary_file': os.path.basename(summary_file) if summary_file else None,
                'consensus_file': os.path.basename(consensus_file) if consensus_file else None
            }
            
            print(f"\n{Colors.GREEN}[SUCCESS]{Colors.RESET} {target_label} processing completed:")
            print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Successful experiments: {successful_optimizations}/{experiment_count}")
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} Session folder: {session_folder_name}")
            
            # If not the last label, add a gap
            if label_index < len(target_labels):
                print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Preparing to process the next label...")
        
        # 9. Generate multi-label overall summary report
        print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Generating multi-label overall summary report...")
        multi_summary_file = generate_multi_label_summary(
            all_label_results=all_label_results,
            config=OPTIMIZATION_CONFIG,
            main_output_dir=main_output_dir,
            session_timestamp=session_timestamp
        )
        
        # 10. Generate consensus feature summary txt file (strict mode quality control)
        consensus_txt_file = generate_consensus_summary_txt(
            main_output_dir=main_output_dir,
            config=OPTIMIZATION_CONFIG,
            session_timestamp=session_timestamp,
            all_label_results=all_label_results
        )
        
        # 11. Final summary (including strict mode quality control information)
        total_experiments = len(target_labels) * experiment_count
        strict_mode = OPTIMIZATION_CONFIG.get("strict_mode", False)
        min_success_rate = OPTIMIZATION_CONFIG.get("min_success_rate", 1.0)
        
        print(f"\n{Colors.GREEN}[SUCCESS]{Colors.RESET} Multi-label optimization completed!")
        print(f"{'='*70}")
        print(f"Processed label count: {len(target_labels)}")
        print(f"Successful labels: {total_successful_labels}/{len(target_labels)} ({total_successful_labels/len(target_labels)*100:.1f}%)")
        print(f"Total experiments: {total_experiments}")
        print(f"Successful experiments: {total_successful_experiments}/{total_experiments} ({total_successful_experiments/total_experiments*100:.1f}%)")
        
        # Strict mode quality control report
        print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Quality control status:")
        print(f"Strict mode: {'Enabled' if strict_mode else 'Disabled'}")
        print(f"Minimum success rate: {min_success_rate*100:.0f}%")
        
        if strict_mode:
            # Count quality control results
            compliant_labels = []
            non_compliant_labels = []
            
            for label, results in all_label_results.items():
                success_rate = results['successful_experiments'] / results['total_experiments'] if results['total_experiments'] > 0 else 0
                if success_rate >= min_success_rate:
                    compliant_labels.append(label)
                else:
                    non_compliant_labels.append(label)
            
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} Labels meeting quality requirements: {len(compliant_labels)}/{len(target_labels)} ({len(compliant_labels)/len(target_labels)*100:.1f}%)")
            
            if compliant_labels:
                print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Compliant labels: {', '.join(compliant_labels)}")
            
            if non_compliant_labels:
                print(f"{Colors.RED}[ERROR]{Colors.RESET} Non-compliant labels: {', '.join(non_compliant_labels)}")
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} Note: The consensus and final summary of non-compliant labels have been skipped")
            
            overall_quality_status = "Passed" if len(non_compliant_labels) == 0 else "Failed"
            status_icon = f"{Colors.GREEN}[SUCCESS]{Colors.RESET}" if len(non_compliant_labels) == 0 else f"{Colors.RED}[ERROR]{Colors.RESET}"
            print(f"{status_icon} Overall quality status: {overall_quality_status}")
        
        print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Output files:")
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Main output folder: {main_folder_name}")
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Full path: {main_output_dir}")
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Overall summary report: {os.path.basename(multi_summary_file)}")
        
        if consensus_txt_file:
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} Consensus feature summary: {os.path.basename(consensus_txt_file)}")
            if strict_mode:
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} Note: This summary file only contains labels that meet quality requirements")
        else:
            if strict_mode:
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} Consensus feature summary: Not generated (Strict mode quality control failed)")
            else:
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} Consensus feature summary: Not generated (Processing failed)")
        
        print(f"\n{Colors.GREEN}[SUCCESS]{Colors.RESET} Generated label session folders:")
        for label, results in all_label_results.items():
            status = f"{Colors.GREEN}[SUCCESS]{Colors.RESET}" if results['successful_experiments'] > 0 else f"{Colors.RED}[ERROR]{Colors.RESET}"
            print(f"   {status} {label}: {results['session_folder']} ({results['successful_experiments']}/{results['total_experiments']} Successful)")
        
        print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} View results:")
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Main folder: {main_output_dir}")
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Overall report: {multi_summary_file}")
        if consensus_txt_file:
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} Consensus feature summary: {consensus_txt_file}")
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Detailed results for each label: session_summary and consensus files in each label folder")
        
    except Exception as e:
        print(f"\n{Colors.RED}[ERROR]{Colors.RESET} Script execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 