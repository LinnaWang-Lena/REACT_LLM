#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Causal feature generation script

Function: Call the large model API to generate causal feature lists for each outcome based on clinical knowledge
- Input 262 complete feature lists (excluding 4 basic features)
- Output causal feature lists for 7 outcomes
- Generate detailed summary reports
"""

import os
import json
import sys
import uuid
import glob
import difflib
from datetime import datetime

# Import existing API call and feature loading functions
from prediction import (
    query_model, 
    load_feature_descriptions, 
    load_feature_lists
)

# ANSI color definition class
class Colors:
    """Academic terminal output color definition"""
    RED = '\033[91m'      # Error information
    GREEN = '\033[92m'    # Success information  
    YELLOW = '\033[93m'   # Warning information
    WHITE = '\033[97m'    # General information
    RESET = '\033[0m'     # Reset color

# All supported outcome labels list
ALL_OUTCOME_LABELS = [
    "DIEINHOSPITAL",
    "Readmission_30", 
    "Multiple_ICUs",
    "sepsis_all",
    "FirstICU24_AKI_ALL", 
    "LOS_Hospital",
    "ICU_within_12hr_of_admit"
]

# Causal feature generation configuration
CAUSAL_GENERATION_CONFIG = {
    # Run mode configuration
    "run_mode": "all_labels",  # "single_label" or "all_labels"
    
    # Model configuration (reuse the structure of optimize_features.py)
    "model_config": {
        "api_type": "openai",  # "openai" or "dashscope"
        "model_name": "o3-mini",  # Choose model here
        "display_name": "o3-mini",
        
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
            "temperature": 0.0,      # Slightly increase to ensure some diversity
            "max_tokens": 4000       # Increase token limit to accommodate all outcomes
        }
    },
    
    # Target outcome (used in single-label mode, keep backward compatibility)
    "target_outcome": "Readmission_30", # Optional values: 'DIEINHOSPITAL','Readmission_30','Multiple_ICUs','sepsis_all','FirstICU24_AKI_ALL','LOS_Hospital','ICU_within_12hr_of_admit'
    
    # Experiment configuration
    "experiment_count": 3,  # Repeat experiment times
    
    # Output configuration
    "output_dir": "causal_features_generation_o3-mini_all_labels",
    
    # Quality control configuration
    "strict_mode": True,              # Strict mode: require 100% success rate to generate final results
    "min_success_rate": 1.0,          # Minimum success rate threshold (1.0 = 100%)
    "allow_partial_consensus": False, # Whether to allow partial success generation of consensus
    
    # Category exemption configuration
    "allow_category_exemption": False,        # Whether to allow category exemption
    "require_exemption_justification": False, # Whether to require exemption justification
    "min_exemption_reason_length": 50,       # Minimum length of exemption reason
    "category_min_percentages": {            # Minimum percentage requirements for each category
        "Diag": 0.1,  # 10%
        "Proc": 0.1,  # 10%
        "Med": 0.1,   # 10% 
        "TS": 0.1     # 10%
    }
}

def load_all_available_features():
    """
    Get the complete list of 262 feature descriptions (excluding 4 basic features), organized by category
    Reuse the logic in optimize_features.py
    
    Returns:
        tuple: (categorized feature dictionary, total feature description list, categorized original feature name dictionary)
    """
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Loading complete feature list (excluding basic features)...")
    
    # 1. Load 262 original feature names (excluding 4 basic features) from selected_features.txt
    feature_lists = load_feature_lists()
    
    # Organize original feature names by category
    categorized_names = {
        'Diag': feature_lists['Diag'],    
        'Proc': feature_lists['Proc'],    
        'Med': feature_lists['Med'],      
        'TS': feature_lists['TS']          
    }
    
    # Count the number of features in each category
    category_counts = {category: len(features) for category, features in categorized_names.items()}
    total_count = sum(category_counts.values())
    
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Loaded categorized features:")
    for category, count in category_counts.items():
        print(f"{category}: {count} features")
    print(f"Total: {total_count} features")
    
    # 2. Load feature description mapping table
    feature_descriptions = load_feature_descriptions()
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Loaded {len(feature_descriptions)} feature description mappings")
    
    # 3. Map original feature names to descriptions (by category)
    categorized_features = {}
    all_feature_descriptions = []
    missing_descriptions = []
    
    for category, feature_names in categorized_names.items():
        category_descriptions = []
        
        for feature_name in feature_names:
            description = feature_descriptions.get(feature_name, None)
        if description:
            # Remove extra spaces and use the description
                clean_description = description.strip()
                category_descriptions.append(clean_description)
                all_feature_descriptions.append(clean_description)
        else:
            # If no description is found, use the original feature name and record it
                category_descriptions.append(feature_name)
                all_feature_descriptions.append(feature_name)
                missing_descriptions.append(feature_name)
        
        categorized_features[category] = category_descriptions
    
    if missing_descriptions:
        print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} {len(missing_descriptions)} features have no description mapping, using original names")
        print(f"Features with missing descriptions: {missing_descriptions[:10]}{'...' if len(missing_descriptions) > 10 else ''}")
    
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Completed feature mapping: {len(all_feature_descriptions)} feature descriptions")
    
    # Add debug information: check if Med features are correctly loaded
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Debug information - Med feature loading check:")
    med_features = categorized_features.get('Med', [])
    print(f"Med feature count: {len(med_features)}")
    if len(med_features) > 0:
        print(f"First 5 Med features: {med_features[:5]}")
    else:
        print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} No Med features loaded!")
        print(f"Original Med feature names: {categorized_names.get('Med', [])[:5]}")
    
    return categorized_features, all_feature_descriptions, categorized_names

def get_outcome_descriptions():
    """
    Provide clinical descriptions for each outcome to help AI understand
    
    Returns:
        dict: outcome name to clinical description mapping
    """
    return {
        "DIEINHOSPITAL": "In-hospital Death (mortality during hospitalization)",
        "Readmission_30": "30-day Hospital Readmission (readmission within 30 days after discharge)", 
        "Multiple_ICUs": "Multiple ICU Stays (patient transferred between different ICUs during hospitalization)",
        "sepsis_all": "Sepsis (all types of sepsis including severe sepsis and septic shock)",
        "FirstICU24_AKI_ALL": "Acute Kidney Injury within first 24 hours of ICU admission (all stages of AKI)",
        "LOS_Hospital": "Length of Hospital Stay (continuous outcome measuring total days in hospital)",
        "ICU_within_12hr_of_admit": "ICU Admission within 12 hours (early ICU admission after hospital admission)"
    }

def build_causal_generation_prompt(categorized_features, target_outcome, config):
    """
    Build the English prompt for causal feature generation (supports category display and exemption mechanism)
    
    Args:
        categorized_features (dict): categorized feature dictionary
        target_outcome (str): target outcome
        config (dict): configuration information
        
    Returns:
        str: built prompt
    """
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Building causal feature generation prompt...")
    
    # Get configuration parameters
    allow_exemption = config.get("allow_category_exemption", False)
    require_justification = config.get("require_exemption_justification", True)
    category_min_percentages = config.get("category_min_percentages", {})
    
    # Build categorized feature display
    features_display = []
    category_requirements = []
    
    category_sizes = {}  # Record the size of each category
    
    for category, features in categorized_features.items():
        category_size = len(features)
        category_sizes[category] = category_size
    
    # Format feature list
        features_list_str = "[" + ", ".join([f"'{desc}'" for desc in features]) + "]"
        features_display.append(f"{category} features ({category_size} features): {features_list_str}")
        
        # Add debug information: check Med features
        if category == "Med":
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} Debug information - Med features in Prompt:")
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} Med feature count: {category_size}")
            if category_size > 0:
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} First 3 Med features: {features[:3]}")
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} Med feature list length: {len(features_list_str)} characters")
            else:
                print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Critical warning: Med feature list is empty!")
        
        # Calculate minimum requirements
        min_percentage = category_min_percentages.get(category, 0.1)
        min_required = int(category_size * min_percentage)
        category_requirements.append(f"  * {category} ({'Diagnosis' if category == 'Diag' else 'Procedures' if category == 'Proc' else 'Medications' if category == 'Med' else 'Time Series'}): At least {min_required} features ({min_percentage*100:.0f}% of {category_size})")
    
    features_str = "\n".join(features_display)
    requirements_str = "\n".join(category_requirements)
    
    # Get outcome description
    outcome_descriptions = get_outcome_descriptions()
    outcome_description = outcome_descriptions.get(target_outcome, target_outcome)
    
    # Build JSON format example (supports exemption)
    if allow_exemption:
        json_example = f"""{{
  "{target_outcome}": {{
    "selected_features": {{
      "Diag": ["feature_description_1", "feature_description_2", ...],
      "Proc": ["feature_description_1", "feature_description_2", ...],
      "Med": ["feature_description_1", "feature_description_2", ...],
      "TS": ["feature_description_1", "feature_description_2", ...]
    }},
    "category_exemptions": {{
      "Proc": "Medical justification for why procedures are not causally relevant to this outcome...",
      "Med": null
    }}
  }}
}}"""
    else:
        json_example = f"""{{
  "{target_outcome}": {{
    "selected_features": {{
      "Diag": ["feature_description_1", "feature_description_2", ...],
      "Proc": ["feature_description_1", "feature_description_2", ...], 
      "Med": ["feature_description_1", "feature_description_2", ...],
      "TS": ["feature_description_1", "feature_description_2", ...]
    }}
  }}
}}"""
    
    # Build exemption rules explanation
    exemption_rules = ""
    if allow_exemption:
        exemption_rules = f"""
CATEGORY EXEMPTION RULES:
- You may completely skip a category (select 0 features) ONLY if you have strong clinical justification
- If you exempt a category, you MUST provide a detailed medical reason in the "category_exemptions" field
- Exemptions should be rare and only for categories that are genuinely irrelevant to the outcome
- Exemption reasons must be at least {config.get('min_exemption_reason_length', 50)} characters and explain the clinical rationale
- Use null for categories you are NOT exempting
"""
    
    prompt = f"""You are a clinical expert with extensive experience in intensive care medicine. Your task is to identify features that have DIRECT or INDIRECT causal relationships with the outcome based on established clinical knowledge and clinical experience.

1. All Available Features (organized by category):
{features_str}

2. You need to analyze the above features to identify those that have direct or indirect causal relationships with {outcome_description}.

Consider the following:
- DIRECT causal relationships: Features that directly cause or strongly predict the outcome
- INDIRECT causal relationships: Features that are part of the causal pathway, represent underlying pathophysiology, or are established risk factors
- Include features that reflect disease severity, organ dysfunction, or treatment interventions related to the outcome
- EXCLUDE features that are purely correlational without established causal basis
- EXCLUDE features that are consequences rather than causes of the outcome

3. Selection Requirements:
- Select features that have direct or indirect causal relationships with {outcome_description}
- MINIMUM feature count per category:
{requirements_str}
- You may select more than the minimum if clinically relevant

4. Output Requirements:
Return your analysis in the following JSON format with feature descriptions EXACTLY matching those provided in the feature list:
{json_example}

IMPORTANT: 
- Use EXACT feature descriptions as provided in the feature list
- Organize selections by category (Diag, Proc, Med, TS)
- Meet minimum requirements for each category unless providing valid exemption
- Base selections on established clinical evidence and pathophysiology
- Provide valid JSON format without additional explanations"""

    total_features = sum(len(features) for features in categorized_features.values())
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Prompt built, total length: {len(prompt)} characters")
    print(f"Total features: {total_features} features, 4 categories")
    for category, size in category_sizes.items():
        min_pct = category_min_percentages.get(category, 0.1)
        min_required = int(size * min_pct)
        print(f"{category}: {size} features, minimum required: {min_required} features")
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Exemption mechanism: {'enabled' if allow_exemption else 'disabled'}")
    #print(f"prompt: {prompt}")  # Uncomment to print the prompt
    return prompt

def validate_causal_results_with_exemptions(results, config, target_outcome, categorized_features):
    """
    Validate the validity of the generated results (supports category exemption mechanism)
    
    Args:
        results (dict): API returned results
        config (dict): configuration information
        target_outcome (str): target outcome
        categorized_features (dict): categorized valid feature dictionary
        
    Returns:
        tuple: (is_valid, validation_report)
    """
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Validating generated results (supports exemption mechanism)...")
    
    # Get configuration parameters
    allow_exemption = config.get("allow_category_exemption", False)
    require_justification = config.get("require_exemption_justification", True)
    min_exemption_length = config.get("min_exemption_reason_length", 50)
    category_min_percentages = config.get("category_min_percentages", {})
    
    validation_report = {
        "target_outcome": target_outcome,
        "found_outcome": False,
        "category_compliance": {},
        "exemptions_granted": {},
        "exemptions_rejected": [],
        "invalid_features": {},
        "overall_valid": True,
        "total_features": 0,
        "new_format": False  # Whether to use the new classification format
    }
    
    # Build all valid feature set
    all_valid_features = set()
    category_sizes = {}
    for category, features in categorized_features.items():
        all_valid_features.update(features)
        category_sizes[category] = len(features)
    
    # Check target outcome
    if target_outcome not in results:
        validation_report["found_outcome"] = False
        validation_report["overall_valid"] = False
        print(f"{Colors.RED}[ERROR]{Colors.RESET} Target outcome not found")
        return False, validation_report
    
    validation_report["found_outcome"] = True
    outcome_data = results[target_outcome]
    
    # Check result format (new format has selected_features field)
    if isinstance(outcome_data, dict) and "selected_features" in outcome_data:
        validation_report["new_format"] = True
        selected_features = outcome_data["selected_features"]
        exemptions = outcome_data.get("category_exemptions", {})
        
        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Detected new format (categorized features + exemption mechanism)")
        
        # Validate each category
        total_features = 0
        for category in ["Diag", "Proc", "Med", "TS"]:
            category_features = selected_features.get(category, [])
            category_size = category_sizes.get(category, 0)
            min_percentage = category_min_percentages.get(category, 0.1)
            min_required = int(category_size * min_percentage)
            actual_count = len(category_features)
            total_features += actual_count
            
            # Check invalid features
            invalid_in_category = []
            valid_category_features = categorized_features.get(category, [])
            for feature in category_features:
                if feature not in valid_category_features:
                    invalid_in_category.append(feature)
            
            validation_report["invalid_features"][category] = invalid_in_category
            
            # Check category requirements and exemptions
            if actual_count < min_required:
                # Special warning for Med category problem
                if category == "Med":
                    print(f"{Colors.RED}[ERROR]{Colors.RESET} Critical warning: {category} category does not meet minimum requirements!")
                    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Requirements: {min_required} features, actual: {actual_count} features")
                    if actual_count == 0:
                        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Med features completely missing! Please check prompt and model response")
                
                # Check if there is an exemption
                exemption_reason = exemptions.get(category)
                if allow_exemption and exemption_reason:
                    # Validate exemption reason
                    if (len(exemption_reason) >= min_exemption_length and 
                        any(keyword in exemption_reason.lower() for keyword in 
                            ["causal", "relevant", "determine", "predict", "pathophysiology", "clinical", "medical"])):
                        validation_report["exemptions_granted"][category] = exemption_reason
                        validation_report["category_compliance"][category] = {
                            "required": min_required,
                            "actual": actual_count,
                            "compliant": True,  # Exemption is considered compliant
                            "exempted": True,
                            "exemption_reason": exemption_reason
                        }
                        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} {category} category exemption approved: {exemption_reason[:100]}...")
                    else:
                        validation_report["exemptions_rejected"].append({
                            "category": category,
                            "reason": "Exemption reason insufficient",
                            "provided_reason": exemption_reason,
                            "min_length_required": min_exemption_length
                        })
                        validation_report["category_compliance"][category] = {
                            "required": min_required,
                            "actual": actual_count,
                            "compliant": False,
                            "exempted": False
                        }
                        validation_report["overall_valid"] = False
                else:
                    # No exemption and does not meet requirements
                    validation_report["category_compliance"][category] = {
                        "required": min_required,
                        "actual": actual_count,
                        "compliant": False,
                        "exempted": False
                    }
                    validation_report["overall_valid"] = False
            else:
                # Meets requirements
                validation_report["category_compliance"][category] = {
                    "required": min_required,
                    "actual": actual_count,
                    "compliant": True,
                    "exempted": False
                }
            
            # Check invalid features
            if invalid_in_category:
                print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} {category} category contains {len(invalid_in_category)} invalid features")
                validation_report["overall_valid"] = False
        
        validation_report["total_features"] = total_features
        
    else:
        # Old format (simple list)
        validation_report["new_format"] = False
        if isinstance(outcome_data, list):
            features = outcome_data
        else:
            features = []
            validation_report["overall_valid"] = False
        
        print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Detected old format (simple list), does not support exemption mechanism")
        
        validation_report["total_features"] = len(features)
        
        # Check invalid features (old format)
        invalid_features = []
        for feature in features:
            if feature not in all_valid_features:
                invalid_features.append(feature)
        
        validation_report["invalid_features"]["all"] = invalid_features
        
        if invalid_features or not features:
            validation_report["overall_valid"] = False
    
    # Output validation results
    if validation_report["overall_valid"]:
        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Validation passed")
        print(f"Found target outcome: {target_outcome}")
        print(f"Total features: {validation_report['total_features']} features")
        if validation_report["new_format"]:
            for category, compliance in validation_report["category_compliance"].items():
                status = f"{Colors.GREEN}[SUCCESS]{Colors.RESET}" if compliance["compliant"] else f"{Colors.RED}[ERROR]{Colors.RESET}"
                exempted_info = " (exemption)" if compliance.get("exempted", False) else ""
                print(f"{category}: {compliance['actual']}/{compliance['required']} {status}{exempted_info}")
    else:
        print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Validation found problems:")
        if not validation_report["found_outcome"]:
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Target outcome not found: {target_outcome}")
        if validation_report["new_format"]:
            for category, compliance in validation_report["category_compliance"].items():
                if not compliance["compliant"]:
                    print(f"{category}: {compliance['actual']}/{compliance['required']} {Colors.RED}[ERROR]{Colors.RESET} Non-compliant")
        if validation_report["exemptions_rejected"]:
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Exemption rejected: {len(validation_report['exemptions_rejected'])} categories")
        
        total_invalid = sum(len(invalid_list) for invalid_list in validation_report["invalid_features"].values())
        if total_invalid > 0:
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Total invalid features: {total_invalid} features")
    
    return validation_report["overall_valid"], validation_report

def validate_causal_results(results, all_features, target_outcome):
    """
    Validate the validity of the generated results (backward compatibility)
    
    Args:
        results (dict): API returned results
        all_features (list): all available features list
        target_outcome (str): target outcome
        
    Returns:
        tuple: (is_valid, validation_report)
    """
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Validating generated results (backward compatibility)...")
    
    validation_report = {
        "target_outcome": target_outcome,
        "found_outcome": False,
        "invalid_features": [],
        "empty_outcome": False,
        "feature_count": 0
    }
    
    all_features_set = set(all_features)
    
    # Check target outcome
    if target_outcome in results:
        validation_report["found_outcome"] = True
        outcome_data = results[target_outcome]
        
        # Process new format
        if isinstance(outcome_data, dict) and "selected_features" in outcome_data:
            # Extract all selected features
            all_selected = []
            for category_features in outcome_data["selected_features"].values():
                all_selected.extend(category_features)
            outcome_features = all_selected
        else:
            # Old format
            outcome_features = outcome_data if isinstance(outcome_data, list) else []
            
            # Check if it is empty
            if not outcome_features:
                validation_report["empty_outcome"] = True   
            else:
                validation_report["feature_count"] = len(outcome_features)
                
                # Check if the feature is valid
                for feature in outcome_features:
                    if feature not in all_features_set:
                        validation_report["invalid_features"].append(feature)
    
    # Check if it is valid
    is_valid = (
        validation_report["found_outcome"] and
        not validation_report["empty_outcome"] and
        len(validation_report["invalid_features"]) == 0
    )
    
    # Output validation results
    if is_valid:
        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Validation passed")
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Found target outcome: {target_outcome}")
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Feature count: {validation_report['feature_count']} features")
    else:
        print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Validation found problems:")
        if not validation_report["found_outcome"]:
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Target outcome not found: {target_outcome}")
        if validation_report["empty_outcome"]:
            print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Feature list is empty")
        if validation_report["invalid_features"]:
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Invalid features ({len(validation_report['invalid_features'])} features): {validation_report['invalid_features'][:5]}...")
    
    return is_valid, validation_report

def clean_json_response(response):
    """
    Clean JSON response that may contain markdown markers
    
    Args:
        response (str): original API response
        
    Returns:
        str: cleaned pure JSON string
    """
    response = response.strip()
    
    # Check if there is a markdown code block marker
    if response.startswith('```json'):
        # Extract the content between ```json and ```
        start_marker = '```json'
        end_marker = '```'
        
        start_idx = response.find(start_marker) + len(start_marker)
        end_idx = response.rfind(end_marker)
        
        if end_idx > start_idx:
            response = response[start_idx:end_idx].strip()
            print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Cleaned markdown format JSON response")
        else:
            print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Detected ```json marker but cannot find end marker, using original response")
    
    elif response.startswith('```'):
        # Process code blocks without language identifiers
        lines = response.split('\n')
        if len(lines) > 2:
            response = '\n'.join(lines[1:-1]).strip()
            print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Cleaned generic markdown code block format")
        else:
            print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Detected ``` marker but format not as expected, using original response")
    
    return response

def generate_causal_features_single_experiment(model_config, categorized_features, target_outcome, config, experiment_num=None):
    """
    Call API to generate causal features (single experiment) - supports categorized features and exemption mechanism
    
    Args:
        model_config (dict): model configuration
        categorized_features (dict): categorized feature dictionary
        target_outcome (str): target outcome
        config (dict): complete configuration information
        experiment_num (int, optional): experiment number
        
    Returns:
        dict: generated result
    """
    exp_info = f" (experiment {experiment_num})" if experiment_num else ""
    print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Start generating causal features{exp_info}")
    
    try:
        # 1. Build prompt (using new classification format)
        prompt = build_causal_generation_prompt(categorized_features, target_outcome, config)
        
        # 2. Call API
        total_features = sum(len(features) for features in categorized_features.values())
        print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Calling {model_config['model_name']} API...")
        print(f"Prompt length: {len(prompt)} characters")
        print(f"Input features: {total_features} features (4 categories)")
        print(f"Target outcome: {target_outcome}")
        print(f"Exemption mechanism: {'enabled' if config.get('allow_category_exemption', False) else 'disabled'}")
        
        response = query_model(prompt, model_config)
        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} API call successful, response length: {len(response)} characters")
        
        # 3. Parse JSON response
        cleaned_response = response  # Initialize as original response, in case cleaning fails
        try:
            # Clean possible markdown format
            cleaned_response = clean_json_response(response)
            result = json.loads(cleaned_response)
            print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} JSON parsed successfully")
            
            # Add debug information: check the number of features in each category
            if target_outcome in result:
                outcome_data = result[target_outcome]
                if isinstance(outcome_data, dict) and "selected_features" in outcome_data:
                    selected_features = outcome_data["selected_features"]
                    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Debug information - number of features selected in each category:")
                    for category in ["Diag", "Proc", "Med", "TS"]:
                        features = selected_features.get(category, [])
                        print(f"{category}: {len(features)} features")
                        if category == "Med" and len(features) == 0:
                            print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} {category} category has no features selected!")
                        elif category == "Med" and len(features) > 0:
                            print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} {category} category selected features: {features[:3]}..." if len(features) > 3 else f"{Colors.GREEN}[SUCCESS]{Colors.RESET} {category} category selected features: {features}")
                else:
                    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Debug information - detected old format (simple list)")
                    if isinstance(outcome_data, list):
                        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Total features: {len(outcome_data)}")
                        # Check if there are medicine-related features
                        med_keywords = ["Usage", "Prednisone", "Vancomycin", "Insulin", "Furosemide", "Norepinephrine"]
                        med_features = [f for f in outcome_data if any(keyword in f for keyword in med_keywords)]
                        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Suspected Med features: {len(med_features)}")
                        if len(med_features) == 0:
                            print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} No medicine-related features found!")
            
            # 4. Validate results (using new exemption validation function)
            is_valid, validation_report = validate_causal_results_with_exemptions(
                result, config, target_outcome, categorized_features
            )
            
            # 5. Add metadata
            result["metadata"] = {
                "model_name": model_config["model_name"],
                "timestamp": datetime.now().isoformat(),
                "total_input_features": total_features,
                "target_outcome": target_outcome,
                "experiment_num": experiment_num,
                "generation_params": model_config["generation_params"],
                "validation_report": validation_report,
                "is_valid": is_valid,
                "prompt_length": len(prompt),
                "response_length": len(response),
                "exemption_enabled": config.get("allow_category_exemption", False),
                "category_requirements": config.get("category_min_percentages", {}),
                "format_version": "v2_categorized_with_exemptions"
            }
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"{Colors.RED}[ERROR]{Colors.RESET} JSON parsing failed: {e}")
            print(f"Original response: {response[:500]}...")
            print(f"Cleaned response: {cleaned_response[:500]}...")
            
            # Return the result containing the original response
            return {
                "error": "JSON parsing failed",
                "json_error": str(e),
                "raw_response": response,
                "cleaned_response": cleaned_response,
                "metadata": {
                    "model_name": model_config["model_name"],
                    "timestamp": datetime.now().isoformat(),
                    "target_outcome": target_outcome,
                    "experiment_num": experiment_num,
                    "status": "failed",
                    "format_version": "v2_categorized_with_exemptions"
                }
            }
            
    except Exception as e:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} Error occurred during generation: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "error": "Generation failed",
            "error_message": str(e),
            "metadata": {
                "model_name": model_config.get("model_name", "unknown"),
                "timestamp": datetime.now().isoformat(),
                "target_outcome": target_outcome,
                "experiment_num": experiment_num,
                "status": "error",
                "format_version": "v2_categorized_with_exemptions"
            }
        }

def find_best_feature_match(invalid_feature, all_valid_features, threshold=0.8):
    """
    Find the best fuzzy match for invalid features
    
    Args:
        invalid_feature (str): invalid feature description
        all_valid_features (list): all valid features list
        threshold (float): similarity threshold, default 0.8
        
    Returns:
        str or None: best match of valid feature, return None if no enough similarity
    """
    # Use difflib to calculate sequence matching
    best_match = None
    best_ratio = 0
    
    for valid_feature in all_valid_features:
        # Calculate similarity
        ratio = difflib.SequenceMatcher(None, invalid_feature.lower(), valid_feature.lower()).ratio()
        
        if ratio > best_ratio and ratio >= threshold:
            best_ratio = ratio
            best_match = valid_feature
    
    return best_match, best_ratio if best_match else (None, 0)

def filter_and_match_features(features, all_valid_features, experiment_num=None):
    """
    Filter invalid features and try fuzzy matching
    
    Args:
        features (list): original features list
        all_valid_features (set): all valid features set
        experiment_num (int, optional): experiment number, for logging
        
    Returns:
        tuple: (valid features list, match report)
    """
    valid_features = []
    match_report = {
        "original_count": len(features),
        "direct_matches": 0,
        "fuzzy_matches": 0,
        "no_matches": 0,
        "fuzzy_match_details": [],
        "unmatched_features": []
    }
    
    exp_info = f"Experiment {experiment_num}" if experiment_num else "Feature set"
    
    for feature in features:
        if feature in all_valid_features:
            # Direct match
            valid_features.append(feature)
            match_report["direct_matches"] += 1
        else:
            # Try fuzzy match
            best_match, similarity = find_best_feature_match(feature, list(all_valid_features))
            
            if best_match:
                valid_features.append(best_match)
                match_report["fuzzy_matches"] += 1
                match_report["fuzzy_match_details"].append({
                    "original": feature,
                    "matched": best_match,
                    "similarity": similarity
                })
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} {exp_info} fuzzy match: '{feature[:50]}...' → '{best_match[:50]}...' (similarity: {similarity:.2f})")
            else:
                match_report["no_matches"] += 1
                match_report["unmatched_features"].append(feature)
                print(f"{Colors.RED}[ERROR]{Colors.RESET} {exp_info} cannot match: '{feature[:50]}...'")
    
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} {exp_info} feature matching result: direct match {match_report['direct_matches']}, fuzzy match {match_report['fuzzy_matches']}, no match {match_report['no_matches']}")
    
    return valid_features, match_report

def generate_consensus_features(all_results, config, target_outcome):
    """
    Generate consensus features based on multiple experiments (supports categorized format, invalid feature filtering, fuzzy matching and exemption mechanism)
    
    Args:
        all_results (list): all experiment results list
        config (dict): configuration information
        target_outcome (str): target outcome
        
    Returns:
        dict: consensus result
    """
    print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Generating consensus features...")
    
    # 0. Get all valid features list (for validation and fuzzy matching)
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Loading valid features list for validation...")
    categorized_features, all_valid_features, _ = load_all_available_features()
    all_valid_features_set = set(all_valid_features)
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Loaded {len(all_valid_features)} valid features (4 categories)")
    
    # 1. Collect all successful experiment feature sets (supports new and old formats)
    feature_sets = []
    successful_experiments = []
    total_experiments = len(all_results)
    all_match_reports = []
    exemption_statistics = {
        "Diag": {"count": 0, "reasons": []},
        "Proc": {"count": 0, "reasons": []},
        "Med": {"count": 0, "reasons": []},
        "TS": {"count": 0, "reasons": []}
    }
    
    print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Start feature validation and matching...")
    
    for i, result in enumerate(all_results, 1):
        if "error" not in result and target_outcome in result:
            outcome_data = result[target_outcome]
            
            # Check result format
            if isinstance(outcome_data, dict) and "selected_features" in outcome_data:
                # New format (categorized features + exemptions)
                print(f"\n📋 Processing experiment {i} (new format, categorized features):")
                
                selected_features = outcome_data["selected_features"]
                exemptions = outcome_data.get("category_exemptions", {})
                
                # Collect all selected features
                all_selected_features = []
                category_counts = {}
                
                for category, features in selected_features.items():
                    all_selected_features.extend(features)
                    category_counts[category] = len(features)
                    
                    # Record exemption statistics
                    if category in exemptions and exemptions[category]:
                        exemption_statistics[category]["count"] += 1
                        exemption_statistics[category]["reasons"].append(exemptions[category])
                
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} Number of features in each category: {category_counts}")
                if exemptions:
                    exempted_categories = [cat for cat, reason in exemptions.items() if reason]
                    if exempted_categories:
                        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Exempted categories: {exempted_categories}")
                
                raw_features = all_selected_features
            else:
                # Old format (simple list)
                print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Processing experiment {i} (old format, simple list):")
                raw_features = outcome_data if isinstance(outcome_data, list) else []
            
            if raw_features:  # Ensure feature set is not empty
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} Number of raw features: {len(raw_features)}")
                
                # Apply filtering and fuzzy matching
                valid_features, match_report = filter_and_match_features(
                    raw_features, all_valid_features_set, experiment_num=i
                )
                
                if valid_features:  # Only participate in consensus if there are valid features after processing
                    feature_sets.append(valid_features)
                    successful_experiments.append(i)
                    all_match_reports.append(match_report)
                    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Experiment {i}: {len(valid_features)} valid features (original {len(raw_features)} features)")
                else:
                    print(f"{Colors.RED}[ERROR]{Colors.RESET} Experiment {i}: No valid features after filtering, skipped")
            else:
                print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Experiment {i}: Original feature list is empty")
    
    successful_count = len(feature_sets)
    success_rate = successful_count / total_experiments if total_experiments > 0 else 0
    
    print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Experiment success statistics: {successful_count}/{total_experiments} ({success_rate*100:.1f}%)")
    
    # Summarize exemption statistics
    total_exemptions = sum(stats["count"] for stats in exemption_statistics.values())
    if total_exemptions > 0:
        print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Category exemption statistics:")
        for category, stats in exemption_statistics.items():
            if stats["count"] > 0:
                percentage = stats["count"] / total_experiments * 100
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} {category}: {stats['count']}/{total_experiments} exemptions ({percentage:.1f}%)")
    
    # Summarize matching statistics
    if all_match_reports:
        total_original = sum(report["original_count"] for report in all_match_reports)
        total_direct = sum(report["direct_matches"] for report in all_match_reports)
        total_fuzzy = sum(report["fuzzy_matches"] for report in all_match_reports)
        total_unmatched = sum(report["no_matches"] for report in all_match_reports)
        
        print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Feature matching summary:")
        print(f"Total original features: {total_original}")
        print(f"Direct matches: {total_direct} ({total_direct/total_original*100:.1f}%)")
        print(f"Fuzzy matches: {total_fuzzy} ({total_fuzzy/total_original*100:.1f}%)")
        print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} No matches: {total_unmatched} ({total_unmatched/total_original*100:.1f}%)")
    
    # 2. Strict mode quality control check
    strict_mode = config.get("strict_mode", False)
    min_success_rate = config.get("min_success_rate", 1.0)
    
    if strict_mode and success_rate < min_success_rate:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} Quality control failed: success rate {success_rate*100:.1f}% < required {min_success_rate*100:.0f}%")
        return None
    
    # 3. Check minimum experiment number requirement
    min_experiments_for_consensus = 2
    if successful_count < min_experiments_for_consensus:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} Not enough successful experiments ({successful_count}<{min_experiments_for_consensus}), cannot generate reliable consensus")
        return None
    
    # 4. Count feature frequency
    feature_counts = {}
    for feature_set in feature_sets:
        for feature in feature_set:
            feature_counts[feature] = feature_counts.get(feature, 0) + 1
    
    # 5. Generate consensus features based on threshold
    consensus_threshold = 2  # At least 2 times
    consensus_features = [
        feature for feature, count in feature_counts.items() 
        if count >= consensus_threshold
    ]
    
    # 6. Sort features by frequency
    sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Categorized features
    high_consensus = [(f, c) for f, c in sorted_features if c >= consensus_threshold]
    low_consensus = [(f, c) for f, c in sorted_features if c < consensus_threshold]
    
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Consensus analysis completed:")
    print(f"Total features (unique): {len(feature_counts)}")
    print(f"High consensus features (≥{consensus_threshold} times): {len(high_consensus)}")
    print(f"Low consensus features (<{consensus_threshold} times): {len(low_consensus)}")
    
    # 7. Build consensus result
    consensus_result = {
        target_outcome: {
            "consensus_feature_set": consensus_features,
            "consensus_threshold": consensus_threshold,
            "total_successful_experiments": len(feature_sets),
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
            },
            "exemption_statistics": exemption_statistics,
            "feature_matching_statistics": {
                "total_original_features": sum(report["original_count"] for report in all_match_reports) if all_match_reports else 0,
                "total_direct_matches": sum(report["direct_matches"] for report in all_match_reports) if all_match_reports else 0,
                "total_fuzzy_matches": sum(report["fuzzy_matches"] for report in all_match_reports) if all_match_reports else 0,
                "total_unmatched_features": sum(report["no_matches"] for report in all_match_reports) if all_match_reports else 0,
                "fuzzy_match_details": [detail for report in all_match_reports for detail in report["fuzzy_match_details"]] if all_match_reports else [],
                "unmatched_features": [feature for report in all_match_reports for feature in report["unmatched_features"]] if all_match_reports else []
            }
        },
        "metadata": {
            "model_name": config['model_config']['model_name'],
            "timestamp": datetime.now().isoformat(),
            "target_outcome": target_outcome,
            "generation_method": "consensus_from_multiple_experiments_with_categorized_features_and_exemptions",
            "consensus_threshold": consensus_threshold,
            "total_experiments": len(all_results),
            "successful_experiments": len(feature_sets),
            "feature_filtering_enabled": True,
            "fuzzy_matching_enabled": True,
            "fuzzy_matching_threshold": 0.8,
            "exemption_mechanism_enabled": config.get("allow_category_exemption", False),
            "format_version": "v2_categorized_with_exemptions"
        }
    }
    
    return consensus_result

def save_experiment_result(result, session_dir, model_name, target_outcome, experiment_num, session_timestamp):
    """
    Save single experiment result
    
    Args:
        result (dict): experiment result
        session_dir (str): session directory path
        model_name (str): model name
        target_outcome (str): target label
        experiment_num (int): experiment number
        session_timestamp (str): session timestamp
        
    Returns:
        str: saved file path
    """
    clean_model_name = model_name.replace(" ", "_").replace("-", "_")
    
    # Generate file name
    filename = f"causal_features_{clean_model_name}_{target_outcome}_exp{experiment_num}_{session_timestamp}.json"
    filepath = os.path.join(session_dir, filename)
    
    # Save result
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Experiment {experiment_num} result saved: {filename}")
    return filepath

def save_consensus_results(consensus_result, session_dir, model_name, target_outcome, session_timestamp):
    """
    Save consensus result (JSON and TXT two versions)
    
    Args:
        consensus_result (dict): consensus result
        session_dir (str): session directory path
        model_name (str): model name
        target_outcome (str): target label
        session_timestamp (str): session timestamp
        
    Returns:
        tuple: (JSON file path, TXT file path)
    """
    clean_model_name = model_name.replace(" ", "_").replace("-", "_")
    
    # 1. Save JSON version
    json_filename = f"causal_features_{clean_model_name}_{target_outcome}_consensus_{session_timestamp}.json"
    json_filepath = os.path.join(session_dir, json_filename)
    
    with open(json_filepath, 'w', encoding='utf-8') as f:
        json.dump(consensus_result, f, ensure_ascii=False, indent=2)
    
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Consensus JSON saved: {json_filename}")
    
    # 2. Save TXT version
    txt_filename = f"causal_features_{clean_model_name}_{target_outcome}_consensus_{session_timestamp}.txt"
    txt_filepath = os.path.join(session_dir, txt_filename)
    
    # Get consensus features and statistics
    consensus_features = consensus_result[target_outcome]["consensus_feature_set"]
    consensus_threshold = consensus_result[target_outcome]["consensus_threshold"]
    total_experiments = consensus_result[target_outcome]["total_successful_experiments"]
    feature_frequency = consensus_result[target_outcome]["feature_frequency_analysis"]["all_feature_counts"]
    high_consensus = consensus_result[target_outcome]["feature_frequency_analysis"]["high_consensus_features"]
    low_consensus = consensus_result[target_outcome]["feature_frequency_analysis"]["low_consensus_features"]
    matching_stats = consensus_result[target_outcome]["feature_matching_statistics"]
    
    # Get outcome description
    outcome_descriptions = get_outcome_descriptions()
    outcome_description = outcome_descriptions.get(target_outcome, target_outcome)
    
    with open(txt_filepath, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"Consensus causal feature set - {target_outcome}\n")
        f.write("="*80 + "\n\n")
        
        f.write("Consensus configuration:\n")
        f.write("-"*50 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Target label: {target_outcome} ({outcome_description})\n")
        f.write(f"Consensus threshold: ≥{consensus_threshold} times\n")
        f.write(f"Successful experiments: {total_experiments} experiments\n")
        f.write(f"Session time: {session_timestamp}\n\n")
        
        f.write("[INFO] Consensus statistics:\n")
        f.write("-"*50 + "\n")
        f.write(f"Total features (unique): {len(feature_frequency)}\n")
        f.write(f"Consensus features: {len(consensus_features)}\n")
        f.write(f"Consensus ratio: {len(consensus_features) / len(feature_frequency) * 100:.1f}%\n\n")
        
        # Feature matching statistics
        f.write("[INFO] Feature matching statistics:\n")
        f.write("-"*50 + "\n")
        f.write(f"Total original features: {matching_stats['total_original_features']}\n")
        f.write(f"Direct matches: {matching_stats['total_direct_matches']} ({matching_stats['total_direct_matches']/matching_stats['total_original_features']*100:.1f}%)\n")
        f.write(f"Fuzzy matches: {matching_stats['total_fuzzy_matches']} ({matching_stats['total_fuzzy_matches']/matching_stats['total_original_features']*100:.1f}%)\n")
        f.write(f"No matches: {matching_stats['total_unmatched_features']} ({matching_stats['total_unmatched_features']/matching_stats['total_original_features']*100:.1f}%)\n")
        
        # Category exemption statistics
        exemption_stats = consensus_result[target_outcome].get("exemption_statistics", {})
        total_exemptions = sum(stats["count"] for stats in exemption_stats.values())
        if total_exemptions > 0:
            f.write(f"\n[INFO] Category exemption statistics:\n")
            f.write("-"*50 + "\n")
            for category, stats in exemption_stats.items():
                if stats["count"] > 0:
                    percentage = stats["count"] / total_experiments * 100
                    f.write(f"{category} category: {stats['count']}/{total_experiments} exemptions ({percentage:.1f}%)\n")
                    
                    # Display exemption reasons (if any)
                    if stats["reasons"]:
                        f.write(f"   Exemption reasons example:\n")
                        for i, reason in enumerate(stats["reasons"][:2], 1):  # Display up to 2 reasons
                            f.write(f" {i}. {reason[:100]}...\n")
                        if len(stats["reasons"]) > 2:
                            f.write(f" ... and {len(stats['reasons']) - 2} other reasons\n")
        else:
            f.write(f"\n[INFO] Category exemption statistics: no exemption records\n")
        
        # Display fuzzy match details (if any)
        if matching_stats['fuzzy_match_details']:
            f.write(f"\n[INFO] Fuzzy match details ({len(matching_stats['fuzzy_match_details'])} features):\n")
            f.write("-"*50 + "\n")
            for i, detail in enumerate(matching_stats['fuzzy_match_details'][:10], 1):  # Display up to 10 details
                f.write(f"{i:2d}. '{detail['original'][:40]}...' → '{detail['matched'][:40]}...' (similarity: {detail['similarity']:.2f})\n")
            if len(matching_stats['fuzzy_match_details']) > 10:
                f.write(f" ... and {len(matching_stats['fuzzy_match_details']) - 10} other fuzzy matches\n")
        
        # Display unmatched features (if any)
        if matching_stats['unmatched_features']:
            f.write(f"\n[ERROR] Unmatched features ({len(matching_stats['unmatched_features'])} features):\n")
            f.write("-"*50 + "\n")
            for i, feature in enumerate(matching_stats['unmatched_features'][:5], 1):  
                f.write(f"{i:2d}. {feature[:60]}...\n")
            if len(matching_stats['unmatched_features']) > 5:
                f.write(f" ... and {len(matching_stats['unmatched_features']) - 5} other unmatched features\n")
        
        f.write("\n")
        
        f.write("[SUCCESS] Consensus feature list:\n")
        f.write("-"*50 + "\n")
        for i, feature in enumerate(consensus_features, 1):
            count = feature_frequency[feature]
            percentage = count / total_experiments * 100
            f.write(f"{i:3d}. {feature} - {count}/{total_experiments} times ({percentage:.0f}%)\n")
        
        if low_consensus:
            f.write(f"\n[INFO] Unconsensus features (<{consensus_threshold} times):\n")
            f.write("-"*50 + "\n")
            for feature, count in low_consensus:
                percentage = count / total_experiments * 100
                f.write(f"{feature} - {count}/{total_experiments} times ({percentage:.0f}%)\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Consensus analysis completed\n")
        f.write("="*80 + "\n")
    
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Consensus TXT saved: {txt_filename}")
    
    return json_filepath, txt_filepath

def save_session_summary(all_results, config, session_dir, target_outcome, session_timestamp):
    """
    Generate session summary file
    
    Args:
        all_results (list): all experiment results
        config (dict): configuration information
        session_dir (str): session directory path
        target_outcome (str): target label
        session_timestamp (str): session timestamp
        
    Returns:
        str: summary file path
    """
    clean_model_name = config['model_config']['model_name'].replace(" ", "_").replace("-", "_")
    summary_filename = f"session_summary_{clean_model_name}_{target_outcome}_{session_timestamp}.txt"
    summary_filepath = os.path.join(session_dir, summary_filename)
    
    # Analyze experiment results
    successful_experiments = []
    failed_experiments = []
    
    for i, result in enumerate(all_results, 1):
        if "error" in result:
            failed_experiments.append(i)
        else:
            successful_experiments.append(result)
    
    # Get outcome description
    outcome_descriptions = get_outcome_descriptions()
    outcome_description = outcome_descriptions.get(target_outcome, target_outcome)
    
    with open(summary_filepath, 'w', encoding='utf-8') as f:
        # Write title and basic information
        f.write("="*80 + "\n")
        f.write("Causal feature generation session summary report\n")
        f.write("="*80 + "\n\n")
        
        # Configuration information
        f.write("Configuration information:\n")
        f.write("-"*50 + "\n")
        f.write(f"Model: {config['model_config']['model_name']}\n")
        f.write(f"Target label: {target_outcome} ({outcome_description})\n")
        f.write(f"Experiment count: {config['experiment_count']}\n")
        f.write(f"Input features: 262 features (excluding 4 basic features)\n")
        f.write(f"Session time: {session_timestamp}\n")
        f.write(f"Temperature setting: {config['model_config']['generation_params']['temperature']}\n\n")
        
        # Experiment result statistics
        f.write("[INFO] Experiment result statistics:\n")
        f.write("-"*50 + "\n")
        f.write(f"Total experiments: {len(all_results)}\n")
        f.write(f"Successful experiments: {len(successful_experiments)}\n") 
        f.write(f"Failed experiments: {len(failed_experiments)}\n")
        if failed_experiments:
            f.write(f"Failed experiment numbers: {', '.join(map(str, failed_experiments))}\n")
        f.write(f"Success rate: {len(successful_experiments)/len(all_results)*100:.1f}%\n\n")
        
        if successful_experiments:
            # Feature statistics analysis
            f.write("[INFO] Feature statistics analysis:\n")
            f.write("-"*50 + "\n")
            
            feature_counts_per_exp = []
            all_features_found = set()
            
            for i, result in enumerate(successful_experiments, 1):
                if target_outcome in result:
                    features = result[target_outcome]
                    feature_count = len(features)
                    feature_counts_per_exp.append(feature_count)
                    all_features_found.update(features)
                    f.write(f"Experiment {successful_experiments.index(result)+1}: {feature_count} features\n")
            
            if feature_counts_per_exp:
                avg_features = sum(feature_counts_per_exp) / len(feature_counts_per_exp)
                f.write(f"Average number of features per experiment: {avg_features:.1f}\n")
                f.write(f"Total number of features found (unique): {len(all_features_found)}\n\n")
        
        # Quality control information
        strict_mode = config.get("strict_mode", False)
        min_success_rate = config.get("min_success_rate", 1.0)
        
        f.write("Quality control information:\n")
        f.write("-"*50 + "\n")
        f.write(f"Strict mode: {'Enabled' if strict_mode else 'Disabled'}\n")
        f.write(f"Minimum success rate requirement: {min_success_rate*100:.0f}%\n")
        
        if strict_mode:
            success_rate = len(successful_experiments) / len(all_results) if all_results else 0
            quality_passed = success_rate >= min_success_rate
            f.write(f"Actual success rate: {success_rate*100:.1f}%\n")
            f.write(f"Quality control status: {'[SUCCESS] Passed' if quality_passed else '[ERROR] Failed'}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Session summary completed\n")
        f.write("="*80 + "\n")
    
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Session summary saved: {summary_filename}")
    return summary_filepath

def update_final_consensus_file(output_dir, model_name, target_outcome, consensus_features):
    """
    Update or create FINAL consensus file
    
    Args:
        output_dir (str): output directory
        model_name (str): model name
        target_outcome (str): target label
        consensus_features (list): consensus feature list
        
    Returns:
        str: FINAL file path
    """
    clean_model_name = model_name.replace(" ", "_").replace("-", "_")
    final_filename = f"FINAL_{clean_model_name}_causal_features_consensus.txt"
    final_filepath = os.path.join(output_dir, final_filename)
    
    # Format feature list
    features_str = str(consensus_features).replace("'", "'")  # Uniform quote format
    new_line = f"{target_outcome} = {features_str}"
    
    # Read existing file content (if exists)
    existing_lines = []
    label_found = False
    
    if os.path.exists(final_filepath):
        with open(final_filepath, 'r', encoding='utf-8') as f:
            existing_lines = f.readlines()
        
        # Check if label exists, if exists, update
        for i, line in enumerate(existing_lines):
            if line.strip().startswith(f"{target_outcome} ="):
                existing_lines[i] = new_line + "\n"
                label_found = True
                break
    
    # Write to file
    with open(final_filepath, 'w', encoding='utf-8') as f:
        if existing_lines:
            # Write existing content (may be updated)
            f.writelines(existing_lines)
            
            # If label does not exist, append new line
            if not label_found:
                # Ensure file ends with empty line
                if existing_lines and not existing_lines[-1].strip() == "":
                    f.write("\n")
                f.write(new_line + "\n\n")
        else:
            # Create new file
            f.write(new_line + "\n\n")
    
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} FINAL file updated: {final_filename}")
    return final_filepath

def save_causal_generation_results(results, config, all_features, timestamp):
    """
    Save causal feature generation results
    
    Args:
        results (dict): generation results
        config (dict): configuration information
        all_features (list): all feature list
        timestamp (str): timestamp
        
    Returns:
        dict: save path information
    """
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    model_name = config["model_config"]["model_name"]
    clean_model_name = model_name.replace(" ", "_").replace("-", "_")
    
    saved_files = {}
    
    # 1. Save main result file
    main_file = f"causal_features_{clean_model_name}_{timestamp}.json"
    main_path = os.path.join(output_dir, main_file)
    
    with open(main_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Main result saved: {main_file}")
    saved_files["main_file"] = main_path
    
    # 2. Save individual outcome files (if enabled)
    if config.get("save_individual_files", False) and "error" not in results:
        individual_dir = os.path.join(output_dir, "individual_outcomes")
        os.makedirs(individual_dir, exist_ok=True)
        
        for outcome in config["target_outcomes"]:
            if outcome in results:
                individual_file = f"{outcome}_causal_features_{timestamp}.txt"
                individual_path = os.path.join(individual_dir, individual_file)
                
                with open(individual_path, 'w', encoding='utf-8') as f:
                    f.write(f"Causal feature list - {outcome}\n")
                    f.write("="*60 + "\n\n")
                    f.write(f"Generation time: {timestamp}\n")
                    f.write(f"Model: {model_name}\n")
                    f.write(f"Feature count: {len(results[outcome])}\n\n")
                    f.write("Feature list:\n")
                    f.write("-"*30 + "\n")
                    
                    for i, feature in enumerate(results[outcome], 1):
                        f.write(f"{i:3d}. {feature}\n")
                
                print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} {outcome} individual file saved")
        
        saved_files["individual_dir"] = individual_dir
    
    return saved_files

def generate_causal_summary_report(results, config, all_features, timestamp):
    """
    Generate detailed summary report
    
    Args:
        results (dict): generation results
        config (dict): configuration information
        all_features (list): all feature list
        timestamp (str): timestamp
        
    Returns:
        str: summary report file path
    """
    output_dir = config["output_dir"]
    summary_file = os.path.join(output_dir, f"causal_generation_summary_{timestamp}.txt")
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        # Write title and basic information
        f.write("="*80 + "\n")
        f.write("Causal feature generation summary report\n")
        f.write("="*80 + "\n\n")
        
        # Configuration information
        f.write("Configuration information:\n")
        f.write("-"*50 + "\n")
        f.write(f"Model: {config['model_config']['model_name']}\n")
        f.write(f"Input features: {len(all_features)}\n")
        f.write(f"Target outcomes: {len(config['target_outcomes'])}\n")
        f.write(f"Generation time: {timestamp}\n")
        f.write(f"Temperature setting: {config['model_config']['generation_params']['temperature']}\n")
        f.write(f"Max tokens: {config['model_config']['generation_params']['max_tokens']}\n\n")
        
        # Target outcome list
        f.write("Target outcome list:\n")
        f.write("-"*50 + "\n")
        outcome_descriptions = get_outcome_descriptions()
        for i, outcome in enumerate(config['target_outcomes'], 1):
            description = outcome_descriptions.get(outcome, outcome)
            f.write(f"{i:2d}. {outcome}: {description}\n")
        f.write("\n")
        
        if "error" in results:
            # Error case
            f.write("[ERROR] Generation failed:\n")
            f.write("-"*50 + "\n")
            f.write(f"Error type: {results['error']}\n")
            if "error_message" in results:
                f.write(f"Error message: {results['error_message']}\n")
        else:
            # Success case
            # Generate result statistics
            f.write("Generate result statistics:\n")
            f.write("-"*50 + "\n")
            
            total_features = 0
            outcome_stats = []
            
            for outcome in config['target_outcomes']:
                if outcome in results:
                    feature_count = len(results[outcome])
                    total_features += feature_count
                    outcome_stats.append((outcome, feature_count))
                    f.write(f"{outcome}: {feature_count} causal features\n")
                else:
                    f.write(f"{outcome}: No results found\n")
            
            f.write(f"\nTotal causal features: {total_features}\n")
            f.write(f"Average number of features per outcome: {total_features/len(config['target_outcomes']):.1f} features\n\n")
            
            # Feature overlap analysis
            if len([stats for stats in outcome_stats if stats[1] > 0]) > 1:
                f.write("Feature overlap analysis:\n")
                f.write("-"*50 + "\n")
                
                # Count how many times each feature appears in each outcome
                feature_counts = {}
                for outcome in config['target_outcomes']:
                    if outcome in results:
                        for feature in results[outcome]:
                            feature_counts[feature] = feature_counts.get(feature, 0) + 1
                
                # Sort by frequency
                sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
                
                f.write(f"Total unique features: {len(sorted_features)}\n")
                f.write("Most common causal features (appearing in multiple outcomes):\n")
                
                # Display features appearing in multiple outcomes
                multi_outcome_features = [(feature, count) for feature, count in sorted_features if count > 1]
                if multi_outcome_features:
                    for i, (feature, count) in enumerate(multi_outcome_features[:20], 1):  # Display up to 20 features
                        percentage = count / len(config['target_outcomes']) * 100
                        f.write(f"{i:3d}. {feature} - {count}/{len(config['target_outcomes'])} outcomes ({percentage:.0f}%)\n")
                else:
                    f.write("No features appear in multiple outcomes\n")
                
                f.write(f"\nFeatures appearing in only one outcome: {len(sorted_features) - len(multi_outcome_features)} features\n\n")
            
            # Detailed feature list for each outcome
            f.write("[INFO] Detailed feature list:\n")
            f.write("-"*50 + "\n")
            
            for outcome in config['target_outcomes']:
                if outcome in results and results[outcome]:
                    f.write(f"\n【{outcome}】({len(results[outcome])} features):\n")
                    for i, feature in enumerate(results[outcome], 1):
                        f.write(f"  {i:3d}. {feature}\n")
                else:
                    f.write(f"\n【{outcome}】: No features or not generated\n")
        
        # Validation information
        if "metadata" in results and "validation_report" in results["metadata"]:
            validation = results["metadata"]["validation_report"]
            f.write(f"\n[INFO] Validation information:\n")
            f.write("-"*50 + "\n")
            f.write(f"Result validity: {'[SUCCESS] Valid' if results['metadata']['is_valid'] else '[ERROR] Invalid'}\n")
            f.write(f"Found outcome: {validation['found_outcome']}/{validation['total_input_features']}\n") 
            if validation['invalid_features']:
                f.write(f"Outcome with invalid features: {len(validation['invalid_features'])}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Report generated\n")
        f.write("="*80 + "\n")
    
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Summary report saved: {os.path.basename(summary_file)}")
    return summary_file

def validate_config():
    """
    Validate configuration validity
    """
    print("Validating configuration...")
    
    # Check required files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    required_files = ["selected_features.txt", "features-desc.csv", "prediction.py"]
    
    for file_name in required_files:
        file_path = os.path.join(script_dir, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{Colors.RED}[ERROR]{Colors.RESET} Required file not found: {file_path}")
    
    # Validate run mode
    run_mode = CAUSAL_GENERATION_CONFIG.get("run_mode", "single_label")
    if run_mode not in ["single_label", "all_labels"]:
        raise ValueError(f"{Colors.RED}[ERROR]{Colors.RESET} Unsupported run mode: {run_mode}")
    
    # Validate outcome configuration (single label mode)
    if run_mode == "single_label":
        target_outcome = CAUSAL_GENERATION_CONFIG["target_outcome"]
        if not target_outcome:
            raise ValueError(f"{Colors.RED}[ERROR]{Colors.RESET} Target outcome cannot be empty in single label mode")
        if target_outcome not in ALL_OUTCOME_LABELS:
            raise ValueError(f"{Colors.RED}[ERROR]{Colors.RESET} Invalid target outcome: {target_outcome}")
    
    # Validate category exemption configuration
    category_min_percentages = CAUSAL_GENERATION_CONFIG.get("category_min_percentages", {})
    for category, min_percent in category_min_percentages.items():
        if min_percent < 0 or min_percent > 1:
            raise ValueError(f"{Colors.RED}[ERROR]{Colors.RESET} Category minimum percentage '{min_percent}' must be between 0 and 1")
    
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Configuration validated")

def create_batch_session_directory(config):
    """
    Create batch session directory
    
    Args:
        config (dict): configuration information
        
    Returns:
        str: batch session directory path
    """
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_uuid = str(uuid.uuid4())[:8]
    clean_model_name = config['model_config']['model_name'].replace(" ", "_").replace("-", "_")
    
    batch_session_dir_name = f"{clean_model_name}_batch_{session_timestamp}_{session_uuid}"
    batch_session_dir = os.path.join(config["output_dir"], batch_session_dir_name)
    
    os.makedirs(batch_session_dir, exist_ok=True)
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Batch session directory created: {batch_session_dir_name}")
    
    return batch_session_dir

def create_label_subdirectory(batch_session_dir, target_outcome):
    """
    Create subdirectory for specific label
    
    Args:
        batch_session_dir (str): batch session directory
        target_outcome (str): target outcome
        
    Returns:
        str: label subdirectory path
    """
    label_dir = os.path.join(batch_session_dir, target_outcome)
    os.makedirs(label_dir, exist_ok=True)
    return label_dir

def process_single_label_in_batch(target_outcome, config, label_dir):
    """
    Process single label in batch mode
    
    Args:
        target_outcome (str): target outcome
        config (dict): configuration information
        label_dir (str): label directory path
        
    Returns:
        dict: processing result
    """
    try:
        # 1. Load 262 features (excluding basic features)
        print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Loading all features...")
        categorized_features, all_feature_descriptions, categorized_names = load_all_available_features()
        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} All features loaded: {len(all_feature_descriptions)} features (4 categories)")
        
        # 2. Create timestamp
        session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 3. Execute multiple experiments
        experiment_count = config["experiment_count"]
        all_results = []
        
        print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Starting {experiment_count} repeated experiments")
        print(f"-" * 40)
        
        for i in range(1, experiment_count + 1):
            print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Experiment {i}/{experiment_count}")
            
            # Single experiment (using new categorized feature structure)
            result = generate_causal_features_single_experiment(
                config['model_config'], categorized_features, target_outcome, config, i
            )
            
            # Save single experiment result
            save_experiment_result(
                result, label_dir, config['model_config']['model_name'], 
                target_outcome, i, session_timestamp
            )
            
            all_results.append(result)
            print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Experiment {i} completed")
        
        # Save session summary
        print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Generating session summary...")
        save_session_summary(all_results, config, label_dir, target_outcome, session_timestamp)
        
        # Generate consensus
        print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Generating consensus feature set")
        print(f"-" * 40)
        
        consensus_result = generate_consensus_features(all_results, config, target_outcome)
        
        if consensus_result:
            # Save consensus result
            consensus_json_path, consensus_txt_path = save_consensus_results(
                consensus_result, label_dir, config['model_config']['model_name'], 
                target_outcome, session_timestamp
            )
            
            # Extract consensus feature set
            consensus_features = consensus_result[target_outcome]["consensus_feature_set"]
            
            print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Consensus generated successfully!")
            print(f"Consensus feature count: {len(consensus_features)}")
            
            return {
                "success": True,
                "target_outcome": target_outcome,
                "consensus_features": consensus_features,
                "consensus_count": len(consensus_features),
                "total_experiments": experiment_count,
                "successful_experiments": sum(1 for r in all_results if 'error' not in r),
                "session_dir": label_dir,
                "consensus_json_path": consensus_json_path,
                "consensus_txt_path": consensus_txt_path,
                "session_timestamp": session_timestamp
            }
        else:
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Consensus generation failed!")
            return {
                "success": False,
                "target_outcome": target_outcome,
                "error": "Consensus generation failed: quality control not passed or successful experiment count insufficient",
                "total_experiments": experiment_count,
                "successful_experiments": sum(1 for r in all_results if 'error' not in r),
                "session_dir": label_dir,
                "session_timestamp": session_timestamp
            }
            
    except Exception as e:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} Error processing label {target_outcome}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "target_outcome": target_outcome,
            "error": str(e),
            "session_dir": label_dir,
            "session_timestamp": session_timestamp
        }

def generate_batch_summary(all_results, batch_session_dir, config):
    """
    Generate summary report for batch run
    
    Args:
        all_results (dict): all labels processing results
        batch_session_dir (str): batch session directory
        config (dict): configuration information
        
    Returns:
        str: summary report file path
    """
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_model_name = config['model_config']['model_name'].replace(" ", "_").replace("-", "_")
    summary_filename = f"batch_run_summary_{clean_model_name}_{session_timestamp}.txt"
    summary_filepath = os.path.join(batch_session_dir, summary_filename)
    
    # Statistics
    total_labels = len(ALL_OUTCOME_LABELS)
    successful_labels = sum(1 for result in all_results.values() if result.get("success", False))
    failed_labels = total_labels - successful_labels
    total_consensus_features = sum(result.get("consensus_count", 0) for result in all_results.values())
    
    with open(summary_filepath, 'w', encoding='utf-8') as f:
        # Write title and basic information
        f.write("="*80 + "\n")
        f.write("Batch causal feature generation summary report\n")
        f.write("="*80 + "\n\n")
        
        # Configuration information
        f.write("Batch run configuration:\n")
        f.write("-"*50 + "\n")
        f.write(f"Model: {config['model_config']['model_name']}\n")
        f.write(f"Run mode: batch process all labels\n")
        f.write(f"Number of experiments per label: {config['experiment_count']}\n")
        f.write(f"Input features: 262 features (excluding 4 basic features)\n")
        f.write(f"Batch session time: {session_timestamp}\n")
        f.write(f"Exemption mechanism: {'Enabled' if config.get('allow_category_exemption', False) else 'Disabled'}\n\n")
        
        # Overall statistics
        f.write("Batch run statistics:\n")
        f.write("-"*50 + "\n")
        f.write(f"Total labels processed: {total_labels}\n")
        f.write(f"Successful labels: {successful_labels}\n")
        f.write(f"Failed labels: {failed_labels}\n")
        f.write(f"Total experiments: {total_labels * config['experiment_count']}\n")
        f.write(f"Total consensus features: {total_consensus_features}\n")
        if successful_labels > 0:
            f.write(f"Average number of features per label: {total_consensus_features / successful_labels:.1f}\n")
        f.write("\n")
        
        # Detailed results for each label
        f.write("Detailed results for each label:\n")
        f.write("-"*50 + "\n")
        
        for label in ALL_OUTCOME_LABELS:
            result = all_results.get(label, {})
            if result.get("success", False):
                f.write(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} {label}:\n")
                f.write(f"Consensus feature count: {result.get('consensus_count', 0)}\n")
                f.write(f"Successful experiments: {result.get('successful_experiments', 0)}/{result.get('total_experiments', 0)}\n")
                f.write(f"Result directory: {os.path.basename(result.get('session_dir', ''))}\n")
            else:
                f.write(f"{Colors.RED}[ERROR]{Colors.RESET} {label}:\n")
                f.write(f"Error reason: {result.get('error', 'Unknown error')}\n")
                f.write(f"Successful experiments: {result.get('successful_experiments', 0)}/{result.get('total_experiments', 0)}\n")
            f.write("\n")
        
        # File organization structure
        f.write("File organization structure:\n")
        f.write("-"*50 + "\n")
        f.write(f"Batch session directory: {os.path.basename(batch_session_dir)}/\n")
        for label in ALL_OUTCOME_LABELS:
            result = all_results.get(label, {})
            if result.get("success", False):
                f.write(f"├── {label}/\n")
                f.write(f"│   ├── experiment result files (3 .json)\n")
                f.write(f"│   ├── consensus result (.json & .txt)\n")
                f.write(f"│   └── session summary (.txt)\n")
        f.write(f"└── {summary_filename}\n\n")
        
        f.write("="*80 + "\n")
        f.write("Batch run completed\n")
        f.write("="*80 + "\n")
    
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Batch run summary saved: {summary_filename}")
    return summary_filepath

def update_final_consensus_batch(all_results, config):
    """
    Update FINAL consensus file
    
    Args:
        all_results (dict): all labels processing results
        config (dict): configuration information
    """
    print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Updating FINAL consensus file...")
    
    output_dir = config["output_dir"]
    model_name = config['model_config']['model_name']
    
    successful_count = 0
    
    for label in ALL_OUTCOME_LABELS:
        result = all_results.get(label, {})
        if result.get("success", False):
            consensus_features = result.get("consensus_features", [])
            if consensus_features:
                update_final_consensus_file(output_dir, model_name, label, consensus_features)
                successful_count += 1
    
            print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} FINAL file updated, successfully added {successful_count}/{len(ALL_OUTCOME_LABELS)} labels")

def process_all_labels(config):
    """
    Process all labels in batch mode
    
    Args:
        config (dict): configuration information
    """
    print(f"\n{'='*80}")
    print(f"Batch causal feature generation started - {config['model_config']['display_name']}")
    print(f"{'='*80}")
    print(f"Expected labels to process: {len(ALL_OUTCOME_LABELS)}")
    print(f"Expected total experiments: {len(ALL_OUTCOME_LABELS) * config['experiment_count']}")
    print(f"Label list: {', '.join(ALL_OUTCOME_LABELS)}")
    print(f"{'='*80}")
    
    # User confirmation
    print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Continue with batch processing? (y/n): ", end="")
    user_input = input().strip().lower()
    if user_input not in ['y', 'yes']:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} User cancelled batch operation")
        return
        
    start_time = datetime.now()
    
    # 1. Create batch session directory
    batch_session_dir = create_batch_session_directory(config)
    
    # 2. Iterate through all labels
    all_results = {}
    
    for i, target_outcome in enumerate(ALL_OUTCOME_LABELS, 1):

        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Processing label {i}/{len(ALL_OUTCOME_LABELS)}: {target_outcome}")
        print(f"{'='*80}")
        
        # 3. Create subdirectory for current label
        label_dir = create_label_subdirectory(batch_session_dir, target_outcome)
        
        # 4. Process single label
        label_result = process_single_label_in_batch(target_outcome, config, label_dir)
        all_results[target_outcome] = label_result
        
        if label_result.get("success", False):
            print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Label {target_outcome} processed successfully")
        else:
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Label {target_outcome} processing failed")
    
    end_time = datetime.now()
    total_duration = end_time - start_time
    
    # 5. Generate batch run summary
    print(f"\n{'='*80}")
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Generating batch run summary...")
    print(f"{'='*80}")
    
    generate_batch_summary(all_results, batch_session_dir, config)
    
    # 6. Update FINAL file (summarize all results at once)
    update_final_consensus_batch(all_results, config)
    
    # 7. Final statistics
    successful_labels = sum(1 for result in all_results.values() if result.get("success", False))
    failed_labels = len(ALL_OUTCOME_LABELS) - successful_labels
    total_consensus_features = sum(result.get("consensus_count", 0) for result in all_results.values())
    
    print(f"\n{'='*80}")
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Batch causal feature generation completed!")
    print(f"{'='*80}")
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Batch session directory: {os.path.basename(batch_session_dir)}")
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Processed labels: {successful_labels}/{len(ALL_OUTCOME_LABELS)} successfully")
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Total consensus features: {total_consensus_features}")
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Average per label: {total_consensus_features / successful_labels:.1f}" if successful_labels > 0 else "")
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Total time: {total_duration}")
    if failed_labels > 0:
        failed_label_names = [label for label, result in all_results.items() if not result.get("success", False)]
        print(f"{Colors.RED}[ERROR]{Colors.RESET} Failed labels: {', '.join(failed_label_names)}")
    print(f"{'='*80}")

def process_single_label(target_outcome, config):
    """
    Process single label (original main() logic, keeping backward compatibility)
    
    Args:
        target_outcome (str): target outcome
        config (dict): configuration information
    """
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Single label causal feature generation mode - {target_outcome}")
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} {'='*60}")
    
    model_config = config["model_config"]
    experiment_count = config["experiment_count"]
    
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Model: {model_config['model_name']} ({model_config['display_name']})")
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} API type: {model_config['api_type']}")
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Temperature setting: {model_config['generation_params']['temperature']}")
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Target outcome: {target_outcome}")
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Experiment count: {experiment_count}")
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Output directory: {config['output_dir']}")
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Quality control: {'Strict mode' if config['strict_mode'] else 'Loose mode'}")
    
    # User confirmation
    print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Continue with execution? (y/n): ", end="")
    user_input = input().strip().lower()
    if user_input not in ['y', 'yes']:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} User cancelled operation")
        return
    
    # Check API connection
    print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Checking API connection...")
    test_response = query_model("Test connection", model_config)
    if "error" in test_response.lower():
        print(f"{Colors.RED}[ERROR]{Colors.RESET} API connection failed: {test_response}")
        return
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} API connection successful")
    
    # 4. Load 262 features (excluding basic features)
    print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Loading complete feature list...")
    categorized_features, all_feature_descriptions, categorized_names = load_all_available_features()
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Feature loading completed: {len(all_feature_descriptions)} features (4 categories)")
    
    # 5. Create session directory
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_uuid = str(uuid.uuid4())[:8]
    clean_model_name = model_config['model_name'].replace(" ", "_").replace("-", "_")
    
    session_dir_name = f"{clean_model_name}_{target_outcome}_{session_timestamp}_{session_uuid}"
    session_dir = os.path.join(config["output_dir"], session_dir_name)
    
    os.makedirs(session_dir, exist_ok=True)
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Session directory created: {session_dir_name}")
    
    # 6. Execute multiple experiments
    print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} {'='*60}")
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Starting to execute {experiment_count} repeated experiments")
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} {'='*60}")
    
    all_results = []
    
    for i in range(1, experiment_count + 1):
        print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Experiment {i}/{experiment_count}")
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} {'-' * 40}")
        
        # Single experiment (using new classification feature structure)
        result = generate_causal_features_single_experiment(
            model_config, categorized_features, target_outcome, config, i
        )
        
        # Save single experiment result
        save_experiment_result(
            result, session_dir, model_config['model_name'], 
            target_outcome, i, session_timestamp
        )
        
        all_results.append(result)
        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Experiment {i} completed")
    
    # 7. Save session summary
    print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Generating session summary...")
    save_session_summary(all_results, config, session_dir, target_outcome, session_timestamp)
    
    # 8. Generate consensus
    print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} {'='*60}")
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Generating consensus feature set")
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} {'='*60}")
    
    consensus_result = generate_consensus_features(all_results, config, target_outcome)
    
    if consensus_result:
        # Save consensus result
        consensus_json_path, consensus_txt_path = save_consensus_results(
            consensus_result, session_dir, model_config['model_name'], 
            target_outcome, session_timestamp
        )
        
        # Update FINAL file
        consensus_features = consensus_result[target_outcome]["consensus_feature_set"]
        update_final_consensus_file(
            config["output_dir"], model_config['model_name'], 
            target_outcome, consensus_features
        )
        
        print(f"\n{Colors.GREEN}[SUCCESS]{Colors.RESET} Consensus generation successful!")
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Consensus feature count: {len(consensus_features)}")
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} JSON file: {os.path.basename(consensus_json_path)}")
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} TXT file: {os.path.basename(consensus_txt_path)}")
        
    else:
        print(f"\n{Colors.RED}[ERROR]{Colors.RESET} Consensus generation failed!")
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Reason: quality control not passed or successful experiment count insufficient")
    
    print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} {'='*60}")
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Single label causal feature generation completed!")
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Session directory: {session_dir_name}")
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Total experiments: {experiment_count}")
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Successful experiments: {sum(1 for r in all_results if 'error' not in r)}")
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} {'='*60}")

def main():
    """
    Main function - supports single label and batch processing modes
    """
    try:
        # 1. Configuration validation
        validate_config()
        
        config = CAUSAL_GENERATION_CONFIG
        run_mode = config.get("run_mode", "single_label")
        
        if run_mode == "single_label":
            # Original single label processing logic (keeping backward compatibility)
            target_outcome = config["target_outcome"]
            process_single_label(target_outcome, config)
            
        elif run_mode == "all_labels":
            # New batch processing logic
            process_all_labels(config)
            
        else:
            raise ValueError(f"{Colors.RED}[ERROR]{Colors.RESET} Unsupported run mode: {run_mode}")
        
    except Exception as e:
        print(f"\n{Colors.RED}[ERROR]{Colors.RESET} Program execution error:")
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Error information: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 