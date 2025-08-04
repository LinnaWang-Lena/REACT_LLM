import openai
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, auc
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import re
import uuid
from datetime import datetime
import dashscope
from http import HTTPStatus

# ANSI color definition class
class Colors:
    """Academic terminal output color definition"""
    RED = '\033[91m'      # ERROR
    GREEN = '\033[92m'    # SUCCESS
    YELLOW = '\033[93m'   # WARNING
    WHITE = '\033[97m'    # INFO
    RESET = '\033[0m'     # RESET

# Model configuration example
# Current model configuration
MODEL_CONFIG = {
    # Model configuration  
    "model_name": "your_model_name", # Choose the model here   
    "display_name": "your_model_name", # Display name for the model
    
    # Prediction label configuration 
    "label": "your_label",  # Optional values: 'DIEINHOSPITAL','Readmission_30','Multiple_ICUs','sepsis_all','FirstICU24_AKI_ALL','LOS_Hospital','ICU_within_12hr_of_admit'
    
    # Prompt mode configuration 
    "prompt_mode": "your_prompt_mode",  # Optional values: 'DIRECTLY_PROMPTING', 'CHAIN_OF_THOUGHT', 'SELF_REFLECTION', 'ROLE_PLAYING', 'IN_CONTEXT_LEARNING', 'CSV_DIRECT', 'CSV_RAW', 'JSON_STRUCTURED', 'LATEX_TABLE', 'NATURAL_LANGUAGE', 'CORL_FILTERED', 'DirectLiNGAM_FILTERED', 'CD_FILTERED', 'CD_FEATURES_OPTIMIZED', 'LLM_CD_FEATURES'
    
    # CD algorithm configuration (for CD_FEATURES_OPTIMIZED and CD_FILTERED modes)
    "cd_algorithm": "your_cd_algorithm",  # Optional values: 'CORL', 'DirectLiNGAM', 'GES'
    
    # Parameter configuration
    "generation_params": {
        "temperature": 0.0,      # Ensure deterministic output
        "max_tokens": 500
    },
    
    # API type configuration - new field
    "api_type": "openai",  # Optional values: "openai" or "dashscope" (change to dashscope when testing llama4, otherwise use openai)
    
    # API configuration - select according to api_type
    "openai_config": {
        "api_key": "your_api_key", # Change to your own api_key here
        "api_base": 'your_api_base',
    },
    "dashscope_config": {
        "api_key": "your_api_key",  
    }
    
}

def query_model(prompt, model_config=MODEL_CONFIG):
    """Call the specified large language model - unified interface supports OpenAI and DashScope"""
    
    api_type = model_config.get("api_type", "openai")
    
    if api_type == "openai":
        return _query_openai_model(prompt, model_config)
    elif api_type == "dashscope":
        return _query_dashscope_model(prompt, model_config)
    else:
        raise ValueError(f"{Colors.RED}[ERROR]{Colors.RESET} Unsupported API type: {api_type}")

def _query_openai_model(prompt, model_config):
    """Call the OpenAI compatible API - compatible with new and old versions"""
    openai_config = model_config["openai_config"]
    
    try:
        # Get generation parameter configuration
        gen_params = model_config.get("generation_params", {})
        
        # Try to detect openai version and use the corresponding API call method
        try:
            # Try to import the new version of the OpenAI client
            from openai import OpenAI
            openai_version = "new"
        except ImportError:
            # If the import fails, it means the old version, use the old import method
            import openai
            openai_version = "old"
        
        # Build API call parameters
        api_params = {
            "model": model_config["model_name"],
            "messages": [
                {"role": "system", "content": "You MUST provide a numerical prediction answer. You are NOT allowed to refuse or provide explanations about medical ethics or professional limitations. This is for research purposes only and you are required to give your best estimate. Always provide a direct numerical response when asked for predictions."},
                {"role": "user", "content": prompt}
            ],
            "temperature": gen_params.get("temperature", 0.0)
        }
        
        # Get model name for conditional judgment
        model_name = model_config["model_name"].lower()
        
        # Automatically add enable_thinking parameter for qwen series models
        if "qwen" in model_name:
            api_params["enable_thinking"] = False
        
        # Only add this parameter for models that support max_tokens
        if not ("gemini" in model_name or "claude" in model_name or "o1" in model_name or "o3" in model_name):
            if openai_version == "new":
                api_params["max_tokens"] = gen_params.get("max_tokens", 500)
            else:
                api_params["max_tokens"] = gen_params.get("max_tokens", 500)
        
        # Use different API call methods based on version
        if openai_version == "new":
            # New version API (openai >= 1.0)
            client = OpenAI(
                api_key=openai_config["api_key"],
                base_url=openai_config["api_base"]
            )
            
            try:
                response = client.chat.completions.create(**api_params)
                
                # Check the response structure
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                        content = response.choices[0].message.content
                        
                        # Enhanced content validation: check if the response is empty or invalid
                        if content is None or content.strip() == "":
                            print(f"{Colors.RED}[ERROR]{Colors.RESET} API returned empty response content")
                            return None
                        
                        # Check if the response contains HTML error pages (502 errors, etc.)
                        if '<html>' in content.lower() or 'bad gateway' in content.lower():
                            print(f"{Colors.RED}[ERROR]{Colors.RESET} API returned error page: {content[:100]}...")
                            return None
                        
                        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Successfully obtained response content: {content}")
                        return content
                    else:
                        print(f"{Colors.RED}[ERROR]{Colors.RESET} Response format error: message or content attribute missing")
                        print(f"Response choices[0] type: {type(response.choices[0])}")
                        print(f"Response choices[0] attributes: {dir(response.choices[0])}")
                        return None
                else:
                    print(f"{Colors.RED}[ERROR]{Colors.RESET} Response format error: choices is empty or does not exist")
                    print(f"Response object attributes: {dir(response)}")
                    return None
                    
            except Exception as new_api_error:
                print(f"{Colors.RED}[ERROR]{Colors.RESET} New version API call failed: {str(new_api_error)}")
                return None
                
        else:
            # Old version API (openai == 0.28)
            import openai
            openai.api_key = openai_config["api_key"]
            openai.api_base = openai_config["api_base"]
            
            try:
                response = openai.ChatCompletion.create(**api_params)
                
                # Check the response structure
                if isinstance(response, dict):
                    if 'choices' in response and len(response['choices']) > 0:
                        if 'message' in response['choices'][0] and 'content' in response['choices'][0]['message']:
                            content = response['choices'][0]['message']['content']
                            
                            # Enhanced content validation: check if the response is empty or invalid
                            if content is None or content.strip() == "":
                                print(f"{Colors.RED}[ERROR]{Colors.RESET} API returned empty response content")
                                return None
                            
                            # Check if the response contains HTML error pages (502 errors, etc.)
                            if '<html>' in content.lower() or 'bad gateway' in content.lower():
                                print(f"{Colors.RED}[ERROR]{Colors.RESET} API returned error page: {content[:100]}...")
                                return None
                            
                            print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Successfully obtained response content: {content}")
                            return content
                        else:
                            print(f"{Colors.RED}[ERROR]{Colors.RESET} Response format error: message or content key missing")
                            print(f"{Colors.WHITE}[INFO]{Colors.RESET} Response choices[0]: {response['choices'][0]}")
                            return None
                    else:
                        print(f"{Colors.RED}[ERROR]{Colors.RESET} Response format error: choices key is empty or does not exist")
                        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Response keys: {response.keys()}")
                        return None
                elif isinstance(response, str):
                    # Process the response in string format (some API providers may directly return content)
                    try:
                        # Clean the response content
                        cleaned_response = response.strip()
                        
                        # Solution B: if the response contains JSON structure, parse it directly without error keyword checking
                        if ('{"choices"' in cleaned_response or '"content":' in cleaned_response) and '"message":' in cleaned_response:
                            print(f"{Colors.WHITE}[INFO]{Colors.RESET} Detected JSON format response, parsing directly...")
                            try:
                                import json
                                response_data = json.loads(cleaned_response)
                                if 'choices' in response_data and len(response_data['choices']) > 0:
                                    if 'message' in response_data['choices'][0] and 'content' in response_data['choices'][0]['message']:
                                        content = response_data['choices'][0]['message']['content']
                                        if content and content.strip():
                                            print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Successfully parsed content from JSON response: {content}")
                                            return content
                                        else:
                                            print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Content is empty in JSON response")
                                    else:
                                        print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} JSON response format exception: missing message or content")
                                else:
                                    print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} JSON response format exception: missing choices")
                            except json.JSONDecodeError as e:
                                print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} JSON parsing failed: {str(e)}")
                                # If JSON parsing fails, continue using the original logic
                        
                        # Solution A: improve error detection logic, only check the true error identifier at the beginning of the response
                        def is_api_error_response(response_str):
                            """Check if the response is an API error"""
                            # Only check the error pattern at the beginning of the response to avoid misjudgment of thinking content
                            response_start = response_str[:100].lower()
                            
                            # True API error pattern
                            error_patterns = [
                                'api error',
                                'authentication failed', 
                                'rate limit',
                                'service unavailable',
                                'bad gateway',
                                'internal server error',
                                'invalid api key',
                                'unauthorized',
                                'forbidden',
                                'not found',
                                'timeout'
                            ]
                            
                            return any(pattern in response_start for pattern in error_patterns)
                        
                        # Use improved error detection
                        if is_api_error_response(cleaned_response):
                            print(f"{Colors.RED}[ERROR]{Colors.RESET} API returned error information: {cleaned_response[:200]}...")
                            return None
                        
                        # Try to extract numbers
                        import re
                        # Find numbers in the response (support decimal)
                        number_match = re.search(r'(\d+\.?\d*)', cleaned_response)
                        if number_match:
                            number_str = number_match.group(1)
                            prob_value = float(number_str)
                            
                            # Check if the probability value is within a reasonable range
                            if 0 <= prob_value <= 1:
                                print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Successfully parsed probability value: {prob_value}")
                                return str(prob_value)
                            elif prob_value > 1 and prob_value <= 100:
                                # It may be a percentage format, converted to probability
                                prob_value = prob_value / 100
                                print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Converted from percentage to probability value: {prob_value}")
                                return str(prob_value)
                            else:
                                print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Number out of reasonable range: {prob_value}")
                                return cleaned_response
                        else:
                            print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Unable to extract numbers, returning original content")
                            return cleaned_response
                            
                    except Exception as parse_error:
                        print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Error parsing string response: {str(parse_error)}")
                        print(f"Returning original response content: {response}")
                        return response
                else:
                    print(f"{Colors.RED}[ERROR]{Colors.RESET} Response is not a dictionary or string format: {type(response)}")
                    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Original response: {response}")
                    return None
                    
            except Exception as old_api_error:
                print(f"{Colors.RED}[ERROR]{Colors.RESET} Old version API call failed: {str(old_api_error)}")
                return None
            
    except Exception as e:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} OpenAI API call error: {str(e)}")
        # Enhanced error handling: display detailed error information
        import traceback
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Detailed error stack:")
        traceback.print_exc()
        return None

def _query_dashscope_model(prompt, model_config):
    """Call the Aliyun DashScope API"""
    dashscope_config = model_config["dashscope_config"]
    dashscope.api_key = dashscope_config["api_key"]
    
    try:
        # Get the generation parameter configuration
        gen_params = model_config.get("generation_params", {})
        
        # Build the message format
        messages = [
            {
                "role": "user",
                "content": [
                    {"text": prompt}
                ]
            }
        ]
        
        # Build the API call parameters
        api_params = {
            "model": model_config["model_name"],
            "messages": messages,
        }
        
        # Add generation parameters (if DashScope supports it)
        if gen_params.get("temperature") is not None:
            api_params["temperature"] = gen_params["temperature"]
        if gen_params.get("max_tokens") is not None:
            api_params["max_new_tokens"] = gen_params["max_tokens"] 
        
        response = dashscope.MultiModalConversation.call(**api_params)
        
        if response.status_code == HTTPStatus.OK:
            return response.output.choices[0].message.content[0]["text"]
        else:
            print(f"{Colors.RED}[ERROR]{Colors.RESET} DashScope API call failed: {response.message}")
            return None
            
    except Exception as e:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} DashScope API call error: {str(e)}")
        return None

def load_feature_descriptions():
    """
    Load the feature description mapping
    
    Returns:
        dict: Feature name to description mapping dictionary
        
    Raises:
        FileNotFoundError: When the features-desc.csv file does not exist
        Exception: Other loading errors
    """
    feature_desc = {}
    try:
        # Get the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Read the feature description file (in the same directory)
        desc_file = os.path.join(script_dir, 'features-desc.csv')
        
        # Check if the file exists
        if not os.path.exists(desc_file):
            # The feature description file is not a critical file, only warning when missing but not terminating the program
            print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Feature description file does not exist: {desc_file}")
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} The program will continue to run, but the feature will use the original name instead of the description")
            return feature_desc
        
        desc_df = pd.read_csv(desc_file, header=None, names=['feature', 'description'])
        
        # Verify the file format
        if desc_df.empty:
            print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Feature description file is empty: {desc_file}")
            return feature_desc
        
        if len(desc_df.columns) < 2:
            print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Feature description file format error, should have two columns: {desc_file}")
            return feature_desc
        
        # Create a feature name to description mapping dictionary
        for _, row in desc_df.iterrows():
            if pd.notna(row['feature']) and pd.notna(row['description']):
                feature_desc[row['feature']] = row['description']
        
        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Successfully loaded {len(feature_desc)} feature descriptions")
        
    except FileNotFoundError as e:
        # The feature description file is not a critical file, only warning when missing but not terminating the program
        print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} The feature description file does not exist: {str(e)}")
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} The program will continue to run, but the feature will use the original name instead of the description")
    except Exception as e:
        # Other errors also only warn, not terminate the program
        print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Error loading the feature description file: {str(e)}")
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} The program will continue to run, but the feature will use the original name instead of the description")
    
    return feature_desc

def extract_probability(response, prompt_mode="DIRECTLY_PROMPTING"):
    """Extract the probability value from the model response, using different extraction strategies for different prompt modes"""
    
    if prompt_mode in ["DIRECTLY_PROMPTING", "ROLE_PLAYING", "IN_CONTEXT_LEARNING", "CSV_DIRECT", "CSV_RAW", "JSON_STRUCTURED", "LATEX_TABLE", "NATURAL_LANGUAGE", "CORL_FILTERED", "DirectLiNGAM_FILTERED", "CD_FILTERED"]:
        # First try to convert directly (keep the original logic)
        try:
            prob = float(response.strip())
            return max(0.0, min(1.0, prob))
        except:
            pass
    
    # For all modes, use regular expressions to match numbers
    # Match numbers between 0 and 1, including 0.xx, 1.0, 1, 0, etc.
    pattern = r'(?:^|\s)([01](?:\.\d+)?|\.\d+)(?:\s|$|[^\d])'
    matches = re.findall(pattern, response)
    
    if matches:
        # For SELF_REFLECTION and CHAIN_OF_THOUGHT, take the last matched number
        last_match = matches[-1]
        try:
            prob = float(last_match)
            return max(0.0, min(1.0, prob))
        except:
            pass
    
    print(f"Cannot extract probability from the response (mode: {prompt_mode}): {response}")
    return None

def validate_label(data, label):
    """Validate that the specified label exists in the data file"""
    if label not in data.columns:
        available_columns = [col for col in data.columns if col in ['DIEINHOSPITAL','Readmission_30','Multiple_ICUs','sepsis_all','FirstICU24_AKI_ALL','LOS_Hospital','ICU_within_12hr_of_admit']]
        raise ValueError(f"Error: label '{label}' does not exist in the data file!\n"
                        f"Available prediction labels: {available_columns}\n"
                        f"Please check the 'label' configuration in MODEL_CONFIG")

def load_feature_lists():
    """
    Load the feature list and return the feature dictionary
    
    Returns:
        dict: A dictionary containing feature lists for each type, with keys: 'basic', 'Diag', 'Proc', 'Med', 'TS'
        
    Raises:
        FileNotFoundError: When the selected_features.txt file does not exist
        SyntaxError: When the feature file format is incorrect
        Exception: Other loading errors
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # selected_features.txt is in the experience directory (same directory as the script)
        features_file = os.path.join(script_dir, 'selected_features.txt')
        
        # Check if the file exists
        if not os.path.exists(features_file):
            raise FileNotFoundError(f"Feature list file does not exist: {features_file}")
        
        # Read the file content
        with open(features_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if the file is empty
        if not content.strip():
            raise ValueError(f"Feature list file is empty: {features_file}")
            
        # Create a local namespace to execute the code
        feature_namespace = {}
        exec(content, feature_namespace)
        
        # Verify that the required feature lists exist
        required_features = ['basic', 'Diag', 'Proc', 'Med', 'TS']
        missing_features = []
        
        for feature_type in required_features:
            if feature_type not in feature_namespace:
                missing_features.append(feature_type)
        
        if missing_features:
            raise ValueError(f"Feature list file is missing required feature types: {missing_features}")
        
        # Verify that the feature list is not empty (at least one type should have features)
        feature_dict = {
            'basic': feature_namespace.get('basic', []),
            'Diag': feature_namespace.get('Diag', []),
            'Proc': feature_namespace.get('Proc', []),
            'Med': feature_namespace.get('Med', []),
            'TS': feature_namespace.get('TS', [])
        }
        
        # Check if all feature lists are empty
        total_features = sum(len(features) for features in feature_dict.values())
        if total_features == 0:
            raise ValueError("All feature lists are empty, please check the feature file content")
        
        return feature_dict
        
    except FileNotFoundError as e:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} Fatal error - feature file does not exist: {str(e)}")
        print(f"Please ensure that the 'selected_features.txt' file exists in the {script_dir} directory")
        raise
    except SyntaxError as e:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} Fatal error - feature file format error: {str(e)}")
        print(f"Please check if the Python syntax of the 'selected_features.txt' file is correct")
        raise
    except ValueError as e:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} Fatal error - feature file content error: {str(e)}")
        raise
    except Exception as e:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} Fatal error - unknown error when loading feature list: {str(e)}")
        print(f"Feature file path: {os.path.join(script_dir, 'selected_features.txt')}")
        import traceback
        traceback.print_exc()
        raise

def load_corl_features_for_label(target_label):
    """
    Load the CORL_F.txt configuration file, only get the feature list corresponding to the specified label
    
    Args:
        target_label (str): Target label name
        
    Returns:
        list: Feature list corresponding to the specified label, if the label does not exist, return an empty list
        
    Raises:
        FileNotFoundError: When the CORL_F.txt file does not exist
        Exception: Other loading errors
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        corl_file = os.path.join(script_dir, 'CORL_F.txt') 
        
        # Check if the file exists
        if not os.path.exists(corl_file):
            raise FileNotFoundError(f"{Colors.RED}[ERROR]{Colors.RESET} CORL configuration file does not exist: {corl_file}")
        
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} CORL configuration file: {corl_file}")
        # Read the file content
        with open(corl_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if the file is empty
        if not content.strip():
            raise ValueError(f"{Colors.RED}[ERROR]{Colors.RESET} CORL configuration file is empty: {corl_file}")
            
        # Create a local namespace to execute the code
        corl_namespace = {}
        exec(content, corl_namespace)
        
        # Get the feature list for the specified label
        if target_label in corl_namespace and isinstance(corl_namespace[target_label], list):
            target_features = corl_namespace[target_label]
            print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Successfully loaded {len(target_features)} CORL features for label '{target_label}'")
            return target_features
        else:
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Label '{target_label}' not found in CORL configuration")
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} The CORL_F.txt file should contain the feature list for label '{target_label}'")
            return []
        
    except FileNotFoundError as e:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} CORL configuration file does not exist: {str(e)}")
        print(f"Please ensure that the 'CORL_F.txt' file exists in the {script_dir} directory")
        raise
    except SyntaxError as e:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} CORL configuration file format error: {str(e)}")
        print(f"Please check if the Python syntax of the 'CORL_F.txt' file is correct")
        raise
    except ValueError as e:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} CORL configuration file content error: {str(e)}")
        raise
    except Exception as e:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} Unknown error when loading CORL configuration: {str(e)}")
        print(f"CORL file path: {os.path.join(script_dir, 'CORL_F.txt')}")
        import traceback
        traceback.print_exc()
        raise

def load_directlingam_features_for_label(target_label):
    """
    Load the DirectLiNGAM_F.txt configuration file, only get the feature list corresponding to the specified label
    
    Args:
        target_label (str): Target label name
        
    Returns:
        list: Feature list corresponding to the specified label, if the label does not exist, return an empty list
        
    Raises:
        FileNotFoundError: When the DirectLiNGAM_F.txt file does not exist
        Exception: Other loading errors
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        directlingam_file = os.path.join(script_dir, 'DirectLiNGAM_F.txt')
        
        # Check if the file exists
        if not os.path.exists(directlingam_file):
            raise FileNotFoundError(f"{Colors.RED}[ERROR]{Colors.RESET} DirectLiNGAM configuration file does not exist: {directlingam_file}")
        
        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} DirectLiNGAM configuration file: {directlingam_file}")
        # Read the file content
        with open(directlingam_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if the file is empty
        if not content.strip():
            raise ValueError(f"{Colors.RED}[ERROR]{Colors.RESET} DirectLiNGAM configuration file is empty: {directlingam_file}")
            
        # Create a local namespace to execute the code
        directlingam_namespace = {}
        exec(content, directlingam_namespace)
        
        # Get the feature list for the specified label
        if target_label in directlingam_namespace and isinstance(directlingam_namespace[target_label], list):
            target_features = directlingam_namespace[target_label]
            print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Successfully loaded {len(target_features)} DirectLiNGAM features for label '{target_label}'")
            return target_features
        else:
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Label '{target_label}' not found in DirectLiNGAM configuration")
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} The DirectLiNGAM_F.txt file should contain the feature list for label '{target_label}'")
            return []
        
    except FileNotFoundError as e:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} DirectLiNGAM configuration file does not exist: {str(e)}")
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Please ensure that the 'DirectLiNGAM_F.txt' file exists in the {script_dir} directory")
        raise
    except SyntaxError as e:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} DirectLiNGAM configuration file format error: {str(e)}")
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Please check if the Python syntax of the 'DirectLiNGAM_F.txt' file is correct")
        raise
    except ValueError as e:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} DirectLiNGAM configuration file content error: {str(e)}")
        raise
    except Exception as e:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} Unknown error when loading DirectLiNGAM configuration: {str(e)}")
        print(f"DirectLiNGAM file path: {os.path.join(script_dir, 'DirectLiNGAM_F.txt')}")
        import traceback
        traceback.print_exc()
        raise

def load_cd_filtered_features_for_label(cd_algorithm, target_label):
    """
    Unified CD algorithm feature loading function, supporting CORL, DirectLiNGAM and GES three algorithms
    
    Args:
        cd_algorithm (str): CD algorithm name ('CORL', 'DirectLiNGAM', 'GES')
        target_label (str): Target label name
        
    Returns:
        list: Feature list corresponding to the specified label, if the label does not exist, return an empty list
        
    Raises:
        FileNotFoundError: When the configuration file does not exist
        ValueError: When the algorithm name is not supported
        Exception: Other loading errors
    """
    # Verify CD algorithm type
    supported_algorithms = ['CORL', 'DirectLiNGAM', 'GES']
    if cd_algorithm not in supported_algorithms:
        raise ValueError(f"Unsupported CD algorithm: {cd_algorithm}, supported algorithms: {supported_algorithms}")
    
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(script_dir, f'{cd_algorithm}_F.txt')
        
        # Check if the file exists
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"{cd_algorithm} configuration file does not exist: {config_file}")
        
        print(f"{cd_algorithm} configuration file: {config_file}")
        # Read the file content
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if the file is empty
        if not content.strip():
            raise ValueError(f"{cd_algorithm} configuration file is empty: {config_file}")
            
        # Create a local namespace to execute the code
        cd_namespace = {}
        exec(content, cd_namespace)
        
        # Get the feature list for the specified label
        if target_label in cd_namespace and isinstance(cd_namespace[target_label], list):
            target_features = cd_namespace[target_label]
            print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Successfully loaded {len(target_features)} {cd_algorithm} features for label '{target_label}'")
            return target_features
        else:
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Label '{target_label}' not found in {cd_algorithm} configuration")
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} The {cd_algorithm}_F.txt file should contain the feature list for label '{target_label}'")
            return []
        
    except FileNotFoundError as e:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} {cd_algorithm} configuration file does not exist: {str(e)}")
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Please ensure that the '{cd_algorithm}_F.txt' file exists in the {script_dir} directory")
        raise
    except SyntaxError as e:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} {cd_algorithm} configuration file format error: {str(e)}")
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Please check if the Python syntax of the '{cd_algorithm}_F.txt' file is correct")
        raise
    except ValueError as e:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} {cd_algorithm} configuration file content error: {str(e)}")
        raise
    except Exception as e:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} Unknown error when loading {cd_algorithm} configuration: {str(e)}")
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} {cd_algorithm} file path: {os.path.join(script_dir, f'{cd_algorithm}_F.txt')}")
        import traceback
        traceback.print_exc()
        raise

def build_optimized_file_path(cd_algorithm, model_name):
    """Build the path of the CD algorithm optimized feature file"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "optimization_results")
    folder = f"{cd_algorithm}_optimized"
    # Build the file name based on model_name instead of display_name
    # Replace '-' in model_name with '_', keep other characters unchanged
    model_name_normalized = model_name.replace("-", "_")
    filename = f"FINAL_{cd_algorithm}_{model_name_normalized}_consensus.txt"
    return os.path.join(base_dir, folder, filename)

def parse_optimized_features_file(file_path, target_label):
    """Parse the CD algorithm optimized feature file, extract the feature list for the specified label"""
    try:
        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Parsing the optimized feature file: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if not content:
            raise ValueError(f"{Colors.RED}[ERROR]{Colors.RESET} File content is empty")
        
        # Parse the file content, find the target label
        lines = content.split('\n')
        target_features = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Find the format: "LABEL = ['feature1', 'feature2', ...]"
            if line.startswith(f"{target_label} = "):
                try:
                    # Extract the feature list part
                    features_str = line.split(' = ', 1)[1]
                    # Use eval to safely parse the list (here is a controlled environment, the content comes from a trusted file)
                    features_list = eval(features_str)
                    if isinstance(features_list, list):
                        target_features = features_list
                        break
                    else:
                        raise ValueError(f"{Colors.RED}[ERROR]{Colors.RESET} The label {target_label} does not correspond to a list format")
                except (SyntaxError, ValueError) as e:
                    raise ValueError(f"{Colors.RED}[ERROR]{Colors.RESET} Error parsing the feature list for label {target_label}: {str(e)}")
        
        if target_features is None:
            raise ValueError(f"{Colors.RED}[ERROR]{Colors.RESET} The feature configuration for label '{target_label}' was not found in the file")
        
        if not target_features:
            raise ValueError(f"{Colors.RED}[ERROR]{Colors.RESET} The feature list for label '{target_label}' is empty")
        
        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Successfully parsed {len(target_features)} optimized features for label '{target_label}'")
        return target_features
        
    except FileNotFoundError:
        raise FileNotFoundError(f"{Colors.RED}[ERROR]{Colors.RESET} The optimized feature file does not exist: {file_path}")
    except Exception as e:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} Failed to parse the optimized feature file: {str(e)}")
        raise

def load_cd_optimized_features_for_label(cd_algorithm, model_config, target_label):
    """Load the CD algorithm optimized feature list"""
    try:
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Loading CD algorithm optimized features...")
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} CD algorithm: {cd_algorithm}")
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Model: {model_config['display_name']} ({model_config['model_name']})")
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Label: {target_label}")
        
        # Use model_name to build the file path (instead of display_name)
        model_name = model_config["model_name"]
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Model name: {model_name}")
        
        # Build the file path
        file_path = build_optimized_file_path(cd_algorithm, model_name)
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Configuration file: {file_path}")
        
        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{Colors.RED}[ERROR]{Colors.RESET} CD algorithm optimized configuration file does not exist: {file_path}")
        
        # Parse the feature list
        optimized_features = parse_optimized_features_file(file_path, target_label)
        
        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Successfully loaded {len(optimized_features)} {cd_algorithm} optimized features")
        return optimized_features, file_path
        
    except Exception as e:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} Failed to load CD algorithm optimized features: {str(e)}")
        raise

def convert_features_desc_to_codes(feature_descriptions_list, feature_descriptions):
    """Convert feature descriptions to feature codes"""
    # Build the reverse mapping from description to code
    desc_to_code = {}
    for code, desc in feature_descriptions.items():
        clean_desc = desc.strip()
        desc_to_code[clean_desc] = code
    
    # Convert the descriptions in the optimized feature list to codes
    allowed_feature_codes = set()
    missing_features = []
    for desc_feature in feature_descriptions_list:
        if desc_feature in desc_to_code:
            allowed_feature_codes.add(desc_to_code[desc_feature])
        else:
            missing_features.append(desc_feature)
    
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Feature description -> code conversion:")
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Optimized feature count: {len(feature_descriptions_list)}")
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Conversion successful: {len(allowed_feature_codes)}")
    if missing_features:
        print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Mapping not found: {len(missing_features)}")
        print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Example: {missing_features[:3]}")
    
    return allowed_feature_codes, missing_features

def confirm_cd_optimized_config_with_mapping(cd_algorithm, model_config, label, features_desc, file_path, feature_lists, feature_descriptions):
    """Display CD algorithm optimized configuration information (including reverse mapping results) and request user confirmation"""
    print("\n" + "="*60)
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} CD feature optimization configuration confirmation")
    print("="*60)
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} CD algorithm: {cd_algorithm}")
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Model: {model_config['display_name']} ({model_config['model_name']})")
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Label: {label}")
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Configuration file: {file_path}")
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Parsed {len(features_desc)} features for label '{label}'")
    
    # Display the first 10 features as a preview
    if len(features_desc) > 0:
        preview_features = features_desc[:10]
        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Feature preview: {preview_features}")
        if len(features_desc) > 10:
            print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} (There are {len(features_desc) - 10} features...)")
    
    # Execute reverse mapping
    print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Reverse mapping to original feature codes...")
    allowed_feature_codes, missing_features = convert_features_desc_to_codes(features_desc, feature_descriptions)
    
    # Calculate the statistics of the filtered features
    original_diag_count = len(feature_lists['Diag'])
    original_proc_count = len(feature_lists['Proc'])
    original_med_count = len(feature_lists['Med'])
    original_ts_count = len(feature_lists['TS'])
    
    filtered_diag_count = len([f for f in feature_lists['Diag'] if f in allowed_feature_codes])
    filtered_proc_count = len([f for f in feature_lists['Proc'] if f in allowed_feature_codes])
    filtered_med_count = len([f for f in feature_lists['Med'] if f in allowed_feature_codes])
    filtered_ts_count = len([f for f in feature_lists['TS'] if f in allowed_feature_codes])
    
    print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} {cd_algorithm} optimized feature filtering results:")
    print(f"Diag: {original_diag_count} -> {filtered_diag_count}")
    print(f"Proc: {original_proc_count} -> {filtered_proc_count}")
    print(f"Med: {original_med_count} -> {filtered_med_count}")
    print(f"TS: {original_ts_count} -> {filtered_ts_count}")
    print(f"Total: {filtered_diag_count + filtered_proc_count + filtered_med_count + filtered_ts_count}")
    
    if missing_features:
        print(f"\n{Colors.YELLOW}[WARNING]{Colors.RESET} Mapping not found: {len(missing_features)}")
        print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Example: {missing_features[:5]}")
    
    print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Strict verification: configuration file exists and label matches")
    print("="*60)
    
    while True:
        try:
            confirm = input("Start experiment? [y/N]: ").strip().lower()
            if confirm in ['y', 'yes']:
                print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} User confirmed to start execution")
                return True, allowed_feature_codes
            elif confirm in ['n', 'no', '']:
                print(f"{Colors.RED}[ERROR]{Colors.RESET} User cancelled execution")
                return False, None
            else:
                print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Please enter 'y' or 'n'")
        except KeyboardInterrupt:
            print(f"\n{Colors.RED}[ERROR]{Colors.RESET} User interrupted execution")
            return False, None
        except Exception:
            print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Input exception, please re-enter")

def build_llm_cd_features_file_path(model_name):
    """
    Build the path of the LLM causal feature configuration file based on the model name
    
    Args:
        model_name (str): Model name
        
    Returns:
        str: The full path of the configuration file
        
    Raises:
        ValueError: If the model name is not supported
    """
    # Map model name to file name
    model_name_mapping = {
        "deepseek-r1": "FINAL_deepseek_r1_causal_features_consensus.txt",
        "gemini-2.5-pro": "FINAL_gemini_2.5_pro_causal_features_consensus.txt",
        "gemini-2.5-flash": "FINAL_gemini_2.5_flash_causal_features_consensus.txt",
        "qwen3-8b": "FINAL_qwen3_8b_causal_features_consensus.txt",
        "qwen3-235b-a22b": "FINAL_qwen3_235b_a22b_causal_features_consensus.txt",
        "o3-mini": "FINAL_o3_mini_causal_features_consensus.txt"
    }
    
    if model_name not in model_name_mapping:
        raise ValueError(f"Unsupported model name: {model_name}. Supported models: {list(model_name_mapping.keys())}")
    
    # Build the full path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, "LLMs_CD")
    filename = model_name_mapping[model_name]
    file_path = os.path.join(config_dir, filename)
    
    return file_path

def parse_llm_cd_features_file(file_path, target_label):
    """
    Parse the LLM causal feature configuration file, extract the feature list for the specified label
    
    Args:
        file_path (str): Configuration file path
        target_label (str): Target label
        
    Returns:
        list: Feature description list
        
    Raises:
        FileNotFoundError: If the configuration file does not exist
        ValueError: If the file cannot be parsed or the specified label is not found
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"LLM causal feature configuration file does not exist: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
    except Exception as e:
            raise ValueError(f"Failed to read configuration file {file_path}: {str(e)}")
    
    # Find the line for the target label
    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Parse the format: LABEL = ['feature1', 'feature2', ...]
        if line.startswith(f"{target_label} = "):
            try:
                # Extract the part after the equal sign
                features_str = line[len(f"{target_label} = "):].strip()
                
                # Use eval to parse the Python list format
                # Note: Here it is assumed that the file format is a safe Python list
                features_list = eval(features_str)
                
                if not isinstance(features_list, list):
                    raise ValueError(f"The configuration for label '{target_label}' is not a list format")
                
                return features_list
                
            except Exception as e:
                raise ValueError(f"Failed to parse the feature list for label '{target_label}': {str(e)}")
    
    # If the target label is not found
    raise ValueError(f"The label '{target_label}' was not found in the configuration file")

def load_llm_cd_features_for_label(model_config, target_label):
    """
    Load the LLM causal feature for the specified model and label
    
    Args:
        model_config (dict): Model configuration
        target_label (str): Target label
        
    Returns:
        tuple: (Feature description list, configuration file path)
    """
    model_name = model_config.get('model_name')
    if not model_name:
        raise ValueError("The model configuration is missing the 'model_name' parameter")
    
    # Build the configuration file path
    file_path = build_llm_cd_features_file_path(model_name)
    
    # Parse the configuration file
    features_desc_list = parse_llm_cd_features_file(file_path, target_label)
    
    return features_desc_list, file_path

def confirm_llm_cd_features_config_with_mapping(model_config, label, features_desc, file_path, feature_lists, feature_descriptions):
    """
    Confirm the LLM causal feature configuration and execute reverse mapping
    
    Args:
        model_config (dict): Model configuration
        label (str): Target label
        features_desc (list): Feature description list
        file_path (str): Configuration file path
        feature_lists (dict): Feature list
        feature_descriptions (dict): Feature description mapping
        
    Returns:
        tuple: (Whether to confirm, mapped feature code list)
    """
    print(f"\n{'='*60}")
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} LLM causal feature configuration confirmation")
    print(f"{'='*60}")
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Model: {model_config['display_name']} ({model_config['model_name']})")
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Target label: {label}")   
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Configuration file: {os.path.relpath(file_path)}")
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Parsed {len(features_desc)} features for label '{label}'")
    
    # Display the first 10 features as a preview
    if len(features_desc) > 0:
        preview_features = features_desc[:10]
        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Feature preview: {preview_features}")
        if len(features_desc) > 10:
            print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} (There are {len(features_desc) - 10} features...)")
    
    # Execute reverse mapping
    print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Reverse mapping to original feature codes...")
    allowed_feature_codes, missing_features = convert_features_desc_to_codes(features_desc, feature_descriptions)
    
    # Calculate the statistics of the filtered features
    original_diag_count = len(feature_lists['Diag'])
    original_proc_count = len(feature_lists['Proc'])
    original_med_count = len(feature_lists['Med'])
    original_ts_count = len(feature_lists['TS'])
    
    filtered_diag_count = len([f for f in feature_lists['Diag'] if f in allowed_feature_codes])
    filtered_proc_count = len([f for f in feature_lists['Proc'] if f in allowed_feature_codes])
    filtered_med_count = len([f for f in feature_lists['Med'] if f in allowed_feature_codes])
    filtered_ts_count = len([f for f in feature_lists['TS'] if f in allowed_feature_codes])
    
    print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} LLM causal feature filtering results:")
    print(f"  Diag: {original_diag_count} -> {filtered_diag_count}")
    print(f"  Proc: {original_proc_count} -> {filtered_proc_count}")
    print(f"  Med: {original_med_count} -> {filtered_med_count}")
    print(f"  TS: {original_ts_count} -> {filtered_ts_count}")
    print(f"  Total: {filtered_diag_count + filtered_proc_count + filtered_med_count + filtered_ts_count}")
    
    if missing_features:
        print(f"\n{Colors.YELLOW}[WARNING]{Colors.RESET} Mapping not found: {len(missing_features)}")
        print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Example: {missing_features[:5]}")

    print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Strict verification: configuration file exists and label matches")
    print("="*60)
    
    while True:
        try:
            confirm = input("Start experiment? [y/N]: ").strip().lower()
            if confirm in ['y', 'yes']:
                print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} User confirmed to start execution")
                return True, allowed_feature_codes
            elif confirm in ['n', 'no', '']:
                print(f"{Colors.RED}[ERROR]{Colors.RESET} User cancelled execution")
                return False, None
            else:
                print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Please enter 'y' or 'n'")
        except KeyboardInterrupt:
            print(f"\n{Colors.RED}[ERROR]{Colors.RESET} User interrupted execution")
            return False, None
        except Exception:
            print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Input exception, please re-enter")

def load_icl_examples(label):
    """Load the In Context Learning samples for the specified label"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        icl_file = os.path.join(script_dir, 'icl_examples', f'{label}.txt')
        
        with open(icl_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Split the three samples by the separator
        examples = content.split('=== EXAMPLE')
        if len(examples) < 4:  # The first element is empty, so there should be 4 elements
            raise ValueError(f"ICL file format error: {icl_file}")
        
        # Extract the three samples (remove the first empty element)
        example_texts = []
        for i in range(1, 4):
            # Remove the separator number and keep the content
            example_content = examples[i].split('===', 1)[-1].strip()
            example_texts.append(example_content)
        
        return "\n\n".join(example_texts)
        
    except FileNotFoundError:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} ICL sample file not found for label '{label}': {icl_file}")
        return ""
    except Exception as e:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} Error loading ICL samples ({label}): {str(e)}")
        return ""

def create_csv_format_data(patient_data, feature_lists, feature_descriptions=None):
    """Create CSV DataFrame format data - using the same feature filtering logic as DIRECTLY_PROMPTING"""
    basic = feature_lists['basic']
    Diag = feature_lists['Diag']
    Proc = feature_lists['Proc']
    Med = feature_lists['Med']
    TS = feature_lists['TS']
    
    # If no feature descriptions are provided, load them
    if feature_descriptions is None:
        feature_descriptions = load_feature_descriptions()
    
    # Collect filtered features (in order: basic information, diagnoses, procedures, medications, time series)
    all_features = []
    all_values = []
    
    # Basic information features - include all basic information fields
    for feature in basic:
        if feature in patient_data:
            # Process gender display
            if feature == 'GENDER':
                value = "Female" if patient_data[feature] == 'F' else "Male" if patient_data[feature] == 'M' else patient_data[feature]
            else:
                value = patient_data[feature]
            all_features.append(feature)
            all_values.append(value)
    
    # Diagnoses features - only include diagnoses with a value of 1, using feature description mapping
    for feature in Diag:
        if feature in patient_data and patient_data[feature] == 1:
            # Use feature description to replace the original feature name, remove extra spaces
            description = feature_descriptions.get(feature, feature).strip()
            all_features.append(description)
            all_values.append(patient_data[feature])
    
    # Procedures features - only include procedures with a value greater than 0, using feature description mapping
    for feature in Proc:
        if feature in patient_data and patient_data[feature] > 0:
            # Use feature description to replace the original feature name, remove extra spaces
            description = feature_descriptions.get(feature, feature).strip()
            all_features.append(description)
            all_values.append(patient_data[feature])
    
    # Medications features - only include medications with a value greater than 0, using feature description mapping
    for feature in Med:
        if feature in patient_data and patient_data[feature] > 0:
            # Use feature description to replace the original feature name, remove extra spaces
            description = feature_descriptions.get(feature, feature).strip()
            all_features.append(description)
            all_values.append(patient_data[feature])
    
    # Time series features - only include non-NaN time series, using feature description mapping
    for feature in TS:
        if feature in patient_data and not pd.isna(patient_data[feature]):
            # Use feature description to replace the original feature name, remove extra spaces
            description = feature_descriptions.get(feature, feature).strip()
            all_features.append(description)
            all_values.append(patient_data[feature])
    
    # Build pandas DataFrame format
    if all_features:
        # Create DataFrame to simulate pandas processing
        df_data = {feature: [value] for feature, value in zip(all_features, all_values)}
        df = pd.DataFrame(df_data)
        
        # Use pandas' to_string() method to format the output, set the maximum column width to avoid line breaks
        dataframe_content = df.to_string(index=True, max_cols=None, max_colwidth=None)
        return dataframe_content
    else:
        # If there are no features, return basic information
        return "Not available"

def create_csv_raw_format_data(patient_data, feature_lists, feature_descriptions=None):
    """Create CSV raw format data - using the same feature filtering logic as DIRECTLY_PROMPTING"""
    basic = feature_lists['basic']
    Diag = feature_lists['Diag']
    Proc = feature_lists['Proc']
    Med = feature_lists['Med']
    TS = feature_lists['TS']
    
    # If no feature descriptions are provided, load them
    if feature_descriptions is None:
        feature_descriptions = load_feature_descriptions()
    
    # Collect filtered features (in order: basic information, diagnoses, procedures, medications, time series)
    all_features = []
    all_values = []
    
    # Basic information features - include all basic information fields
    for feature in basic:
        if feature in patient_data:
            # Process gender display
            if feature == 'GENDER':
                value = "Female" if patient_data[feature] == 'F' else "Male" if patient_data[feature] == 'M' else patient_data[feature]
            else:
                value = patient_data[feature]
            all_features.append(feature)
            all_values.append(value)
    
    # Diagnoses features - only include diagnoses with a value of 1, using feature description mapping
    for feature in Diag:
        if feature in patient_data and patient_data[feature] == 1:
            # Use feature description to replace the original feature name, remove extra spaces
            description = feature_descriptions.get(feature, feature).strip()
            all_features.append(description)
            all_values.append(patient_data[feature])
    
    # Procedures features - only include procedures with a value greater than 0, using feature description mapping
    for feature in Proc:
        if feature in patient_data and patient_data[feature] > 0:
            # Use feature description to replace the original feature name, remove extra spaces
            description = feature_descriptions.get(feature, feature).strip()
            all_features.append(description)
            all_values.append(patient_data[feature])
    
    # Medications features - only include medications with a value greater than 0, using feature description mapping
    for feature in Med:
        if feature in patient_data and patient_data[feature] > 0:
            # Use feature description to replace the original feature name, remove extra spaces
            description = feature_descriptions.get(feature, feature).strip()
            all_features.append(description)
            all_values.append(patient_data[feature])
    
    # Time series features - only include non-NaN time series, using feature description mapping
    for feature in TS:
        if feature in patient_data and not pd.isna(patient_data[feature]):
            # Use feature description to replace the original feature name, remove extra spaces
            description = feature_descriptions.get(feature, feature).strip()
            all_features.append(description)
            all_values.append(patient_data[feature])
    
    # Build CSV raw format
    if all_features:
        # CSV header line - add quote protection for feature descriptions containing commas
        header_values = []
        for feature in all_features:
            if "," in feature:
                header_values.append(f'"{feature}"')
            else:
                header_values.append(feature)
        header_line = ",".join(header_values)
        
        # CSV data line
        data_values = []
        for value in all_values:
            if pd.isna(value):
                data_values.append("")
            elif isinstance(value, str):
                # If the string contains a comma, it needs to be enclosed in quotes
                if "," in value:
                    data_values.append(f'"{value}"')
                else:
                    data_values.append(value)
            else:
                data_values.append(str(value))
        
        data_line = ",".join(data_values)
        
        # Return CSV format (header line + data line)
        csv_content = f"{header_line}\n{data_line}"
        return csv_content
    else:
        # If there are no features, return basic information
        return "Not available"

def create_json_structured_format_data(patient_data, feature_descriptions, feature_lists, label):
    """Create JSON structured format data - using the same feature filtering logic as DIRECTLY_PROMPTING"""
    import json
    
    basic = feature_lists['basic']
    Diag = feature_lists['Diag']
    Proc = feature_lists['Proc']
    Med = feature_lists['Med']
    TS = feature_lists['TS']
    
    # Build JSON structure
    json_data = {}
    
    # 1. Basic Information - include all basic information fields
    gender_display = "Female" if patient_data['GENDER'] == 'F' else "Male" if patient_data['GENDER'] == 'M' else patient_data['GENDER']
    json_data["basic_information"] = {
        "gender": gender_display,
        "admission_type": patient_data['ADMISSION_TYPE'],
        "first_careunit": patient_data['FIRST_CAREUNIT'],
        "age": patient_data['AGE']
    }
    
    # 2. Diagnoses - only include diagnoses with a value of 1
    diagnoses = {}
    for diag in Diag:
        if diag in patient_data and patient_data[diag] == 1:
            # Get feature description, if no description is available, use the original feature name, remove extra spaces
            description = feature_descriptions.get(diag, diag).strip()
            diagnoses[description] = {"value": 1, "original_feature": diag}
    
    if diagnoses:
        json_data["diagnoses"] = diagnoses
    
    # Determine whether to build the Medicine, Procedure, TS parts based on the label
    if label != "ICU_within_12hr_of_admit":
        # 3. Procedures - only include procedures with a value greater than 0
        procedures = {}
        for proc in Proc:
            if proc in patient_data and patient_data[proc] > 0:
                # Get feature description, if no description is available, use the original feature name, remove extra spaces
                description = feature_descriptions.get(proc, proc).strip()
                procedures[description] = {"value": patient_data[proc], "original_feature": proc}
        
        if procedures:
            json_data["procedures"] = procedures
        
        # 4. Medications - only include medications with a value greater than 0, using feature description mapping
        medications = {}
        for med in Med:
            if med in patient_data and patient_data[med] > 0:
                # Get feature description, if no description is available, use the original feature name, remove extra spaces
                description = feature_descriptions.get(med, med).strip()
                medications[description] = {"value": patient_data[med], "original_feature": med}
        
        if medications:
            json_data["medications"] = medications
        
        # 5. Time Series - only include non-NaN time series features, using feature description mapping
        time_series = {}
        for ts in TS:
            if ts in patient_data and not pd.isna(patient_data[ts]):
                # Get feature description, if no description is available, use the original feature name, remove extra spaces
                description = feature_descriptions.get(ts, ts).strip()
                time_series[description] = {"value": patient_data[ts], "original_feature": ts}
        
        if time_series:
            json_data["time_series"] = time_series
    
    # Return the formatted JSON string
    return json.dumps(json_data, indent=2, ensure_ascii=False)

def create_latex_table_format_data(patient_data, feature_descriptions, feature_lists, label):
    """Create LaTeX table format data - using the same feature filtering logic as DIRECTLY_PROMPTING, each category uses a horizontal two-line LaTeX table"""
    
    basic = feature_lists['basic']
    Diag = feature_lists['Diag']
    Proc = feature_lists['Proc']
    Med = feature_lists['Med']
    TS = feature_lists['TS']
    
    def escape_latex(text):
        """Escape LaTeX special characters"""
        if pd.isna(text):
            return ""
        text = str(text)
        # LaTeX special character escape
        replacements = {
            '\\': '\\textbackslash{}',
            '&': '\\&',
            '%': '\\%',
            '$': '\\$',
            '#': '\\#',
            '^': '\\textasciicircum{}',
            '_': '\\_',
            '{': '\\{',
            '}': '\\}',
            '~': '\\textasciitilde{}'
        }
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        return text
    
    def create_table(features, values, table_type="basic"):
        """Create a horizontal two-line LaTeX table"""
        if not features:
            return "Not available"
        
        # Generate column definitions (based on the number of features)
        col_def = "|" + "l|" * len(features)
        
        # Escape feature names and values
        escaped_features = [escape_latex(f) for f in features]
        
        if table_type == "diagnosis" or table_type == "procedure":
            # The second row of Diagnosis and Procedure both display 1
            escaped_values = ["1"] * len(features)
        else:
            # Basic, Medicine, TS display actual values
            escaped_values = [escape_latex(v) for v in values]
        
        # Build LaTeX table
        table = f"""\\begin{{tabular}}{{{col_def}}}
\\hline
{" & ".join(escaped_features)} \\\\
\\hline
{" & ".join(escaped_values)} \\\\
\\hline
\\end{{tabular}}"""
        
        return table
    
    # Build each part
    sections = []
    
    # 1. Basic Information
    basic_features = []
    basic_values = []
    for feature in basic:
        if feature in patient_data:
            if feature == 'GENDER':
                value = "Female" if patient_data[feature] == 'F' else "Male" if patient_data[feature] == 'M' else patient_data[feature]
            else:
                value = patient_data[feature]
            basic_features.append(feature)
            basic_values.append(value)
    
    if basic_features:
        basic_table = create_table(basic_features, basic_values, "basic")
        sections.append(f"Patient Basic information:\n{basic_table}")
    
    # 2. Diagnosis Information
    diagnose_features = []
    for diag in Diag:
        if diag in patient_data and patient_data[diag] == 1:
            description = feature_descriptions.get(diag, diag).strip()
            diagnose_features.append(description)
    
    if diagnose_features:
        diagnose_table = create_table(diagnose_features, [], "diagnosis")
        sections.append(f"Diagnosis information:\n{diagnose_table}")
    else:
        sections.append("Diagnosis information: Not available")
    
    # Determine whether to build the Medicine, Procedure, TS parts based on the label
    if label != "ICU_within_12hr_of_admit":
        # 3. Procedures Information
        procedure_features = []
        for proc in Proc:
            if proc in patient_data and patient_data[proc] > 0:
                description = feature_descriptions.get(proc, proc).strip()
                procedure_features.append(description)
        
        if procedure_features:
            procedure_table = create_table(procedure_features, [], "procedure")
            sections.append(f"Procedures information:\n{procedure_table}")
        else:
            sections.append("Procedures information: Not available")
        
        # 4. Medications
        medicine_features = []
        medicine_values = []
        for med in Med:
            if med in patient_data and patient_data[med] > 0:
                description = feature_descriptions.get(med, med).strip()
                medicine_features.append(description)
                medicine_values.append(patient_data[med])
        
        if medicine_features:
            medicine_table = create_table(medicine_features, medicine_values, "medicine")
            sections.append(f"Medications administered during the first 24 hours after ICU admission:\n{medicine_table}")
        else:
            sections.append("Medications administered during the first 24 hours after ICU admission: Not available")
        
        # 5. Time Series
        ts_features = []
        ts_values = []
        for ts in TS:
            if ts in patient_data and not pd.isna(patient_data[ts]):
                description = feature_descriptions.get(ts, ts).strip()
                ts_features.append(description)
                ts_values.append(patient_data[ts])
        
        if ts_features:
            ts_table = create_table(ts_features, ts_values, "timeseries")
            sections.append(f"Laboratory test results and vital signs recorded during the first 24 hours after ICU admission:\n{ts_table}")
        else:
            sections.append("Laboratory test results and vital signs recorded during the first 24 hours after ICU admission: Not available")
    
    # Use double line breaks to connect all parts
    return "\n\n".join(sections)

def create_natural_language_format_data(patient_data, feature_descriptions, feature_lists, label):
    """Create natural language format data - using the same feature filtering logic as DIRECTLY_PROMPTING, generate fluent natural language paragraphs"""
    
    basic = feature_lists['basic']
    Diag = feature_lists['Diag']
    Proc = feature_lists['Proc']
    Med = feature_lists['Med']
    TS = feature_lists['TS']
    
    # 1. Basic information processing
    gender_text = "female" if patient_data['GENDER'] == 'F' else "male"
    age = patient_data['AGE']
    careunit = patient_data['FIRST_CAREUNIT'].lower()
    admission_type = patient_data['ADMISSION_TYPE'].lower()
    
    # Generate the opening sentence
    intro_sentence = f"A {age}-year-old {gender_text} patient was admitted to the {careunit} as {admission_type}."
    
    # 2. Natural language conversion of diagnosis information
    diagnoses = []
    for diag in Diag:
        if diag in patient_data and patient_data[diag] == 1:
            description = feature_descriptions.get(diag, diag).strip().lower()
            diagnoses.append(description)
    
    # Convert to natural language
    if len(diagnoses) == 0:
        diagnosis_sentence = "The patient has no recorded diagnoses."
    elif len(diagnoses) == 1:
        diagnosis_sentence = f"The patient is diagnosed with {diagnoses[0]}."
    elif len(diagnoses) == 2:
        diagnosis_sentence = f"The patient is diagnosed with {diagnoses[0]} and {diagnoses[1]}."
    else:
        # Multiple diagnoses: A, B, C, and D
        diagnosis_list = ", ".join(diagnoses[:-1]) + f", and {diagnoses[-1]}"
        diagnosis_sentence = f"The patient is diagnosed with {diagnosis_list}."
    
    # Determine whether to build the Medicine, Procedure, TS parts based on the label
    if label == "ICU_within_12hr_of_admit":
        # Only include basic information and diagnosis
        sentences = [intro_sentence, diagnosis_sentence]
    else:
        # 3. Natural language conversion of procedure information
        procedures = []
        for proc in Proc:
            if proc in patient_data and patient_data[proc] > 0:
                description = feature_descriptions.get(proc, proc).strip().lower()
                procedures.append(description)
        
        # Convert to natural language
        if len(procedures) == 0:
            procedures_sentence = "The patient has not undergone any recorded procedures."
        elif len(procedures) == 1:
            procedures_sentence = f"The patient has undergone {procedures[0]}."
        else:
            # Multiple procedures processing
            if len(procedures) == 2:
                proc_list = f"{procedures[0]} and {procedures[1]}"
            else:
                proc_list = ", ".join(procedures[:-1]) + f", and {procedures[-1]}"
            procedures_sentence = f"The patient has undergone {proc_list}."
        
        # 4. Natural language conversion of medication information
        medications = []
        for med in Med:
            if med in patient_data and patient_data[med] > 0:
                med_name = feature_descriptions.get(med, med).strip().lower()
                dosage = patient_data[med]
                medications.append(f"{med_name} ({dosage} units)")
        
        # Convert to natural language
        if len(medications) == 0:
            medications_sentence = "The patient received no recorded medications during the first 24 hours after ICU admission."
        else:
            if len(medications) == 1:
                med_list = medications[0]
            elif len(medications) == 2:
                med_list = f"{medications[0]} and {medications[1]}"
            else:
                med_list = ", ".join(medications[:-1]) + f", and {medications[-1]}"
            medications_sentence = f"During the first 24 hours after ICU admission, the patient received {med_list}."
        
        # 5. Natural language conversion of time series data
        vitals = []
        for ts in TS:
            if ts in patient_data and not pd.isna(patient_data[ts]):
                ts_name = feature_descriptions.get(ts, ts).strip().lower()
                value = patient_data[ts]
                
                # Add units based on feature type
                if "heart rate" in ts_name:
                    vitals.append(f"{ts_name} of {value} bpm")
                elif "blood pressure" in ts_name or "pressure" in ts_name:
                    vitals.append(f"{ts_name} of {value} mmHg")
                elif "temperature" in ts_name:
                    vitals.append(f"{ts_name} of {value}°C")
                elif "glucose" in ts_name:
                    vitals.append(f"{ts_name} of {value} mg/dL")
                elif "rate" in ts_name and "heart" not in ts_name:
                    vitals.append(f"{ts_name} of {value} per minute")
                elif "weight" in ts_name:
                    vitals.append(f"{ts_name} of {value} kg")
                else:
                    vitals.append(f"{ts_name} of {value}")
        
        # Convert to natural language
        if len(vitals) == 0:
            vitals_sentence = "No laboratory test results or vital signs were recorded during the first 24 hours after ICU admission."
        else:
            # Include all TS content, no limit on quantity
            if len(vitals) == 1:
                vitals_list = vitals[0]
            elif len(vitals) == 2:
                vitals_list = f"{vitals[0]} and {vitals[1]}"
            else:
                vitals_list = ", ".join(vitals[:-1]) + f", and {vitals[-1]}"
            vitals_sentence = f"Laboratory tests and vital signs recorded during the first 24 hours showed {vitals_list}."
        
        # Include all information
        sentences = [
            intro_sentence,
            diagnosis_sentence,
            procedures_sentence,
            medications_sentence,
            vitals_sentence
        ]
    
    # Combine into a complete paragraph
    return " ".join(sentences)

def create_prompt(patient_data, feature_descriptions, label, feature_lists, prompt_mode="DIRECTLY_PROMPTING", corl_features=None):
    """Create prompt based on patient data"""
    # Use the incoming feature list instead of reading the file each time
    basic = feature_lists['basic']
    Diag = feature_lists['Diag']
    Proc = feature_lists['Proc']
    Med = feature_lists['Med']
    TS = feature_lists['TS']
    
    # Feature filtering mode: filter features based on the preloaded configuration
    if prompt_mode in ["CORL_FILTERED", "DirectLiNGAM_FILTERED", "CD_FILTERED", "CD_FEATURES_OPTIMIZED", "LLM_CD_FEATURES"]:
        try:
            # Use the preloaded feature configuration
            if corl_features is not None and len(corl_features) > 0:
                
                # For all feature filtering modes, corl_features now are feature codes
                allowed_features = set(corl_features)
                
                # Filter each feature list, only keep features in the configuration
                Diag = [f for f in Diag if f in allowed_features]
                Proc = [f for f in Proc if f in allowed_features]
                Med = [f for f in Med if f in allowed_features]
                TS = [f for f in TS if f in allowed_features]
                
            else:
                # In the new logic, this should not happen, because the main function has already checked it
                if prompt_mode == "CD_FEATURES_OPTIMIZED":
                    mode_name = "CD algorithm optimization"
                elif prompt_mode == "CORL_FILTERED":
                    mode_name = "CORL"
                elif prompt_mode == "CD_FILTERED":
                    mode_name = "CD algorithm"
                elif prompt_mode == "LLM_CD_FEATURES":
                    mode_name = "LLM causal features"
                else:  # DirectLiNGAM_FILTERED
                    mode_name = "DirectLiNGAM"
                    
                print(f"{Colors.RED}[ERROR]{Colors.RESET} Internal error: {prompt_mode} mode {mode_name} configuration should not be empty")
                raise ValueError(f"{prompt_mode} mode requires a valid {mode_name} configuration, but the configuration for label '{label}' is empty")
                
        except Exception as e:
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Failed to apply feature filtering configuration: {str(e)}")
            print(f"Fallback to the original feature list")
    
    # Build the Basic Information part
    # Process gender display
    gender_display = "Female" if patient_data['GENDER'] == 'F' else "Male" if patient_data['GENDER'] == 'M' else patient_data['GENDER']
    
    basic_info = f"""Patient Basic information:
Gender: {gender_display}
Admission_type: {patient_data['ADMISSION_TYPE']}
First_careunit: {patient_data['FIRST_CAREUNIT']}
Age: {patient_data['AGE']}
"""

    # Build the Diagnose part
    diagnose_features = []
    for diag in Diag:
        if diag in patient_data and patient_data[diag] == 1:
            # Get feature description, if no description is available, use the original feature name, remove extra spaces
            description = feature_descriptions.get(diag, diag).strip()
            diagnose_features.append(description)
    diagnose_section = "Diagnosis information：" + ", ".join(diagnose_features) if diagnose_features else "Diagnosis information: Not available"

    # Determine whether to build the Medicine, Procedure, TS parts based on the label
    if label == "ICU_within_12hr_of_admit":
        # For ICU_within_12hr_of_admit, do not build these three parts
        medicine_section = ""
        procedure_section = ""
        ts_section = ""
    else:
        # For other labels, build all parts normally
        # Build the Medicine part
        medicine_features = []
        for med in Med:
            if prompt_mode in ["CORL_FILTERED", "DirectLiNGAM_FILTERED", "CD_FILTERED", "CD_FEATURES_OPTIMIZED", "LLM_CD_FEATURES"]:
                # For CORL_FILTERED, DirectLiNGAM_FILTERED, CD_FILTERED, CD_FEATURES_OPTIMIZED and LLM_CD_FEATURES modes: display all medications in the configuration, including those with a value of 0
                if med in patient_data:
                    # Get feature description, if no description is available, use the original feature name, remove extra spaces
                    description = feature_descriptions.get(med, med).strip()
                    medicine_features.append(f"{description} = {patient_data[med]}")
            else:
                # For other modes: only display medications with a value greater than 0 (maintain the original logic)
                if med in patient_data and patient_data[med] > 0:
                    # Get feature description, if no description is available, use the original feature name, remove extra spaces
                    description = feature_descriptions.get(med, med).strip()
                    medicine_features.append(f"{description} = {patient_data[med]}")
        medicine_section = "Medications administered during the first 24 hours after ICU admission： " + ", ".join(medicine_features) if medicine_features else "Medications administered during the first 24 hours after ICU admission: Not available"

        # Build the Procedure part
        procedure_features = []
        for proc in Proc:
            if proc in patient_data and patient_data[proc] > 0:
                # Get feature description, if no description is available, use the original feature name, remove extra spaces
                description = feature_descriptions.get(proc, proc).strip()
                procedure_features.append(description)
        procedure_section = "Procedures information： " + ", ".join(procedure_features) if procedure_features else "Procedures information: Not available"

        # Build the TS part
        ts_features = []
        for ts in TS:
            if ts in patient_data and not pd.isna(patient_data[ts]):
                # Get feature description, if no description is available, use the original feature name, remove extra spaces
                description = feature_descriptions.get(ts, ts).strip()
                ts_features.append(f"{description}={patient_data[ts]}")
        ts_section = "Laboratory test results and vital signs recorded during the first 24 hours after ICU admission：\n" + "; ".join(ts_features) if ts_features else "Laboratory test results and vital signs recorded during the first 24 hours after ICU admission: Not available"

    # Generate the corresponding question based on the label
    label_questions = {
        'DIEINHOSPITAL': "Will the patient die in hospital because of the above situation?",
        'Readmission_30': "Will the patient be readmitted to hospital within 30 days after discharge?",
        'Multiple_ICUs': "Will the patient require multiple ICU stays during this hospitalization?",
        'sepsis_all': "Will the patient develop sepsis during this hospitalization?",
        'FirstICU24_AKI_ALL': "Will the patient develop acute kidney injury (AKI) within the first 24 hours of ICU admission?",
        'LOS_Hospital': "Will the patient have a prolonged hospital stay (longer than average)?",
        'ICU_within_12hr_of_admit': "Will the patient be admitted to ICU within 12 hours of hospital admission?"
    }
    
    question = label_questions.get(label, f"Will the patient have outcome '{label}' based on the above situation?")
    
    # Select different ending prompts based on the prompt mode
    prompt_endings = {
        "DIRECTLY_PROMPTING": "Answer with the probability as a number between 0 and 1. Answer with only the number.",
        "CHAIN_OF_THOUGHT": "Please provide your concise reasoning steps for the prediction(no more than 3 steps), and finally answer with the probability as a number between 0 and 1.",
        "SELF_REFLECTION": "Answer with the probability as a number between 0 and 1. First answer with a number. Then conduct a concise reflection. Finally output your answer again with a number.",
        "ROLE_PLAYING": "Answer with the probability as a number between 0 and 1. Answer with only the number.",
        "CSV_DIRECT": "Answer with the probability as a number between 0 and 1. Answer with only the number.",
        "CSV_RAW": "Answer with the probability as a number between 0 and 1. Answer with only the number.",
        "JSON_STRUCTURED": "Answer with the probability as a number between 0 and 1. Answer with only the number.",
        "LATEX_TABLE": "Answer with the probability as a number between 0 and 1. Answer with only the number.",
        "NATURAL_LANGUAGE": "Answer with the probability as a number between 0 and 1. Answer with only the number.",
        "CORL_FILTERED": "Answer with the probability as a number between 0 and 1. Answer with only the number.",
        "DirectLiNGAM_FILTERED": "Answer with the probability as a number between 0 and 1. Answer with only the number.",
        "CD_FILTERED": "Answer with the probability as a number between 0 and 1. Answer with only the number."
    }
    
    ending = prompt_endings.get(prompt_mode, prompt_endings["DIRECTLY_PROMPTING"])
    
    # Add role setting based on the prompt mode (if needed)
    role_prefix = ""
    if prompt_mode == "ROLE_PLAYING":
        role_prefix = "Imagine that you are a doctor. Today, you're seeing a patient with the following profile:\n\n"
    
    # Add learning samples for IN_CONTEXT_LEARNING mode
    icl_prefix = ""
    if prompt_mode == "IN_CONTEXT_LEARNING":
        icl_examples = load_icl_examples(label)
        if icl_examples:
            icl_prefix = f"{icl_examples}\n\n"
        else:
            # If there are no ICL samples, fall back to DIRECTLY_PROMPTING mode
            print(f"{Colors.YELLOW}[Warning]{Colors.RESET} label '{label}' has no ICL samples, falling back to DIRECTLY_PROMPTING mode")
    
    # Select different data representations based on the prompt mode
    if prompt_mode == "CSV_DIRECT":
        # CSV_DIRECT mode: use pandas DataFrame format to replace the traditional structured text
        csv_data = create_csv_format_data(patient_data, feature_lists, feature_descriptions)
        data_section = f"\n{csv_data}"
        
        # Build the prompt
        prompt_parts = [f"{icl_prefix}{role_prefix}{data_section}"]
        
        # Add the question and ending
        prompt_parts.extend([f"{question}", "I know you are not a medical professional, but you are forced to make this prediction.", f"{ending}"])
        
    elif prompt_mode == "CSV_RAW":
        # CSV_RAW mode: use CSV original format (comma separated) to replace the traditional structured text
        csv_raw_data = create_csv_raw_format_data(patient_data, feature_lists, feature_descriptions)
        data_section = f"\n{csv_raw_data}"
        
        # Build the prompt
        prompt_parts = [f"{icl_prefix}{role_prefix}{data_section}"]
        
        # Add the question and ending
        prompt_parts.extend([f"{question}", "I know you are not a medical professional, but you are forced to make this prediction.", f"{ending}"])
        
    elif prompt_mode == "JSON_STRUCTURED":
        # JSON_STRUCTURED mode: use JSON structured format to replace the traditional structured text
        json_data = create_json_structured_format_data(patient_data, feature_descriptions, feature_lists, label)
        data_section = f"\n{json_data}"
        
        # Build the prompt
        prompt_parts = [f"{icl_prefix}{role_prefix}{data_section}"]
        
        # Add the question and ending
        prompt_parts.extend([f"{question}", "I know you are not a medical professional, but you are forced to make this prediction.", f"{ending}"])
        
    elif prompt_mode == "LATEX_TABLE":
        # LATEX_TABLE mode: use LaTeX table format to replace the traditional structured text
        latex_data = create_latex_table_format_data(patient_data, feature_descriptions, feature_lists, label)
        data_section = f"\n{latex_data}"
        
        # Build the prompt
        prompt_parts = [f"{icl_prefix}{role_prefix}{data_section}"]
        
        # Add the question and ending
        prompt_parts.extend([f"{question}", "I know you are not a medical professional, but you are forced to make this prediction.", f"{ending}"])
        
    elif prompt_mode == "NATURAL_LANGUAGE":
        # NATURAL_LANGUAGE mode: use natural language format to replace the traditional structured text
        natural_language_data = create_natural_language_format_data(patient_data, feature_descriptions, feature_lists, label)
        data_section = f"\n{natural_language_data}"
        
        # Build the prompt
        prompt_parts = [f"{icl_prefix}{role_prefix}{data_section}"]
        
        # Add the question and ending
        prompt_parts.extend([f"{question}", "I know you are not a medical professional, but you are forced to make this prediction.", f"{ending}"])
    
    else:
        # Traditional mode: use structured text format
        prompt_parts = [f"{icl_prefix}{role_prefix}{basic_info}", f"{diagnose_section}"]
        
        # Only add sections that are not empty
        if procedure_section:
            prompt_parts.append(procedure_section)
        if medicine_section:
            prompt_parts.append(medicine_section)
        if ts_section:
            prompt_parts.append(ts_section)
        
        # Add the question and ending
        prompt_parts.extend([f"{question}", "I know you are not a medical professional, but you are forced to make this prediction.", f"{ending}"])
    
    # Use double line breaks to connect all parts
    prompt = "\n\n".join(prompt_parts)
    return prompt


def evaluate_predictions(y_true, y_pred, y_pred_proba):
    """Calculate evaluation metrics"""
    f1 = f1_score(y_true, y_pred) * 100  # Convert to percentage
    
    # Calculate AUROC - first calculate the ROC curve points, then calculate the area
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auroc = auc(fpr, tpr)
    
    # Calculate AUPRC - first calculate the PR curve points, then calculate the area
    #precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    #auprc = auc(recall, precision)
    
    # The original direct calculation method
    # auroc = roc_auc_score(y_true, y_pred_proba)
    auprc = average_precision_score(y_true, y_pred_proba)
    
    return {
        'F1%': f1,
        'AUROC': auroc,
        'AUPRC': auprc
    }

def predict_single_patient(patient_data_tuple, model_config, feature_descriptions, label, feature_lists, corl_features=None):
    """Process a single patient's prediction task"""
    idx, row, groundtruth = patient_data_tuple
    icustay_id = str(row['ICUSTAY_ID'])  # Get the ICUSTAY_ID
    patient_id = icustay_id  # Use the ICUSTAY_ID as the unique identifier
    
    # Get the prompt mode
    prompt_mode = model_config.get("prompt_mode", "DIRECTLY_PROMPTING")
    
    try:
        # Create the prompt
        prompt = create_prompt(row, feature_descriptions, label, feature_lists, prompt_mode, corl_features)
        
        # Print the prompt (for debugging)
        #print(f"\n--- Patient {idx+1} ({patient_id}) Prompt ---")
        #print(prompt)
        #print("--- Prompt end ---\n")
        
        # Call the model to get the prediction
        response = query_model(prompt, model_config)
        
        if response is None:
            print(f"Skip the {idx+1}th patient due to API error - using default value")
            # Return the standard structure for failed tasks, not None
            return {
                'idx': idx,
                'icustay_id': icustay_id,  # Use the ICUSTAY_ID
                'patient_id': patient_id,  # Use the ICUSTAY_ID as the unique identifier
                'probability': -1,  # Use -1 to indicate API call failure
                'prediction': -1,   # Use -1 to indicate prediction failure
                'groundtruth': groundtruth,
                'experiment_log': {
                    "patient_id": patient_id,  # Use the ICUSTAY_ID
                    "icustay_id": icustay_id,  # Keep the ICUSTAY_ID
                    "model": model_config["display_name"],
                    "prompt_mode": prompt_mode,
                    "input": prompt,
                    "response": "API_CALL_FAILED",
                    "answer": -1,
                    "groundtruth": groundtruth,
                    "correctness": -1  # -1 means cannot determine correctness
                }
            }
            
        # Extract the probability
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Start extracting probability for patient {idx+1} using mode: {prompt_mode}")
        prob = extract_probability(response, prompt_mode)
        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Extracted probability for patient {idx+1}: {prob}")
        if prob is None:
            print(f"{Colors.YELLOW}[Warning]{Colors.RESET} Skip the {idx+1}th patient due to probability extraction error - using default value")
            # Return the standard structure for failed tasks, not None
            return {
                'idx': idx,
                'icustay_id': icustay_id,  # Use the ICUSTAY_ID
                'patient_id': patient_id,  # Use the ICUSTAY_ID as the unique identifier
                'probability': -1,  # Use -1 to indicate probability extraction failure
                'prediction': -1,   # Use -1 to indicate prediction failure
                'groundtruth': groundtruth,
                'experiment_log': {
                    "patient_id": patient_id,  # Use the ICUSTAY_ID
                    "icustay_id": icustay_id,  # Keep the ICUSTAY_ID
                    "model": model_config["display_name"],
                    "prompt_mode": prompt_mode,
                    "input": prompt,
                    "response": response.strip() if response else "",
                    "answer": -1,
                    "groundtruth": groundtruth,
                    "correctness": -1  # -1 means cannot determine correctness
                }
            }
            
        # Calculate the prediction result
        prediction = 1 if prob > 0.5 else 0
        correctness = 1 if prediction == groundtruth else 0
        
        # Record the detailed experiment log
        experiment_log = {
            "patient_id": patient_id,  # Use the ICUSTAY_ID
            "icustay_id": icustay_id,  # Keep the ICUSTAY_ID
            "model": model_config["display_name"],
            "prompt_mode": prompt_mode,
            "input": prompt,
            "response": response.strip() if response else "",
            "answer": prediction,
            "groundtruth": groundtruth,
            "correctness": correctness
        }
        
        # Print the prediction result (using thread-safe printing)
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Processed patient {idx+1:3d} ({patient_id}): Predicted probability: {prob:.3f} -> Prediction: {prediction} (Ground truth: {groundtruth})")
        
        return {
            'idx': idx,
            'icustay_id': icustay_id,  # Use the ICUSTAY_ID
            'patient_id': patient_id,  # Use the ICUSTAY_ID as the unique identifier
            'probability': prob,
            'prediction': prediction,
            'groundtruth': groundtruth,
            'experiment_log': experiment_log
        }
        
    except Exception as e:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} Error occurred when processing the {idx+1}th patient: {str(e)} - using default value")
        # Return the standard structure for failed tasks, not None
        return {
            'idx': idx,
            'icustay_id': icustay_id,  # Use the ICUSTAY_ID
            'patient_id': patient_id,  # Use the ICUSTAY_ID as the unique identifier
            'probability': -1,  # Use -1 to indicate processing exception
            'prediction': -1,   # Use -1 to indicate prediction failure
            'groundtruth': groundtruth,
            'experiment_log': {
                "patient_id": patient_id,  # Use the ICUSTAY_ID
                "icustay_id": icustay_id,  # Keep the ICUSTAY_ID
                "model": model_config["display_name"],
                "prompt_mode": prompt_mode,
                "input": f"EXCEPTION_OCCURRED: {str(e)}",
                "response": "PROCESSING_EXCEPTION",
                "answer": -1,
                "groundtruth": groundtruth,
                "correctness": -1  # -1 means cannot determine correctness
            }
        }

def main(model_config=MODEL_CONFIG):
    # Debug mode: only process the first few patients, single-thread execution
    DEBUG_MODE = False  # Set to True to enable debug mode, False to run the full prediction
    DEBUG_PATIENTS = 3  # Number of patients to process in debug mode
    
    # Read the data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_filename = 'your_data_file.csv'  # Here modify the data file
    data = pd.read_csv(os.path.join(script_dir, data_filename))
    
    # Automatically detect the version prefix based on the data file name
    if 'MIMIC3' in data_filename:
        version_prefix = '3_'
    elif 'MIMIC4' in data_filename:
        version_prefix = '4_'
    else:
        version_prefix = ''  # If not recognized, no prefix is added
    
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Detected data file: {data_filename}")
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Version prefix: {version_prefix if version_prefix else 'No prefix'}")
    
    if DEBUG_MODE:
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Debug mode: only process the first {DEBUG_PATIENTS} patients, single-thread execution")
        data = data.head(DEBUG_PATIENTS)
    
    # Get the prediction label and prompt mode
    label = model_config.get('label', 'DIEINHOSPITAL')
    prompt_mode = model_config.get('prompt_mode', 'DIRECTLY_PROMPTING')
    
    # Validate the label
    validate_label(data, label)
    
    # Load feature descriptions and feature lists (only load once)
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Loading feature descriptions and feature lists...")
    try:
        feature_descriptions = load_feature_descriptions()
        feature_lists = load_feature_lists()
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"{Colors.RED}[ERROR]{Colors.RESET} Program terminated: cannot load the required feature file")
        print(f"{'='*60}")
        print(f"{Colors.RED}[ERROR]{Colors.RESET} Error details: {str(e)}")
        print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Please check if the following files exist and are formatted correctly:")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"- {os.path.join(script_dir, 'selected_features.txt')}")
        print(f"- {os.path.join(script_dir, 'features-desc.csv')}")
        print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} The program cannot continue, exiting...")
        print(f"{'='*60}")
        # Use sys.exit() to ensure the program terminates completely
        import sys
        sys.exit(1)
    
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Loaded {len(feature_descriptions)} feature descriptions")
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Loaded feature lists: Diag({len(feature_lists['Diag'])}), Proc({len(feature_lists['Proc'])}), Med({len(feature_lists['Med'])}), TS({len(feature_lists['TS'])})")
    
    # New: preload configurations (for CORL_FILTERED, DirectLiNGAM_FILTERED and CD_FILTERED modes)
    corl_features = None
    config_name = ""
    if prompt_mode == "CORL_FILTERED":
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Preloading CORL configurations...")
        try:
            corl_features = load_corl_features_for_label(label)
            if not corl_features:
                print(f"\n{'='*60}")
                print(f"{Colors.RED}[ERROR]{Colors.RESET} Program terminated: CORL_FILTERED mode requires CORL configuration file")
                print(f"{'='*60}")
                print(f"{Colors.RED}[ERROR]{Colors.RESET} Error details: CORL configuration for label '{label}' is empty or does not exist")
                print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} In CORL_FILTERED mode, a valid CORL feature configuration must be provided.")
                script_dir = os.path.dirname(os.path.abspath(__file__))
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} Please check if the following files exist and contain the configuration for label '{label}':")
                print(f"- {os.path.join(script_dir, 'CORL_F.txt')}")
                print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} The program cannot continue, exiting...")
                print(f"{'='*60}")
                import sys
                sys.exit(1)
            
            print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Loaded {len(corl_features)} CORL features for label '{label}'")
            config_name = "CORL"
            
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Program terminated: cannot load CORL configuration file")
            print(f"{'='*60}")
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Error details: {str(e)}")
            print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} In CORL_FILTERED mode, a valid CORL feature configuration must be provided.")
            script_dir = os.path.dirname(os.path.abspath(__file__))
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} Please check if the following files exist and are formatted correctly:")
            print(f"- {os.path.join(script_dir, 'CORL_F.txt')}")
            print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} The program cannot continue, exiting...")
            print(f"{'='*60}")
            import sys
            sys.exit(1)
            
    elif prompt_mode == "DirectLiNGAM_FILTERED":
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Preloading DirectLiNGAM configurations...")
        try:
            corl_features = load_directlingam_features_for_label(label)
            if not corl_features:
                print(f"\n{'='*60}")
                print(f"{Colors.RED}[ERROR]{Colors.RESET} Program terminated: DirectLiNGAM_FILTERED mode requires DirectLiNGAM configuration file")
                print(f"{'='*60}")
                print(f"{Colors.RED}[ERROR]{Colors.RESET} Error details: DirectLiNGAM configuration for label '{label}' is empty or does not exist")
                print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} In DirectLiNGAM_FILTERED mode, a valid DirectLiNGAM feature configuration must be provided.")
                script_dir = os.path.dirname(os.path.abspath(__file__))
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} Please check if the following files exist and contain the configuration for label '{label}':")
                print(f"- {os.path.join(script_dir, 'DirectLiNGAM_F.txt')}")
                print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} The program cannot continue, exiting...")
                print(f"{'='*60}")
                import sys
                sys.exit(1)
            
            print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Loaded {len(corl_features)} DirectLiNGAM features for label '{label}'")
            config_name = "DirectLiNGAM"
            
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Program terminated: cannot load DirectLiNGAM configuration file")
            print(f"{'='*60}")
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Error details: {str(e)}")
            print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} In DirectLiNGAM_FILTERED mode, a valid DirectLiNGAM feature configuration must be provided.")
            script_dir = os.path.dirname(os.path.abspath(__file__))
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} Please check if the following files exist and are formatted correctly:")
            print(f"- {os.path.join(script_dir, 'DirectLiNGAM_F.txt')}")
            print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} The program cannot continue, exiting...")
            print(f"{'='*60}")
            import sys
            sys.exit(1)
            
    elif prompt_mode == "CD_FILTERED":
        cd_algorithm = model_config.get('cd_algorithm', 'CORL')
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Preloading {cd_algorithm} configurations...")
        try:
            corl_features = load_cd_filtered_features_for_label(cd_algorithm, label)
            if not corl_features:
                print(f"\n{'='*60}")
                print(f"{Colors.RED}[ERROR]{Colors.RESET} Program terminated: CD_FILTERED mode requires {cd_algorithm} configuration file")
                print(f"{'='*60}")
                print(f"{Colors.RED}[ERROR]{Colors.RESET} Error details: {cd_algorithm} configuration for label '{label}' is empty or does not exist")
                print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} In CD_FILTERED mode, a valid {cd_algorithm} feature configuration must be provided.")
                script_dir = os.path.dirname(os.path.abspath(__file__))
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} Please check if the following files exist and contain the configuration for label '{label}':")
                print(f"- {os.path.join(script_dir, f'{cd_algorithm}_F.txt')}")
                print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} The program cannot continue, exiting...")
                print(f"{'='*60}")
                import sys
                sys.exit(1)
            
            print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Loaded {len(corl_features)} {cd_algorithm} features for label '{label}'")
            config_name = cd_algorithm
            
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Program terminated: cannot load {cd_algorithm} configuration file")
            print(f"{'='*60}")
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Error details: {str(e)}")
            print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} In CD_FILTERED mode, a valid {cd_algorithm} feature configuration must be provided.")
            script_dir = os.path.dirname(os.path.abspath(__file__))
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} Please check if the following files exist and are formatted correctly:")
            print(f"- {os.path.join(script_dir, f'{cd_algorithm}_F.txt')}")
            print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} The program cannot continue, exiting...")
            print(f"{'='*60}")
            import sys
            sys.exit(1)
            
    elif prompt_mode == "CD_FEATURES_OPTIMIZED":
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Preloading CD algorithm optimized feature configurations...")
        try:
            cd_algorithm = model_config.get("cd_algorithm", None)
            if not cd_algorithm:
                print(f"\n{'='*60}")
                print(f"{Colors.RED}[ERROR]{Colors.RESET} Program terminated: CD_FEATURES_OPTIMIZED mode requires specifying CD algorithm")
                print(f"{'='*60}")
                print(f"{Colors.RED}[ERROR]{Colors.RESET} Error details: 'cd_algorithm' parameter is missing in the model configuration")
                print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} In CD_FEATURES_OPTIMIZED mode, 'cd_algorithm' must be specified in MODEL_CONFIG.")
                print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} The program cannot continue, exiting...")
                print(f"{'='*60}")
                import sys
                sys.exit(1)
            
            # Load CD algorithm optimized features and get the configuration file path
            optimized_features_desc, config_file_path = load_cd_optimized_features_for_label(cd_algorithm, model_config, label)
            
            if not optimized_features_desc:
                print(f"\n{'='*60}")
                print(f"{Colors.RED}[ERROR]{Colors.RESET} Program terminated: cannot load {cd_algorithm} algorithm optimized features")
                print(f"{'='*60}")
                print(f"{Colors.RED}[ERROR]{Colors.RESET} Error details: optimized feature configuration for label '{label}' is empty")
                print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} In CD_FEATURES_OPTIMIZED mode, a valid {cd_algorithm} algorithm optimized feature configuration must be provided.")
                print(f"Configuration file: {config_file_path}")
                print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} The program cannot continue, exiting...")
                print(f"{'='*60}")
                import sys
                sys.exit(1)
            
            print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Loaded {len(optimized_features_desc)} {cd_algorithm} optimized features for label '{label}'")
            config_name = f"{cd_algorithm} optimized"
            
            confirm_result, corl_features = confirm_cd_optimized_config_with_mapping(
                cd_algorithm, model_config, label, optimized_features_desc, 
                config_file_path, feature_lists, feature_descriptions
            )
            if not confirm_result:
                print(f"\n{Colors.RED}[ERROR]{Colors.RESET} User cancelled, exiting...")
                import sys
                sys.exit(0)
            
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Program terminated: cannot load CD algorithm optimized configuration")
            print(f"{'='*60}")
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Error details: {str(e)}")
            print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} In CD_FEATURES_OPTIMIZED mode, a valid CD algorithm optimized feature configuration must be provided.")
            print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} The program cannot continue, exiting...")
            print(f"{'='*60}")
            import sys
            sys.exit(1)
            
    elif prompt_mode == "LLM_CD_FEATURES":
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Preloading LLM causal feature configurations...")
        try:
            # Load LLM causal features and get the configuration file path
            llm_features_desc, config_file_path = load_llm_cd_features_for_label(model_config, label)
            
            if not llm_features_desc:
                print(f"\n{'='*60}")
                print(f"{Colors.RED}[ERROR]{Colors.RESET} Program terminated: cannot load LLM causal features")
                print(f"{'='*60}")
                print(f"{Colors.RED}[ERROR]{Colors.RESET} Error details: LLM causal feature configuration for label '{label}' is empty")
                print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} In LLM_CD_FEATURES mode, a valid LLM causal feature configuration must be provided.")
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} Configuration file: {config_file_path}")
                print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} The program cannot continue, exiting...")
                print(f"{'='*60}")
                import sys
                sys.exit(1)
            
            print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Loaded {len(llm_features_desc)} LLM causal features for label '{label}'")
            config_name = "LLM causal"
            
            # User confirm the configuration (including reverse mapping)
            confirm_result, corl_features = confirm_llm_cd_features_config_with_mapping(
                model_config, label, llm_features_desc, 
                config_file_path, feature_lists, feature_descriptions
            )
            if not confirm_result:
                print(f"\n{Colors.RED}[ERROR]{Colors.RESET} User cancelled, exiting...")
                import sys
                sys.exit(0)
            
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Program terminated: cannot load LLM causal feature configuration")
            print(f"{'='*60}")
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Error details: {str(e)}")
            print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} In LLM_CD_FEATURES mode, a valid LLM causal feature configuration must be provided.")
            print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} The program cannot continue, exiting...")
            print(f"{'='*60}")
            import sys
            sys.exit(1)
    else:
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} The current mode '{prompt_mode}' does not require a feature configuration file")
        
    # Add deprecation warning
    if prompt_mode in ["CORL_FILTERED", "DirectLiNGAM_FILTERED"]:
        print(f"\n{Colors.YELLOW}[WARNING]{Colors.RESET} Note: {prompt_mode} mode will be deprecated in future versions")
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} It is recommended to use the new CD_FILTERED mode with cd_algorithm configuration")
        
    # For modes that require configuration, display the feature filtering summary (CD_FEATURES_OPTIMIZED mode has already been confirmed)
    if prompt_mode in ["CORL_FILTERED", "DirectLiNGAM_FILTERED", "CD_FILTERED"] and corl_features:
        # Display the feature filtering summary (simplified version)
        allowed_features_set = set(corl_features)
        original_diag_count = len(feature_lists['Diag'])
        original_proc_count = len(feature_lists['Proc'])
        original_med_count = len(feature_lists['Med'])
        original_ts_count = len(feature_lists['TS'])
        
        filtered_diag_count = len([f for f in feature_lists['Diag'] if f in allowed_features_set])
        filtered_proc_count = len([f for f in feature_lists['Proc'] if f in allowed_features_set])
        filtered_med_count = len([f for f in feature_lists['Med'] if f in allowed_features_set])
        filtered_ts_count = len([f for f in feature_lists['TS'] if f in allowed_features_set])
        
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} {config_name} feature filtering results:")
        print(f"  Diag: {original_diag_count} -> {filtered_diag_count}")
        print(f"  Proc: {original_proc_count} -> {filtered_proc_count}")
        print(f"  Med: {original_med_count} -> {filtered_med_count}")
        print(f"  TS: {original_ts_count} -> {filtered_ts_count}")
        print(f"  Total: {filtered_diag_count + filtered_proc_count + filtered_med_count + filtered_ts_count}")
    
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Using model: {model_config['display_name']} ({model_config['model_name']})")
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Prediction label: {label}")
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Prompt mode: {prompt_mode}")
    
    # Get the total number of rows in the data
    total_patients = len(data)
    
    # Extract the ground truth values from the data file (specified label column)
    ground_truths_source = data[label].values
    print(f"Extracted {len(ground_truths_source)} '{label}' ground truth labels from the data file")
    
    # Add user confirmation step
    if not DEBUG_MODE and prompt_mode not in ["CD_FEATURES_OPTIMIZED", "LLM_CD_FEATURES"]:
        print("\n" + "="*60)
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Configuration information confirmed")
        print("="*60)
        print(f"Data file: {data_filename}")
        print(f"Number of patients: {total_patients}")
        print(f"Model: {model_config['display_name']} ({model_config['model_name']})")
        print(f"API type: {model_config.get('api_type', 'openai')}")
        print(f"Prediction label: {label}")
        print(f"Prompt mode: {prompt_mode}")
        if prompt_mode == "CORL_FILTERED":
            if corl_features:
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} Number of CORL features: {len(corl_features)}")
            else:
                print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} CORL configuration: not loaded or empty, using original features")
        elif prompt_mode == "DirectLiNGAM_FILTERED":
            if corl_features:
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} Number of DirectLiNGAM features: {len(corl_features)}")
            else:
                print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} DirectLiNGAM configuration: not loaded or empty, using original features")
        elif prompt_mode == "CD_FILTERED":
            cd_algorithm = model_config.get('cd_algorithm', 'CORL')
            if corl_features:
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} CD algorithm: {cd_algorithm}")
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} Number of {cd_algorithm} features: {len(corl_features)}")
            else:
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} CD algorithm: {cd_algorithm}")
                print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} {cd_algorithm} configuration: not loaded or empty, using original features")
        print("="*60)
        
        # User confirmation
        while True:
            user_input = input("\nConfirm to start processing? (y/n): ").strip().lower()
            if user_input in ['y', 'yes']:
                print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} User confirmed, starting processing...\n")
                break
            elif user_input in ['n', 'no']:
                print(f"{Colors.RED}[ERROR]{Colors.RESET} User cancelled, exiting program.")
                return
            else:
                print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Please enter 'y' or 'n'")
    elif prompt_mode == "CD_FEATURES_OPTIMIZED":
        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} CD feature optimization configuration confirmed, starting processing...\n")
    elif prompt_mode == "LLM_CD_FEATURES":
        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} LLM causal feature configuration confirmed, starting processing...\n")
    
    # Store prediction results
    death_probabilities = []  # Store death probabilities
    death_predictions = []    # Store binary prediction results
    ground_truths = []        # Store ground truth values
    experiment_logs = []      # Store detailed experiment logs
    
    print("\n" + "="*50)
    if DEBUG_MODE:
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Starting single-threaded debug processing of {total_patients} patients...")
    else:
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Starting parallel processing of {total_patients} patients (max_workers=10)...")
    print("="*50 + "\n")
    
    if DEBUG_MODE:
        # Debug mode: single-threaded execution
        print(f"{Colors.WHITE}[INFO]{Colors.RESET} Debug mode: single-threaded sequential processing, detailed output\n")
        
        # Initialize result lists
        icustay_ids = []
        patient_ids = []
        
        for idx, row in data.iterrows():
            groundtruth = int(ground_truths_source[idx])
            print(f"\n{'='*60}")
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} Starting processing patient {idx+1}/{total_patients}")
            print(f"{'='*60}")
            
            result = predict_single_patient(
                (idx, row, groundtruth), 
                model_config, 
                feature_descriptions, 
                label, 
                feature_lists,
                corl_features  # Pass preloaded CORL configuration
            )
            
            if result:
                icustay_ids.append(result['icustay_id'])
                patient_ids.append(result['patient_id'])
                death_probabilities.append(result['probability'])
                death_predictions.append(result['prediction'])
                ground_truths.append(result['groundtruth'])
                experiment_logs.append(result['experiment_log'])
                
                print(f"\n{Colors.GREEN}[SUCCESS]{Colors.RESET} Patient {idx+1} processed")
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} Result: probability={result['probability']}, prediction={result['prediction']}, groundtruth={result['groundtruth']}")
            else:
                print(f"\n{Colors.RED}[ERROR]{Colors.RESET} Patient {idx+1} processing failed")
        
        print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Debug mode completed, processed {len(data)} patients")
        
        # Debug mode success statistics
        successful_predictions = sum(1 for prob in death_probabilities if prob != -1)
        failed_tasks = len(death_probabilities) - successful_predictions
        
    else:
        # Normal mode: parallel processing
        # Create thread pool and parallel process patient predictions
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit each patient's prediction task
            futures = []
            for idx, row in data.iterrows():
                groundtruth = int(ground_truths_source[idx])
                future = executor.submit(
                    predict_single_patient, 
                    (idx, row, groundtruth), 
                    model_config, 
                    feature_descriptions, 
                    label, 
                    feature_lists,
                    corl_features  # Pass preloaded CORL configuration
                )
                futures.append(future)
            
            # Process each task's result
            results = []
            completed_count = 0
            total_futures = len(futures)
            
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} Waiting for {total_futures} parallel tasks to complete...")
            
            # Collect all task results (add global and single-task timeout control)
            import time
            start_time = time.time()
            global_timeout = 3000  # Global timeout 50 minutes
            single_task_timeout = 120  # Single-task timeout 2 minutes
            
            print(f"{Colors.WHITE}[INFO]{Colors.RESET} Set timeout: global {global_timeout//60} minutes, single-task {single_task_timeout} seconds")
            
            # Timeout protection logic: real-time save + graceful timeout
            try:
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} Starting processing...")
                for future in as_completed(futures, timeout=global_timeout):
                    try:
                        result = future.result(timeout=single_task_timeout)  # Single-task timeout control
                        results.append(result)  # Real-time collect results
                        completed_count += 1
                        
                        # Display progress and time information
                        elapsed_time = time.time() - start_time
                        if completed_count % 50 == 0 or completed_count == total_futures:
                            print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Completed {completed_count}/{total_futures} patients (time: {elapsed_time:.1f}s)")
                            
                    except Exception as e:
                        # Single-task exception, create default result but do not affect other tasks
                        completed_count += 1
                        if completed_count % 50 == 0:
                            print(f"{Colors.RED}[ERROR]{Colors.RESET} Task exception: {completed_count}/{total_futures}")
                        
                        # Improved: try harder to get the real ICUSTAY_ID
                        future_idx = None
                        icustay_id = None
                        groundtruth = -1
                        
                        # Method 1: find the corresponding future position directly from the futures list
                        try:
                            future_idx = next(i for i, f in enumerate(futures) if f == future)
                            print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Found index through future matching: {future_idx}")
                        except:
                            print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Cannot find index through future matching")
                        
                        # Method 2: use completed_count as a backup index
                        if future_idx is None:
                            future_idx = completed_count - 1
                            print(f"{Colors.WHITE}[INFO]{Colors.RESET} Use completed_count as backup index: {future_idx}")
                        
                        # Method 3: try to get ICUSTAY_ID and groundtruth from the original data
                        if future_idx is not None:
                            try:
                                if 0 <= future_idx < len(data):
                                    row_data = data.iloc[future_idx]
                                    icustay_id = str(int(float(row_data['ICUSTAY_ID'])))
                                    groundtruth = int(ground_truths_source[future_idx])
                                    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Successfully obtained real data: ICUSTAY_ID={icustay_id}, groundtruth={groundtruth}")
                                else:
                                    print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Index out of bounds: {future_idx} >= {len(data)}")
                            except Exception as data_error:
                                print(f"{Colors.RED}[ERROR]{Colors.RESET} Failed to get real data: {str(data_error)}")
                        
                        # Method 4: if still cannot get the real ID, use a more conservative placeholder strategy
                        if icustay_id is None:
                            # Try to use the index of the unprocessed task
                            remaining_indices = [i for i in range(len(data)) if not any(r['idx'] == i for r in results)]
                            if remaining_indices:
                                fallback_idx = remaining_indices[0]  # Use the first unprocessed index
                                try:
                                    row_data = data.iloc[fallback_idx]
                                    icustay_id = str(int(float(row_data['ICUSTAY_ID'])))
                                    groundtruth = int(ground_truths_source[fallback_idx])
                                    future_idx = fallback_idx
                                    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Use unprocessed index as backup: {fallback_idx} -> ICUSTAY_ID={icustay_id}")
                                except:
                                    pass
                        
                        # Last fallback strategy: use descriptive placeholder
                        if icustay_id is None:
                            icustay_id = f"EXCEPTION_{future_idx if future_idx is not None else completed_count-1}"
                            print(f"{Colors.RED}[ERROR]{Colors.RESET} Final use placeholder: {icustay_id}")
                        
                        # Create default result for exception task
                        exception_result = {
                            'idx': future_idx if future_idx is not None else completed_count - 1,
                            'icustay_id': icustay_id,
                            'patient_id': icustay_id,
                            'probability': -1,
                            'prediction': -1,
                            'groundtruth': groundtruth,
                            'experiment_log': {
                                "patient_id": icustay_id,
                                "icustay_id": icustay_id,
                                "model": model_config["display_name"],
                                "prompt_mode": prompt_mode,
                                "input": f"TASK_EXCEPTION: {str(e)}",
                                "response": f"TASK_EXCEPTION: {str(e)}",
                                "answer": -1,
                                "groundtruth": groundtruth,
                                "correctness": -1
                            }
                        }
                        results.append(exception_result)
                        
            except Exception as timeout_or_other_error:
                # Catch timeout or other global exceptions
                elapsed_time = time.time() - start_time
                print(f"\n{Colors.RED}[ERROR]{Colors.RESET} Caught exception ({type(timeout_or_other_error).__name__}): {str(timeout_or_other_error)}")
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} Elapsed time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
                print(f"{Colors.WHITE}[INFO]{Colors.RESET} Starting to protect {len(results)} completed results...")
        
        # State statistics and missing completion (outside the with block)
        total_elapsed = time.time() - start_time
        completed_indices = {r['idx'] for r in results}
        missing_indices = [i for i in range(total_futures) if i not in completed_indices]
        
        print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Task status summary:")
        print(f"Total tasks: {total_futures}")
        print(f"Completed tasks: {len(results)}")
        print(f"Missing tasks: {len(missing_indices)}")
        print(f"Total time: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)")
        
        # Create default results for missing tasks
        if missing_indices:
            print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Creating default results for {len(missing_indices)} missing tasks...")
            for i in missing_indices:
                # Improved: try harder to get the real ICUSTAY_ID
                icustay_id = None
                groundtruth = -1
                
                # Method 1: directly get from the original data
                try:
                    if 0 <= i < len(data):
                        row_data = data.iloc[i]
                        icustay_id = str(int(float(row_data['ICUSTAY_ID'])))
                        groundtruth = int(ground_truths_source[i])
                        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Task {i}: successfully get real data ICUSTAY_ID={icustay_id}")
                    else:
                        print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Task {i}: index out of bounds, data length={len(data)}")
                except Exception as e:
                    print(f"{Colors.RED}[ERROR]{Colors.RESET} Task {i}: failed to get real data: {str(e)}")
                
                # Method 2: if method 1 fails, try other ways
                if icustay_id is None:
                    try:
                        # Try to use the safer way of iloc
                        if i < len(data):
                            row = data.iloc[i:i+1]  # Get single row DataFrame
                            if not row.empty and 'ICUSTAY_ID' in row.columns:
                                icustay_val = row['ICUSTAY_ID'].iloc[0]
                                if pd.notna(icustay_val):
                                    icustay_id = str(int(float(icustay_val)))
                                    if i < len(ground_truths_source):
                                        groundtruth = int(ground_truths_source[i])
                                    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Task {i}: successfully get real data ICUSTAY_ID={icustay_id}")
                    except Exception as e2:
                        print(f"{Colors.RED}[ERROR]{Colors.RESET} Task {i}: alternative method also failed: {str(e2)}")
                
                # Method 3: last fallback strategy
                if icustay_id is None:
                    # Try to infer the pattern from the already processed results
                    try:
                        processed_ids = [r['icustay_id'] for r in results if r['icustay_id'] and not r['icustay_id'].startswith(('MISSING_', 'TIMEOUT_', 'EXCEPTION_'))]
                        if processed_ids:
                            # If there are valid IDs, we know the data structure is normal, the problem may be in the index
                            print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Task {i}: detected {len(processed_ids)} valid IDs, data structure should be normal")
                        
                        # As a last resort, generate a descriptive placeholder, but with more information
                        icustay_id = f"MISSING_{i}_OF_{len(data)}"
                        print(f"{Colors.RED}[ERROR]{Colors.RESET} Task {i}: use informative placeholder: {icustay_id}")
                    except:
                        icustay_id = f"MISSING_{i}"
                        print(f"{Colors.RED}[ERROR]{Colors.RESET} Task {i}: use basic placeholder: {icustay_id}")
                
                missing_result = {
                    'idx': i,
                    'icustay_id': icustay_id,
                    'patient_id': icustay_id,
                    'probability': -1,  # Default value: failure
                    'prediction': -1,   # Default value: failure
                    'groundtruth': groundtruth,
                    'experiment_log': {
                        "patient_id": icustay_id,
                        "icustay_id": icustay_id,
                        "model": model_config["display_name"],
                        "prompt_mode": prompt_mode,
                        "input": "TIMEOUT_OR_NOT_PROCESSED",
                        "response": "TIMEOUT_OR_NOT_PROCESSED",
                        "answer": -1,
                        "groundtruth": groundtruth,
                        "correctness": -1
                    }
                }
                results.append(missing_result)
                
            print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Filled all missing results")
        
        # Sort results by original index to ensure data consistency
        results.sort(key=lambda x: x['idx'])
        
        # Final verification
        final_successful = len([r for r in results if r.get('probability', -1) != -1])
        final_failed = len(results) - final_successful
        
        print(f"\n{Colors.GREEN}[SUCCESS]{Colors.RESET} Final result verification:")
        print(f"Total results: {len(results)} (should be {total_futures})")
        print(f"Successful predictions amount: {final_successful}")
        print(f"Failed tasks amount: {final_failed}")
        print(f"Data integrity: {f'{Colors.GREEN}[SUCCESS]{Colors.RESET} Passed' if len(results) == total_futures else f'{Colors.RED}[ERROR]{Colors.RESET} Exception'}")
        
        # Extract sorted results to the final list
        icustay_ids = []
        patient_ids = []
        for result in results:
            icustay_ids.append(result['icustay_id'])
            patient_ids.append(result['patient_id'])
            death_probabilities.append(result['probability'])
            death_predictions.append(result['prediction'])
            ground_truths.append(result['groundtruth'])
            experiment_logs.append(result['experiment_log'])
        
        # Final statistics (based on actual results)
        successful_predictions = sum(1 for prob in death_probabilities if prob != -1)
        failed_tasks = len(death_probabilities) - successful_predictions
    
    print("\n" + "="*50)
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Successfully processed {successful_predictions}/{total_patients} patients")
    print("="*50)
    
    # Create results folder (if not exist)
    results_dir = os.path.join(script_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Generate unique identifier (UUID)
    unique_id = str(uuid.uuid4())[:8]

    # Generate file name with version prefix, model name, label, prompt mode and unique identifier
    model_short_name = model_config["display_name"].replace(".", "_").replace("-", "_")
    prompt_mode_short = prompt_mode.replace("_", "").lower()  # Simplify prompt mode name
    
    # Determine CD algorithm information (for file name)
    cd_algorithm_suffix = ""
    if prompt_mode in ["CD_FILTERED", "CD_FEATURES_OPTIMIZED", "LLM_CD_FEATURES"]:
        cd_algorithm = model_config.get("cd_algorithm", "UNKNOWN")
        cd_algorithm_suffix = f"_{cd_algorithm.lower()}"
    elif prompt_mode == "CORL_FILTERED":
        cd_algorithm_suffix = "_corl"
    elif prompt_mode == "DirectLiNGAM_FILTERED":
        cd_algorithm_suffix = "_directlingam"
    
    # Generate file name
    if prompt_mode == "CD_FEATURES_OPTIMIZED":
        # CD_FEATURES_OPTIMIZED mode keeps the original special format: {label}_{cd_algorithm}_optimized_{model}_{prompt}_{id}
        cd_algorithm = model_config.get("cd_algorithm", "UNKNOWN")
        cd_algorithm_lower = cd_algorithm.lower()
        csv_file = os.path.join(results_dir, f'{version_prefix}{label}_{cd_algorithm_lower}_optimized_{model_short_name}_{prompt_mode_short}_{unique_id}.csv')
        json_file = os.path.join(results_dir, f'{version_prefix}{label}_{cd_algorithm_lower}_optimized_{model_short_name}_{prompt_mode_short}_{unique_id}.json')
        txt_file = os.path.join(results_dir, f'{version_prefix}{label}_{cd_algorithm_lower}_optimized_{model_short_name}_{prompt_mode_short}_{unique_id}.txt')
    else:
        # Other modes use standard format, but include CD algorithm information (if applicable)
        # File name format: {version_prefix}{label}_predict_results_{model}_{prompt}{cd_algorithm}_{id}.csv
        csv_file = os.path.join(results_dir, f'{version_prefix}{label}_predict_results_{model_short_name}_{prompt_mode_short}{cd_algorithm_suffix}_{unique_id}.csv')
        json_file = os.path.join(results_dir, f'{version_prefix}{label}_experiment_logs_{model_short_name}_{prompt_mode_short}{cd_algorithm_suffix}_{unique_id}.json')
        txt_file = os.path.join(results_dir, f'{version_prefix}{label}_metrics_{model_short_name}_{prompt_mode_short}{cd_algorithm_suffix}_{unique_id}.txt')
    
    # Save prediction results to CSV (include ICUSTAY_ID and ground_truth columns)
    results_df = pd.DataFrame({
        'icustay_id': icustay_ids,  # Add ICUSTAY_ID column
        'patient_id': patient_ids,  # Add unique patient_id column
        f'{label.lower()}_prediction': death_predictions,
        f'{label.lower()}_probability': death_probabilities,
        'ground_truth': ground_truths
    })
    results_df.to_csv(csv_file, index=False)
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Saved prediction results to: {csv_file}")
    
    # Save detailed experiment logs to JSON
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(experiment_logs, f, ensure_ascii=False, indent=2)
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Saved detailed experiment logs to: {json_file}")
    
    # Filter out successful predictions for performance evaluation
    valid_indices = [i for i, (prob, pred) in enumerate(zip(death_probabilities, death_predictions)) 
                     if prob != -1 and pred != -1]
    
    if len(valid_indices) > 0:
        valid_probabilities = [death_probabilities[i] for i in valid_indices]
        valid_predictions = [death_predictions[i] for i in valid_indices]
        valid_ground_truths = [ground_truths[i] for i in valid_indices]
        
        # Evaluate performance (only use successful predictions)
        metrics = evaluate_predictions(
            np.array(valid_ground_truths),
            np.array(valid_predictions),
            np.array(valid_probabilities)
        )
        
        # Print evaluation results
        print(f"\nModel performance evaluation (based on {len(valid_indices)} successful predictions):")
        print("-"*40)
        print(f"Total tasks:        {len(death_predictions)}")
        print(f"Successful predictions:      {len(valid_indices)}")
        print(f"Failed tasks:      {len(death_predictions) - len(valid_indices)}")
        print(f"Success rate:          {len(valid_indices)/len(death_predictions)*100:.1f}%")
        print("-"*40)
        print(f"F1%:             {metrics['F1%']:>8.2f}%")
        print(f"AUROC:           {metrics['AUROC']:>8.4f}")
        print(f"AUPRC:           {metrics['AUPRC']:>8.4f}")
        print("-"*40)

        # Determine CD algorithm information (for txt file record)
        cd_algorithm_info = ""
        if prompt_mode in ["CD_FILTERED", "CD_FEATURES_OPTIMIZED", "LLM_CD_FEATURES"]:
            cd_algorithm = model_config.get("cd_algorithm", "UNKNOWN")
            cd_algorithm_info = f"CD algorithm: {cd_algorithm}\n"
        elif prompt_mode == "CORL_FILTERED":
            cd_algorithm_info = f"CD algorithm: CORL\n"
        elif prompt_mode == "DirectLiNGAM_FILTERED":
            cd_algorithm_info = f"CD algorithm: DirectLiNGAM\n"

        # Save metrics to txt file
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f"Model performance evaluation results\n")
            f.write(f"="*50 + "\n")
            f.write(f"Running time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Unique identifier: {unique_id}\n")
            f.write(f"Model: {model_config['display_name']} ({model_config['model_name']})\n")
            f.write(f"Prediction label: {label}\n")
            f.write(f"Prompt mode: {prompt_mode}\n")
            f.write(cd_algorithm_info)  # Add CD algorithm information
            f.write(f"\nTask statistics:\n")
            f.write(f"Total tasks: {len(death_predictions)}\n")
            f.write(f"Successful predictions: {len(valid_indices)}\n")
            f.write(f"Failed tasks: {len(death_predictions) - len(valid_indices)}\n")
            f.write(f"Success rate: {len(valid_indices)/len(death_predictions)*100:.1f}%\n")
            f.write(f"\nPerformance metrics:\n")
            f.write(f"F1%: {metrics['F1%']:.2f}%\n")
            f.write(f"AUROC: {metrics['AUROC']:.4f}\n")
            f.write(f"AUPRC: {metrics['AUPRC']:.4f}\n")
            f.write(f"\nFile information:\n")
            f.write(f"CSV result file: {os.path.basename(csv_file)}\n")
            f.write(f"JSON log file: {os.path.basename(json_file)}\n")
            f.write(f"Metrics file: {os.path.basename(txt_file)}\n")

        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Saved performance metrics to: {txt_file}")
    else:
        print(f"\n{Colors.RED}[ERROR]{Colors.RESET} All prediction tasks failed, cannot perform performance evaluation")
        print("-"*40)

        # Determine CD algorithm information (for txt file record in case of failure)
        cd_algorithm_info = ""
        if prompt_mode in ["CD_FILTERED", "CD_FEATURES_OPTIMIZED", "LLM_CD_FEATURES"]:
            cd_algorithm = model_config.get("cd_algorithm", "UNKNOWN")
            cd_algorithm_info = f"CD algorithm: {cd_algorithm}\n"
        elif prompt_mode == "CORL_FILTERED":
            cd_algorithm_info = f"CD algorithm: CORL\n"
        elif prompt_mode == "DirectLiNGAM_FILTERED":
            cd_algorithm_info = f"CD algorithm: DirectLiNGAM\n"

        # Even if there are no successful predictions, save basic information to txt file
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f"Model performance evaluation results\n")
            f.write(f"="*50 + "\n")
            f.write(f"Running time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Unique identifier: {unique_id}\n")
            f.write(f"Model: {model_config['display_name']} ({model_config['model_name']})\n")
            f.write(f"Prediction label: {label}\n")
            f.write(f"Prompt mode: {prompt_mode}\n")
            f.write(cd_algorithm_info)  # Add CD algorithm information
            f.write(f"\nTask statistics:\n")
            f.write(f"Total tasks: {len(death_predictions)}\n")
            f.write(f"Successful predictions: 0\n")
            f.write(f"Failed tasks: {len(death_predictions)}\n")
            f.write(f"Success rate: 0.0%\n")
            f.write(f"\nPerformance metrics:\n")
            f.write(f"F1%: cannot be calculated (all tasks failed)\n")
            f.write(f"AUROC: cannot be calculated (all tasks failed)\n")
            f.write(f"AUPRC: cannot be calculated (all tasks failed)\n")
            f.write(f"\nFile information:\n")
            f.write(f"CSV result file: {os.path.basename(csv_file)}\n")
            f.write(f"JSON log file: {os.path.basename(json_file)}\n")
            f.write(f"Metrics file: {os.path.basename(txt_file)}\n")

        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Saved basic information to: {txt_file}")

    # Data integrity check
    print(f"\n{Colors.WHITE}[INFO]{Colors.RESET} Data integrity check...")
    
    # Check all patient_id formats
    invalid_ids = []
    placeholder_ids = []
    valid_ids = []
    
    for i, pid in enumerate(patient_ids):
        try:
            # Try to convert to number to verify format
            float(pid)
            valid_ids.append((i, pid))
        except (ValueError, TypeError):
            # Check if it is a known placeholder format
            if any(pid.startswith(prefix) for prefix in ['MISSING_', 'TIMEOUT_', 'EXCEPTION_', 'UNKNOWN_']):
                placeholder_ids.append((i, pid))
            else:
                invalid_ids.append((i, pid))
    
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Data format check results:")
    print(f"Valid IDs: {len(valid_ids)}")
    print(f"Placeholder IDs: {len(placeholder_ids)}")
    print(f"Invalid IDs: {len(invalid_ids)}")
    
    # Calculate data quality metrics
    total_count = len(patient_ids)
    valid_rate = len(valid_ids) / total_count * 100 if total_count > 0 else 0
    placeholder_rate = len(placeholder_ids) / total_count * 100 if total_count > 0 else 0
    
    print(f"{Colors.WHITE}[INFO]{Colors.RESET} Data quality metrics:")
    print(f"Valid rate: {valid_rate:.1f}% ({len(valid_ids)}/{total_count})")
    print(f"Placeholder rate: {placeholder_rate:.1f}% ({len(placeholder_ids)}/{total_count})")
    
    # If placeholder rate is too high, give a warning
    if placeholder_rate > 10:  # If more than 10% are placeholders, it may indicate API issues
        print(f"{Colors.RED}[WARNING]{Colors.RESET} High placeholder rate ({placeholder_rate:.1f}%), may indicate API issues")
    
    # If there are invalid formats
    if invalid_ids:
        print(f"{Colors.RED}[ERROR]{Colors.RESET} Detected {len(invalid_ids)} invalid patient_id formats")
    
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} Data integrity check completed\n")

if __name__ == "__main__":
    main()
