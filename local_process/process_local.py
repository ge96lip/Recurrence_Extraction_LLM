
import os
import glob 
import pandas as pd
import argparse
import json
import asyncio
import aiohttp
import yaml
from pathlib import Path
import time
from datetime import datetime
import math
import traceback
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import ast
import re
from utils import extract_mod_reports, extract_single_reports, extract_last_json_from_text
MODELS = [
    {"name": "Meditron", "filename": "models/meditron-7b.Q4_K_M.gguf"},
    {"name": "Med-Gemma", "filename": "models/medgemma-4b-it-Q4_K_M.gguf"},
    {"name": "Qwen3-Thinking", "filename": "models/Qwen3-4B-Thinking-2507-Q4_K_M.gguf"},
    {"name": "Llama3.1-8B", "filename": "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"},
    {"name": "GLM-4.5-Air", "filename": "models/GLM-4.5-Air-Q4_K_M/Q4_K_M/GLM-4.5-Air-Q4_K_M-00001-of-00002.gguf"},
    {"name": "Qwen3-Instruct", "filename": "models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"},
]

# Utility functions from utils.py
def is_empty_string_nan_or_none(variable) -> bool:
    """
    Check if the input variable is None, an empty string, a string containing only whitespace or '?', or a NaN float value.
    """
    if variable is None:
        return True
    if isinstance(variable, str):
        stripped = variable.strip()
        if stripped == "" or stripped == "?" or variable.isspace():
            return True
        return False
    if isinstance(variable, float) and math.isnan(variable):
        return True
    if isinstance(variable, int) or isinstance(variable, bool):
        return False
    if isinstance(variable, list):
        return len(variable) == 0
    # If variable is not a recognized type (dict, list with items, etc.), return False
    return False


# Core LLM processing class adapted from routes.py
@dataclass
class TextProcessor:
    model_name: str
    api_model: bool
    prompt: str
    system_prompt: str = "You are a helpful assistant that helps extract information from medical reports."
    temperature: float = 0.1
    grammar: str = ""
    json_schema: str = ""
    n_predict: int = 8192
    chat_endpoint: bool = False
    debug: bool = False
    llamacpp_port: int = 2929
    top_k: int = 30
    top_p: float = 0.9
    seed: int = 42

    # OpenAI client placeholder
    openai_client: Optional[Any] = None

    def __post_init__(self):
        if self.api_model and self.openai_client is None:
            print("Warning: OpenAI client not initialized for API model")

    async def fetch_chat_result_openai(self, session: aiohttp.ClientSession, prompt_formatted: str) -> dict:
        """Fetch result from OpenAI-compatible API using chat endpoint"""
        if self.openai_client is None:
            raise ValueError("OpenAI client not initialized")

        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt_formatted}
            ],
            "temperature": self.temperature,
            "max_tokens": self.n_predict
        }

        if self.json_schema and self.json_schema not in ["", None, " ", "  "]:
            data["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "schema": self.json_schema,
                    "strict": True
                }
            }

        # Simulate OpenAI call, for real use self.openai_client
        # response = await asyncio.to_thread(self.openai_client.chat.completions.create, **data)
        # return {"choices": [{"message": {"content": response.choices[0].message.content}}]}

        # For now, return a mock response
        return {"choices": [{"message": {"content": "Mock API response"}}]}

    async def fetch_chat_result_local(self, session: aiohttp.ClientSession, prompt_formatted: str) -> dict:
        """Fetch result from local llama.cpp server using chat endpoint"""
        url = f"http://localhost:{self.llamacpp_port}/v1/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": "Bearer no-key"}

        data = {
            "model": "llmaix",
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt_formatted}
            ],
            "seed": self.seed,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p
        }

        if self.grammar and self.grammar not in ["", None, " ", "  "]:
            data["grammar"] = self.grammar

        if self.json_schema and self.json_schema not in ["", None, " ", "  "]:
            data["json_schema"] = self.json_schema

        async with session.post(url=url, headers=headers, json=data, 
                              timeout=aiohttp.ClientTimeout(total=20*60)) as response:
            result = await response.json()
            
            # Check for error responses
            if "error" in result:
                raise ValueError(f"Server error: {result.get('error', {}).get('message', 'Unknown error')}")
            
            # Check if response has expected format
            if "choices" not in result:
                raise ValueError(f"Unexpected response format from server: {result}")
                
            return result

    async def fetch_completion_result(self, session: aiohttp.ClientSession, prompt_formatted: str) -> dict:
        """Fetch result from local llama.cpp server using completion endpoint"""
        json_data = {
            "prompt": prompt_formatted,
            "n_predict": self.n_predict,
            "temperature": self.temperature,
            "cache_prompt": True,
            "seed": self.seed,
            "top_k": self.top_k,
            "top_p": self.top_p
        }

        if self.grammar and self.grammar not in ["", None, " ", "  "]:
            json_data["grammar"] = self.grammar

        if self.json_schema and self.json_schema not in ["", None, " ", "  "]:
            json_data["json_schema"] = self.json_schema

        async with session.post(f"http://localhost:{self.llamacpp_port}/completion", 
                              json=json_data, 
                              timeout=aiohttp.ClientTimeout(total=20*60)) as response:
            return await response.json()

    async def process_text(self, session: aiohttp.ClientSession, text: str, text_id: str) -> Dict[str, Any]:
        """Process a single text and return the result"""
        if is_empty_string_nan_or_none(text):
            print(f"SKIPPING EMPTY TEXT for ID {text_id}")
            return {"id": text_id, "content": "", "error": "Empty text", "report": text}
        
        prompt_formatted = self.prompt.format(report=text)
        
        try:
            if self.chat_endpoint:
                if self.api_model:
                    result = await self.fetch_chat_result_openai(session, prompt_formatted)
                    content = result["choices"][0]["message"]["content"]
                else:
                    result = await self.fetch_chat_result_local(session, prompt_formatted)
                    content = result["choices"][0]["message"]["content"]
            else:
                if self.api_model:
                    raise NotImplementedError("OpenAI-compatible API does not support non-chat completion")
                else:
                    result = await self.fetch_completion_result(session, prompt_formatted)
                    content = result.get("content", "")
            
            return {
                "id": text_id, 
                "content": content, 
                "prompt": prompt_formatted,
                # "report": text,  # Add original text
                "raw_result": result if self.debug else None
            }
        except Exception as e:
            print(f"Error processing text {text_id}: {e}")
            return {"id": text_id, "content": "", "error": str(e), "report": text}


    async def process_all_texts(self, texts: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Process all texts concurrently"""
        semaphore = asyncio.Semaphore(10)  # Limit concurrent requests

        async def process_with_semaphore(session, text_data):
            async with semaphore:
                return await self.process_text(session, text_data["text"], text_data["id"])

        async with aiohttp.ClientSession() as session:
            tasks = [process_with_semaphore(session, text_data) for text_data in texts]
            return await asyncio.gather(*tasks, return_exceptions=True)

def postprocess_grammar(results, debug=False):
    """
    Mirror the original postprocess_grammar function exactly
    """
    print("POSTPROCESSING")
    
    extracted_data = []
    error_count = 0
    
    # Create a dictionary format similar to original
    result = {}
    for r in results:
        if isinstance(r, Exception) or r.get('error'):
            continue
        result[r['id']] = {
            'content': r.get('content', ''),
            'report': r.get('report', '')  # We'll need to add this
        }
    
    # Iterate over each report and its associated data
    for i, (id, info) in enumerate(result.items()):
        print(f"Processing report {i+1} of {len(result)}")
        
        # Extract the content of the first field
        content = info["content"]
        
        # Parse the content string into a dictionary
        try:
            # Clean up content
            if content.endswith("<|eot_id|>"):
                content = content[: -len("<|eot_id|>")]
            if content.endswith("</s>"):
                content = content[: -len("</s>")]
            
            # Extract all JSON objects from the content
            json_objects = []
            # Find all {...} patterns
            brace_count = 0
            start_idx = -1
            
            for idx, char in enumerate(content):
                if char == '{':
                    if brace_count == 0:
                        start_idx = idx
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and start_idx != -1:
                        # Found a complete JSON object
                        json_str = content[start_idx:idx+1]
                        json_objects.append(json_str)
                        start_idx = -1
            
            print(f"Found {len(json_objects)} JSON object(s) in output")
            
            # Use the last JSON object if available
            if json_objects:
                content = json_objects[-1]
                print(f"Using last JSON object for parsing")
            
            # Clean the selected JSON string
            content = content.replace("\n","")
            content = content.replace("\r","")
            content = content.replace("\\", "")
            content = re.sub(r',\s*}', '}', content) # remove trailing comma
            
            try:
                info_dict_raw = json.loads(content)
            except Exception:
                try:
                    content = content.replace(" null,", '').replace(' "null",', "")
                    info_dict_raw = json.loads(content)
                except Exception as e:
                    print(f"Failed to parse LLM output. Did you set --n_predict too low or is the input too long? Maybe you can try to lower the temperature a little. ({content=})", flush=True)
                    print("RAW LLM OUTPUT: '" + info["content"] + "'", flush=True)
                    print("Error:", e, flush=True)
                    print("TRACEBACK:", traceback.format_exc(), flush=True)
                    info_dict_raw = {}
                    error_count += 1
            
            info_dict = {}
            for key, value in info_dict_raw.items():
                if is_empty_string_nan_or_none(value):
                    info_dict[key] = ""
                elif isinstance(value, (list, dict)):
                    # Convert lists and dicts to JSON strings
                    info_dict[key] = json.dumps(value)
                else:
                    info_dict[key] = str(value)
                    
        except Exception as e:
            print(f"Failed to parse LLM output. Did you set --n_predict too low or is the input too long? Maybe you can try to lower the temperature a little. (Output: {content=})", flush=True)
            print("Error:", e, flush=True)
            print("TRACEBACK:", traceback.format_exc(), flush=True)
            print(f"Will ignore the error for report {i} and continue.", flush=True)
            info_dict = {}
            error_count += 1
        
        # Create basic metadata (simplified since we don't have all the original metadata)
        metadata = {
            "processing_date": datetime.now().isoformat(),
            "model": "llmaix",  # or use actual model name
            "llm_processing": {
                "temperature": 0.0,  # Use actual values from processor
                "model": "llmaix"
            }
        }
        
        # Construct a dictionary containing the report and extracted information
        extracted_info = {
            # "report": info["report"],
            "id": id,
            "metadata": json.dumps(metadata),
        }
        for key, value in info_dict.items():
            extracted_info[key] = value
        
        # Append the extracted information to the list
        extracted_data.append(extracted_info)
    
    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(extracted_data)
    return df, error_count

def load_config(config_path: str) -> dict:
    """Load YAML configuration file"""
    if not os.path.exists(config_path):
        print(f"Warning: Config file {config_path} not found")
        return {"models": []}

    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
def load_prompts_config(prompts_file: str) -> dict:
    """Load prompts configuration from YAML file"""
    if not os.path.exists(prompts_file):
        print(f"Warning: Prompts file {prompts_file} not found, using defaults")
        return {
            'system_prompt': 'You are a helpful assistant.',
            'user_prompt': 'Extract information from the following text:\n\n{report}',
            'grammar': ''
        }
    
    with open(prompts_file, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    return {
        'system_prompt': config.get('system_prompt', 'You are a helpful assistant.'),
        'user_prompt': config.get('user_prompt', 'Extract information from the following text:\n\n{report}'),
        'grammar': config.get('grammar', '')
    }


def read_text_files(input_dir: str) -> List[Dict[str, str]]:
    """Read all .txt files from input directory"""
    txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
    texts = []

    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()

            file_id = os.path.splitext(os.path.basename(txt_file))[0]
            texts.append({
                "id": file_id,
                "text": content,
                "filename": txt_file
            })
        except Exception as e:
            print(f"Error reading {txt_file}: {e}")

    return texts

def write_full_history(input_dir: str): 
    """combine all RAD reports to one report """
    txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
    full_history = ""

    for i, txt_file in enumerate(txt_files):
        if i < 30: 
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    full_history += f"\n\n--- Report: {i+1} Date: {i} ---\n\n{content}"
            except Exception as e:
                print(f"Error reading {txt_file}: {e}")

    return full_history

def save_results_to_csv(results: List[Dict[str, Any]], output_file: str):
    """Save processing results to CSV file"""
    # Process results to handle exceptions
    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            processed_results.append({
                "id": "error",
                "content": "",
                "error": str(result)
            })
        else:
            processed_results.append(result)

    df = pd.DataFrame(processed_results)

    # Add metadata
    df["processing_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")

def extract_content(res):
    """
    General-purpose JSON extraction from model response.
    Handles markdown formatting, multiple JSON blocks, and various output formats.
    Works with any JSON schema - not specific to surgeries/diagnoses.
    """
    try:
        content = res.get('content', '')
        
        if isinstance(content, dict):
            # Already parsed - return as-is with metadata
            return {
                'patient_id': res.get('id', 'unknown'),
                'extracted_data': content,
                'raw_content': json.dumps(content)
            }
            
        elif isinstance(content, str):
            # Strategy 1: Find all JSON objects in the text using brace-matching
            json_objects = []
            brace_count = 0
            start_idx = -1
            
            for idx, char in enumerate(content):
                if char == '{':
                    if brace_count == 0:
                        start_idx = idx
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and start_idx != -1:
                        json_str = content[start_idx:idx+1]
                        try:
                            # Try to parse this JSON object
                            parsed_obj = json.loads(json_str)
                            json_objects.append(parsed_obj)
                        except json.JSONDecodeError:
                            # Try cleaning common issues
                            cleaned = json_str.replace('\n', '').replace('\r', '').replace('\\', '')
                            cleaned = re.sub(r',\s*}', '}', cleaned)  # Remove trailing commas
                            cleaned = re.sub(r',\s*]', ']', cleaned)  # Remove trailing commas in arrays
                            try:
                                parsed_obj = json.loads(cleaned)
                                json_objects.append(parsed_obj)
                            except:
                                pass
                        start_idx = -1
            
            # Use the last valid JSON object if found
            parsed = None
            if json_objects:
                parsed = json_objects[-1]
                print(f"✓ Extracted JSON via brace-matching (found {len(json_objects)} objects, using last)")
            else:
                # Strategy 2: Try extracting from markdown code blocks
                code_block_pattern = r'```json\s*(.*?)\s*```'
                matches = re.findall(code_block_pattern, content, re.DOTALL)
                
                if matches:
                    # Try to parse the last code block
                    for match in reversed(matches):
                        try:
                            parsed = json.loads(match.strip())
                            print(f"✓ Extracted JSON from markdown code block")
                            break
                        except:
                            continue
            
            if not parsed:
                print(f"WARNING: No valid JSON found in content for {res.get('id', 'unknown')}")
                print(f"Content preview: {content[:300]}...")
                return None
            
            # Return general structure that works with ANY schema
            # For backwards compatibility with surgery/diagnosis extraction:
            if 'surgeries' in parsed or 'diagnoses' in parsed:
                result = {
                    'patient_id': res.get('id', 'unknown'),
                    'reasoning': parsed.get('reasoning', ''),
                    'surgeries': parsed.get('surgeries', []),
                    'surgery_count': parsed.get('surgery_count', len(parsed.get('surgeries', []))),
                    'diagnoses': parsed.get('diagnoses', []),
                    'diagnosis_count': parsed.get('diagnosis_count', len(parsed.get('diagnoses', []))),
                    'raw_content': content,
                    'extracted_data': parsed  # Full JSON for reference
                }
                print(f"✓ Extracted: {result['surgery_count']} surgeries, {result['diagnosis_count']} diagnoses")
            else:
                # Generic extraction for any other schema
                result = {
                    'patient_id': res.get('id', 'unknown'),
                    'extracted_data': parsed,  # The actual JSON data
                    'raw_content': content
                }
                # Log what was extracted
                if isinstance(parsed, dict):
                    field_summary = ', '.join(f"{k}" for k in parsed.keys())
                    print(f"✓ Extracted fields: {field_summary}")
            
            return result
            
        else:
            print(f"WARNING: Content is neither dict nor string: {type(content)}")
            return None
        
    except Exception as e:
        print(f"ERROR in extract_content for ID {res.get('id', 'unknown')}: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        print(f"Content preview: {str(res.get('content', ''))[:300]}...")
        return None

async def main():
    parser = argparse.ArgumentParser(description='Process text files with LLM')
    
    # Input/output arguments
    parser.add_argument('--input_dir', default=f'/Users/carlotta/Desktop/Code_MT/data/timeline_ds_STS', help='Directory containing .txt files to process') # data/auxiliary_task_reports/{patient_id}
    parser.add_argument('--patient_id_file', default='patient_ids.yaml', help='YAML file containing patient ID')
    # Model arguments
    parser.add_argument('--model_name', '-m', required=True, 
                       help='Model filename or API model name')
    parser.add_argument('--chat_endpoint', action='store_true', 
                       help='Use chat endpoint instead of completion')
    # default is False
    parser.add_argument('--api_model', action='store_true', 
                       help='Use OpenAI-compatible API instead of local model')

    # LLM parameters
    parser.add_argument('--temperature', '-t', type=float, default=0.1,
                       help='Temperature (0.0-1.0)')
    parser.add_argument('--n_predict', '-n', type=int, default=8192,
                       help='Maximum tokens to predict')
    parser.add_argument('--top_k', type=int, default=30, help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Grammar and schema
    parser.add_argument('--prompts-file', default='./prompts/event_extraction_VIS.yaml', 
                   help='YAML file containing prompts and grammar (default: prompts.yaml)')

    parser.add_argument('--json_schema', help='JSON schema for structured output')

    parser.add_argument('--config_file', default='models/config.yml',
                       help='Model configuration file')
    parser.add_argument('--llamacpp_port', type=int, default=8080,
                       help='llama.cpp server port')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug output')
    
    parser.add_argument('--per_report', action='store_true', 
                       help='Process each report separately instead of combining into full history')
    parser.add_argument('--modality', type=str, default='VIS', help='Modality for report extraction (e.g., RAD, PAT, OPN, PRG)')
    args = parser.parse_args()

    prompts_config = load_prompts_config(args.prompts_file)

    
    system_prompt = prompts_config['system_prompt']
    user_prompt = prompts_config['user_prompt']
    grammar = prompts_config['grammar']
    num_text = 3
    # get patient_id from yaml file
    if not os.path.exists(args.patient_id_file):
        print(f"Error: Patient ID file {args.patient_id_file} not found")
        return 1

    with open(args.patient_id_file, 'r') as f:
        data = yaml.safe_load(f)
        patient_ids = data["patient_id"]

    raw_dir = f"outputs/{args.model_name}"
    OUTPUT_FILE = 'outputs/all_model_results.json'
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    summary = []
    successful = 0
    failed = 0
    # loop all the below logics for each patient_id
    for patient_id in patient_ids:
        events = []
        print(f"Processing patient ID: {patient_id}")
        # Validate input directory
        if not os.path.isdir(args.input_dir):
            print(f"Error: Input directory {args.input_dir} does not exist")
            return 1

        # Read text files
        if args.per_report:
            print(f"Analysing text files from: {args.input_dir} seperately")
            texts = extract_single_reports(args.input_dir, patient_id, args.modality)

            if not texts:
                print("No .txt files found in input directory")
                return 1
            if num_text > len(texts):
                texts = texts[:num_text]

            print(f"Found {len(texts)} text files to process")
        else: 
            # full history 
            print(f"Combining all text files from: {args.input_dir} into one history")
            
            full_history = extract_mod_reports(args.input_dir, patient_id, args.modality)
            # save full history to a text file
            with open(f"data/full_history_{patient_id}_{args.modality}.txt", "w", encoding="utf-8") as f:
                f.write(full_history)

            texts = [{"id": "full_history", "text": full_history, "filename": f"full_history_{patient_id}_{args.modality}.txt"}]

        # Initialize processor
        processor = TextProcessor(
            model_name=args.model_name,
            api_model=args.api_model,
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=args.temperature,
            grammar=grammar or "",
            json_schema=args.json_schema or "",
            n_predict=args.n_predict,
            chat_endpoint=args.chat_endpoint,
            debug=args.debug,
            llamacpp_port=args.llamacpp_port,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed
        )

        print(f"Processing texts with model: {args.model_name}")
        print(f"Using {'chat' if args.chat_endpoint else 'completion'} endpoint")
        print(f"Using {'API' if args.api_model else 'local'} model")

        # Process input
        start_time = time.time()
        
        result = await processor.process_all_texts(texts)
        end_time = time.time()
        
        print(f"Result: \n {result}")
        print(f"Processing completed in {end_time - start_time:.2f} seconds")

        # Save raw model output to text file
        raw_output_file = os.path.join(raw_dir, f"{args.modality}/raw_output_{patient_id}.txt")
        with open(raw_output_file, 'w', encoding='utf-8') as f:
            f.write(f"Raw Model Output for Patient {patient_id}\n")
            f.write("="*80 + "\n\n")
            for res in result:
                if isinstance(res, Exception):
                    f.write(f"ERROR: {str(res)}\n\n")
                else:
                    f.write(f"ID: {res.get('id', 'unknown')}\n")
                    f.write("-"*80 + "\n")
                    f.write(f"CONTENT:\n{res.get('content', '')}\n")
                    f.write("="*80 + "\n\n")
        print(f"Raw model output saved to: {raw_output_file}")

        for res in result:
            if isinstance(res, Exception):
                continue
                
            if isinstance(res, dict) and 'content' in res and not res.get('error'):
                event = extract_content(res)
                if event:
                    events.append(event)
            elif isinstance(res, list):
                # handle nested event lists
                for ev in res:
                    if isinstance(ev, dict) and 'content' in ev and not ev.get('error'):
                        event = extract_content(ev)
                        if event:
                            events.append(event)

        # Update success/failure tracking                    
        if any(not isinstance(res, Exception) and not res.get('error') for res in result):
            successful += 1
        else:
            failed += 1

        # Enhanced summary with new fields
        summary.append({
            "task": "event_extraction_report",
            "model": args.model_name,
            "patient_id": patient_id,
            "total_surgeries": sum(e.get('surgery_count', 0) for e in events),
            "total_diagnoses": sum(e.get('diagnosis_count', 0) for e in events),
            "events_extracted": events
        })

    with open(f"outputs/summary_{args.model_name}.json", "w") as fsum:
        json.dump(summary, fsum, indent=2)

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            all_results = json.load(f)
    else:
        all_results = []

    all_results.extend(summary)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
        # Save results
        # df, error_count = postprocess_grammar(result, debug=args.debug)
        
        # Save the processed DataFrame
        # df.to_csv(f"results/resultsfullhistory{patient_id}.csv", index=False)

        
    # Print summary
    # successful = sum(1 for r in result if not isinstance(r, Exception) and not r.get("error"))
    # failed = len(result) - successful

    print(f"\nSummary:")
    print(f"- Total files: {len(patient_ids)}")
    print(f"- Successful: {successful}")
    print(f"- Failed: {failed}")

    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        exit(exit_code)
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        if "--debug" in os.sys.argv:
            traceback.print_exc()
        exit(1)