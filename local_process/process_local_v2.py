

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
    n_predict: int = 512
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
            "max_tokens": self.n_predict, 
            "stop": ["<|im_end|>", "<|eot_id|>", "</s>"]
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
    -            {"role": "system", "content": self.system_prompt},  # ← Remove this!
                {"role": "user", "content": prompt_formatted}
            ],
            "seed": self.seed,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p, 
            "max_tokens": self.n_predict,
            "cache_prompt": False,
            "stop": ["<|im_end|>", "<|eot_id|>", "</s>"] 
        }

        if self.grammar and self.grammar not in ["", None, " ", "  "]:
            data["grammar"] = self.grammar

        if self.json_schema and self.json_schema not in ["", None, " ", "  "]:
            data["json_schema"] = self.json_schema

        async with session.post(url=url, headers=headers, json=data, 
                              timeout=aiohttp.ClientTimeout(total=20*60)) as response:
            text = await response.text()
            response.raise_for_status()
            try:
                return json.loads(text)
            except Exception:
                print("RAW RESPONSE (first 500):", text[:500])
                raise

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

    async def process_text(self, session: aiohttp.ClientSession, text: str, text_id: str, report_description: str = None, note_date: str = None) -> Dict[str, Any]:
        """Process a single text and return the result"""
        if is_empty_string_nan_or_none(text):
            print(f"SKIPPING EMPTY TEXT for ID {text_id}")
            return {"id": text_id, "content": "", "error": "Empty text", "report": text}
        
        # Format the report with metadata
        """formatted_report = ""
        if note_date:
            formatted_report += f"Date: {note_date}\n"
        if report_description:
            formatted_report += f"Description: {report_description}\n"
        formatted_report += f"Report:\n{text}"
        print(f"formatted_report for ID {text_id}:\n{formatted_report[:500]}...\n")"""
        
        prompt_formatted = self.prompt.format(first_lung_surgery_date=note_date,
            report=text)
        
        try:
            if self.chat_endpoint:
                if self.api_model:
                    result = await self.fetch_chat_result_openai(session, prompt_formatted)
                    content = result["choices"][0]["message"]["content"]
                    print(f"OpenAI Chat Result: {content}")
                else:
                    # print(f"Prompt formatted for local chat:\n{prompt_formatted}\n")
                    result = await self.fetch_chat_result_local(session, prompt_formatted)
                    content = result["choices"][0]["message"]["content"]
                    #print(f"Local Chat Result: {content}")
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
        semaphore = asyncio.Semaphore(3)  # Limit concurrent requests

        async def process_with_semaphore(session, text_data):
            # print(f"text data is: {text_data}")
            async with semaphore:
                return await self.process_text(
                    session, 
                    text_data["text"], 
                    text_data["id"], 
                    text_data.get("report_description"),
                    text_data.get("note_date")
                )

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

def extract_content(res, modality):
    
    """
    Extract structured information from model response.
    Handles both new structured format and fallback parsing.
    """
    try:
        content = res.get('content', '')
        if isinstance(content, dict):
            parsed = content
        elif isinstance(content, str):
            # Remove markdown code blocks if present
            content_clean = content.strip()
            if content_clean.startswith('```json'):
                content_clean = content_clean.split('```json', 1)[1]
            if content_clean.startswith('```'):
                content_clean = content_clean.split('```', 1)[1]
            if content_clean.endswith('```'):
                content_clean = content_clean.rsplit('```', 1)[0]
            
            content_clean = content_clean.strip()
            
            
            try:
                parsed = json.loads(content_clean)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract JSON from text
                import re
                json_match = re.search(r'\{.*\}', content_clean, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                else:
                    print(f"WARNING: Could not parse JSON from content")
                    return None
        else:
            parsed = content
        
        # Handle if parsed is a list (array) - extract the first element
        if isinstance(parsed, list):
            if len(parsed) == 0:
                print(f"WARNING: Parsed JSON is an empty array for {res.get('id', 'unknown')}")
                return None
            # Use the first element if it's an array
            parsed = parsed[0]
            print(f"✓ Extracted first element from JSON array")
        
        # Extract fields from new structured format
        if modality == 'VIS':
            event = {
                'patient_id': res.get('id', 'unknown'),
                'reasoning': parsed.get('reasoning', ''),
                'surgeries': parsed.get('surgeries', []),
                'surgery_count': parsed.get('surgery_count', 0),
                'diagnoses': parsed.get('diagnoses', []),
                'diagnosis_count': parsed.get('diagnosis_count', 0),
                'raw_content': content  # Keep raw content for debugging
            }
            
            return event
        elif modality == 'OPN':
            event = {
                'patient_id': res.get('id', 'unknown'),
                'lung_chest_surgery': parsed.get('lung_chest_surgery', False),
                'reasoning': parsed.get('reasoning', ''),
                'procedure_type': parsed.get('procedure_type', ''),
                'date': parsed.get('date', ''),
                'evidence_snippet': parsed.get('evidence_snippet', ''),
                'confidence': parsed.get('confidence', ''),
                'raw_content': content  # Keep raw content for debugging
            }
            return event
        else:
            print(f"Modality {modality} not supported yet.")
            return None

    except Exception as e:
        print(f"ERROR in extract_content: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return None

