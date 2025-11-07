#!/usr/bin/env python3
"""
Recurrence Extraction Script for Lung Cancer VIS Notes

Usage:
    python main_parallel.py --model Qwen3-Instruct --parallel 4 --max-vis 300 --prompts ./prompts/recurrence_prompt_1107.yaml --prompt_version 11073 --output ./outputs

Examples:
    # Sequential processing (Mac/single slot)
    python main_parallel.py --model Qwen3-Instruct --max-vis 300
    
    # Parallel processing (A100 with --parallel 4)
    python main_parallel.py --model Qwen3-Instruct --parallel 4 --max-vis 300
    
    # Process specific patients only
    python main_parallel.py --model Qwen3-Instruct --patient-ids 104352354 107373506
"""

import argparse
import asyncio
import json
import os
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add local_process to path
sys.path.append('/vast/florian/carlotta/LLMAIx/local_process')

from process_local import TextProcessor, load_prompts_config
from utils import extract_single_reports


async def process_patient_recurrence_task(
    patient_id: str, 
    data_dir: str, 
    prompts_file: str, 
    prompt_version: str, 
    patient_meta: pd.DataFrame, 
    output_root: str, 
    max_vis_reports: int = None,
    model_name: str = "Qwen3-Instruct",
    llamacpp_port: int = 8080
):
    """Process each VIS report for recurrence extraction (sequential)."""
    print(f"patient_id: {patient_id}")
    
    # Get patient metadata
    patient_row = patient_meta[patient_meta["EMPI"] == str(patient_id)]
    
    if patient_row.empty:
        print(f"ERROR: Patient {patient_id} not found in patient_meta!")
        return None
    
    first_surgery_date = patient_row['indexSurgery'].iloc[0]
    print(f"first_surgery_date: {first_surgery_date}")
    
    if pd.isna(first_surgery_date) or not first_surgery_date:
        print(f"Missing first lung surgery date for patient {patient_id}")
        return None

    vis_reports = extract_single_reports(data_dir, patient_id, 'VIS')
    if not vis_reports:
        print(f"No VIS reports found for patient {patient_id}")
        return None
    if max_vis_reports:
        vis_reports = vis_reports[:max_vis_reports]
    
    # Load prompts config
    prompts_config = load_prompts_config(prompts_file)
    system_prompt = prompts_config['system_prompt']
    user_prompt_template = prompts_config['user_prompt']

    shared_processor = TextProcessor(
            model_name=model_name,
            api_model=False,
            prompt="{report}",
            system_prompt=system_prompt,
            temperature=0.1,
            grammar="",
            n_predict=4096,
            chat_endpoint=True,
            debug=False,
            llamacpp_port=llamacpp_port
        )
    
    recurrence_results = []
    for i, vis_report in enumerate(vis_reports):
        if i % 10 == 0:
            print(f"  Processing VIS report {i+1}/{len(vis_reports)} for patient {patient_id}")
        
        vis_date = vis_report['note_date']
        vis_text = vis_report['text']
        
        user_prompt = user_prompt_template.format(
            first_lung_surgery_date=first_surgery_date,
            vis_note_text=vis_text
        )
        
        texts = [{"id": vis_report['id'], "text": user_prompt}]
        try:
            results = await shared_processor.process_all_texts(texts)
            output_json = None
            if results and results[0] and not results[0].get('error'):
                result = results[0]
                content = result.get('content', '')
                if '```json' in content:
                    content = content.split('```json')[-1].split('```')[0]
                
                try:
                    output_json = json.loads(content)
                except Exception as e:
                    print(f"Failed to parse JSON for {patient_id} VIS {vis_report['id']}: {e}")
            else:
                print(f"Model error for {patient_id} VIS {vis_report['id']}")
            
            result_entry = {
                "vis_id": vis_report['id'],
                "vis_date": vis_date,
                "surgery_date": str(first_surgery_date),
                "output": output_json if output_json else {"error": True}
            }
            recurrence_results.append(result_entry)
        except Exception as e:
            print(f"Exception for {patient_id} VIS {vis_report['id']}: {e}")

    # Save results
    date = datetime.now().strftime("%m%d")
    output_folder = os.path.join(output_root, "recurrence_task", model_name, prompt_version, str(patient_id))
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, "output.json")
    with open(output_file, 'w') as f:
        json.dump(recurrence_results, f, indent=2, default=str)
    
    print(f"Saved patient {patient_id} output to {output_file}")
    return recurrence_results


async def process_patient_parallel(
    patient_id: str, 
    data_dir: str, 
    prompts_file: str, 
    prompt_version: str,
    patient_meta: pd.DataFrame, 
    output_root: str, 
    max_vis_reports: int = None, 
    max_concurrent: int = 4,
    model_name: str = "Qwen3-Instruct",
    llamacpp_port: int = 8080
):
    """Process VIS reports in parallel batches (for GPU with --parallel > 1)."""
    print(f"patient_id: {patient_id}")
    
    # Get patient metadata
    patient_row = patient_meta[patient_meta["EMPI"] == str(patient_id)]
    if patient_row.empty:
        print(f"ERROR: Patient {patient_id} not found in patient_meta!")
        return None
    
    first_surgery_date = patient_row['indexSurgery'].iloc[0]
    if pd.isna(first_surgery_date) or not first_surgery_date:
        print(f"Missing first lung surgery date for patient {patient_id}")
        return None

    vis_reports = extract_single_reports(data_dir, patient_id, 'VIS')
    if not vis_reports:
        print(f"No VIS reports found for patient {patient_id}")
        return None
    if max_vis_reports:
        vis_reports = vis_reports[:max_vis_reports]
    
    print(f"Processing {len(vis_reports)} VIS reports in batches of {max_concurrent}")
    
    # Load prompts config once
    prompts_config = load_prompts_config(prompts_file)
    system_prompt = prompts_config['system_prompt']
    user_prompt_template = prompts_config['user_prompt']
    
    async def process_single_vis(vis_report):
        """Process a single VIS report"""
        vis_date = vis_report['note_date']
        vis_text = vis_report['text']
        
        user_prompt = user_prompt_template.format(
            first_lung_surgery_date=first_surgery_date,
            vis_note_text=vis_text
        )
        
        processor = TextProcessor(
            model_name=model_name,
            api_model=False,
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.1,
            grammar="",
            n_predict=4096,
            chat_endpoint=True,
            debug=False,
            llamacpp_port=llamacpp_port
        )
        
        texts = [{"id": vis_report['id'], "text": vis_text}]
        try:
            results = await processor.process_all_texts(texts)
            output_json = None
            if results and results[0] and not results[0].get('error'):
                result = results[0]
                content = result.get('content', '')
                if '```json' in content:
                    content = content.split('```json')[-1].split('```')[0]
                try:
                    output_json = json.loads(content)
                except Exception as e:
                    print(f"Failed to parse JSON for {patient_id} VIS {vis_report['id']}: {e}")
            
            return {
                "vis_id": vis_report['id'],
                "vis_date": vis_date,
                "surgery_date": str(first_surgery_date),
                "output": output_json if output_json else {"error": True}
            }
        except Exception as e:
            print(f"Exception for {patient_id} VIS {vis_report['id']}: {e}")
            return {
                "vis_id": vis_report['id'],
                "vis_date": vis_date,
                "surgery_date": str(first_surgery_date),
                "output": {"error": str(e)}
            }
    
    # Process VIS reports in parallel batches
    recurrence_results = []
    for i in range(0, len(vis_reports), max_concurrent):
        batch = vis_reports[i:i + max_concurrent]
        print(f"  Processing VIS batch {i//max_concurrent + 1}/{(len(vis_reports)-1)//max_concurrent + 1} ({len(batch)} reports)")
        
        # Process batch concurrently
        batch_results = await asyncio.gather(*[process_single_vis(vis) for vis in batch])
        recurrence_results.extend(batch_results)
    
    # Save results
    date = "1031" # datetime.now().strftime("%m%d")
    output_folder = os.path.join(output_root, "recurrence_task", model_name, prompt_version, str(patient_id))
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, "output.json")
    with open(output_file, 'w') as f:
        json.dump(recurrence_results, f, indent=2, default=str)
    
    print(f"Saved patient {patient_id} output to {output_file}")
    return recurrence_results


async def clear_kv_cache(llamacpp_port: int = 8080):
    """Clear the KV cache by sending a request to the llama.cpp server"""
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            url = f"http://localhost:{llamacpp_port}/v1/chat/completions"
            payload = {
                "messages": [{"role": "user", "content": "clear"}],
                "max_tokens": 1,
                "cache_prompt": False
            }
            async with session.post(url, json=payload) as response:
                await response.json()
        print("KV cache cleared")
    except Exception as e:
        print(f"Warning: Could not clear KV cache: {e}")


async def main(args):
    """Main processing function"""
    
    # Load patient metadata
    print(f"Loading patient metadata from: {args.groundtruth}")
    patient_meta = pd.read_csv(args.groundtruth)
    patient_meta['EMPI'] = patient_meta['EMPI'].astype(str)
    
    # Get patient IDs to process
    if args.patient_ids:
        patient_ids = [str(pid) for pid in args.patient_ids]
        print(f"Processing {len(patient_ids)} specified patients")
    else:
        patient_ids = patient_meta["EMPI"].astype(str).tolist()
        print(f"Processing all {len(patient_ids)} patients")
    
    if args.skip_existing:
        original_count = len(patient_ids)
        patient_ids_to_process = []
        already_processed = []
        
        for pid in patient_ids:
            # add date to outputs
            date = "1031" #datetime.now().strftime("%m%d")
            output_file = os.path.join(args.output, f"recurrence_task", args.model, args.prompt_version, str(pid), f"output.json")
            print(f"output file path: {output_file}")
            if os.path.exists(output_file):
                already_processed.append(pid)
            else:
                patient_ids_to_process.append(pid)
        
        patient_ids = patient_ids_to_process
        print(f"Found {len(already_processed)} already processed patients (skipping)")
        print(f"Remaining patients to process: {len(patient_ids)}")
        
        if len(already_processed) > 0:
            print(f"Already processed: {already_processed[:5]}{'...' if len(already_processed) > 5 else ''}")
    else:
        print(f"Note: Use --skip-existing to skip already processed patients")
    
    
    # Determine processing mode
    use_parallel = args.parallel > 1
    
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Prompts: {args.prompts}")
    print(f"  Output: {args.output}")
    print(f"  Max VIS reports: {args.max_vis}")
    print(f"  Processing mode: {'Parallel' if use_parallel else 'Sequential'}")
    if use_parallel:
        print(f"  Parallel slots: {args.parallel}")
    print(f"  LlamaCPP port: {args.port}")
    print()
    
    # Process patients
    start_time = datetime.now()
    sem = asyncio.Semaphore(args.parallel)

    async def _run_patient(pid):
        async with sem:
            print(f"\n{'='*60}")
            print(f"Processing patient: {pid}")
            print(f"{'='*60}\n")
            try:
                # keep per-patient processing sequential (no intra-patient parallelism)
                await process_patient_recurrence_task(
                    patient_id=pid,
                    data_dir=args.data_dir,
                    prompts_file=args.prompts,
                    prompt_version=args.prompt_version,
                    patient_meta=patient_meta,
                    output_root=args.output,
                    max_vis_reports=args.max_vis,
                    model_name=args.model,
                    llamacpp_port=args.port
                )
            except Exception as e:
                print(f"❌ ERROR processing patient {pid}: {e}")
            finally:
                # clear KV cache occasionally to avoid growing context
                # (do this per finished patient to spread the clearing)
                try:
                    await clear_kv_cache(llamacpp_port=args.port)
                except Exception:
                    pass

    tasks = [asyncio.create_task(_run_patient(pid)) for pid in patient_ids]
    # await all; exceptions inside _run_patient are caught there, so gather won't raise
    await asyncio.gather(*tasks)

    # Summary
    elapsed = datetime.now() - start_time
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total patients: {len(patient_ids)}")
    print(f"Total time: {elapsed}")
    print(f"Average per patient: {elapsed / len(patient_ids)}")
    print(f"{'='*60}\n")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Extract recurrence events from VIS notes using LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Sequential processing (Mac/single slot)
            python main_parallel.py --model Qwen3-Instruct --max-vis 300
            
            # Parallel processing (A100 with --parallel 4)
            python main_parallel.py --model Qwen3-Instruct --parallel 4 --max-vis 300 --skip-existing
            
            # Skip already processed patients
            python main_parallel.py --model Qwen3-Instruct --skip-existing
  
            # Process specific patients
            python main_parallel.py --model Qwen3-Instruct --patient-ids 104352354 107373506
                    """
                )
    
    # Required arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., Qwen3-Instruct, Llama-3.1-8B)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/vast/florian/carlotta/data/timeline_ds_STS",
        help="Path to patient data directory (default: %(default)s)"
    )
    
    parser.add_argument(
        "--prompts",
        type=str,
        default=f"/vast/florian/carlotta/LLMAIx/rec_task/recurrence_prompt.yaml",
        help="Path to prompts YAML file (default: %(default)s)"
    )
    
    parser.add_argument(
        "--groundtruth",
        type=str,
        default="/vast/florian/carlotta/data/groundtruth_recurrence.csv",
        help="Path to ground truth CSV with patient metadata (default: %(default)s)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs",
        help="Output directory for results (default: %(default)s)"
    )
    
    parser.add_argument(
        "--max-vis",
        type=int,
        default=300,
        help="Maximum VIS reports to process per patient (default: all)"
    )
    
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel slots (should match llama-server --parallel, default: 1)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="LlamaCPP server port (default: 8080)"
    )
    
    parser.add_argument(
        "--patient-ids",
        type=str,
        nargs='+',
        default=None,
        help="Specific patient IDs to process (default: all patients)"
    )
    parser.add_argument(
        "--skip-existing",
        action='store_true',
        help="Skip patients that already have output files"
    )
    parser.add_argument(
        "--prompt_version",
        type=str,
        default=datetime.now().strftime("%m%d"),
        help="Version of the prompt used (default: %(default)s)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    asyncio.run(main(args))
