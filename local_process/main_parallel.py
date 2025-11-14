#!/usr/bin/env python3
"""
Recurrence Extraction Script for Lung Cancer VIS Notes

Usage:
    python main_parallel.py --model Qwen3-Instruct --parallel 4 --max-vis 300 --prompts ./prompts/recurrence_prompt_1113.yaml --prompt_version 1113 --output ./outputs

    # run full cohort 

Examples:
    # Sequential processing (Mac/single slot)
    python main_parallel.py --model Qwen3-Instruct --max-vis 300
    
    # Parallel processing (A100 with --parallel 4)
    python main_parallel.py --model Qwen3-Instruct --parallel 4 --max-vis 300
    
    # Process specific patients only
    python main_parallel.py --model Qwen3-Instruct  --parallel 4 --max-vis 300 --patient-ids 100072762 100092590 --prompts ./prompts/recurrence_prompt_1113.yaml --prompt_version 11133 --output ./outputs
"""

import argparse
import asyncio
from asyncio.log import logger
import json
import os
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

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
    logger.info("Processing patient_id: %s", patient_id)
    
    # Get patient metadata
    patient_row = patient_meta[patient_meta["EMPI"] == str(patient_id)]
    
    if patient_row.empty:
        logger.error("Patient %s not found in patient_meta!", patient_id)
        return None
    
    first_surgery_date = patient_row['indexSurgery'].iloc[0]
    logger.info("first_surgery_date: %s", first_surgery_date)

    if pd.isna(first_surgery_date) or not first_surgery_date:
        logger.error("Missing first lung surgery date for patient %s", patient_id)
        return None

    vis_reports = extract_single_reports(data_dir, patient_id, 'VIS')
    if not vis_reports:
        logger.error("No VIS reports found for patient %s", patient_id)
        return None
    if max_vis_reports:
        vis_reports = vis_reports[:max_vis_reports]
    
    # Load prompts config
    prompts_config = load_prompts_config(prompts_file)
    system_prompt = prompts_config['system_prompt']
    user_prompt_template = prompts_config['user_prompt']
    
    recurrence_results = []
    user_prompt_template = prompts_config['user_prompt']
    for i, vis_report in enumerate(vis_reports):
        if i % 10 == 0:
            logger.info("Processing VIS report %d/%d for patient %s", i+1, len(vis_reports), patient_id)

        vis_date = vis_report['note_date']
        vis_text = vis_report['text']
        logger.info("VIS %s: date=%s, text_length=%d, preview=%s...", 
                   vis_report['id'], vis_date, len(vis_text), vis_text[:100])
    

        processor = TextProcessor(
            model_name=model_name,
            api_model=False,
            prompt=user_prompt_template,
            # system_prompt=system_prompt,
            temperature=0.1,
            grammar="",
            n_predict=4096,
            chat_endpoint=True,
            debug=False,
            llamacpp_port=llamacpp_port
        )
        
        # texts = [{"id": vis_report['id'], "text": user_prompt}]
        texts = [{
            "id": vis_report['id'],
            "text": vis_text,
            "note_date": vis_date,
            "report_description": f"VIS report for patient after surgery on {first_surgery_date}"
        }]
        logger.info("User prompt length: %d", len(texts))
        try:
            #print(f"texts is: {texts}")
            results = await processor.process_all_texts(texts)
            output_json = None
            print(f"Results: {results}")
            if results and results[0] and not results[0].get('error'):
                result = results[0]
                content = result.get('content', '')
                if '```json' in content:
                    content = content.split('```json')[-1].split('```')[0]
                
                try:
                    output_json = json.loads(content)
                except Exception as e:
                    logger.error("Failed to parse JSON for %s VIS %s: %s", patient_id, vis_report['id'], e)
            else:
                logger.error("Model error for %s VIS %s", patient_id, vis_report['id'])

            result_entry = {
                "vis_id": vis_report['id'],
                "vis_date": vis_date,
                "surgery_date": str(first_surgery_date),
                "output": output_json if output_json else {"error": True}
            }
            recurrence_results.append(result_entry)
        except Exception as e:
            logger.error("Exception for %s VIS %s: %s", patient_id, vis_report['id'], e)

    # Save results
    date = datetime.now().strftime("%m%d")
    output_folder = os.path.join(output_root, "recurrence_task", model_name, prompt_version, str(patient_id))
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, "output.json")
    with open(output_file, 'w') as f:
        json.dump(recurrence_results, f, indent=2, default=str)

    logger.info("Saved patient %s output to %s", patient_id, output_file)
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
    patient_meta = pd.read_csv(args.groundtruth)
    # only keep the EMPI + indexSurgery rows with values
    patient_meta = patient_meta[patient_meta['EMPI'].notna() & patient_meta['indexSurgery'].notna()]
    patient_meta['EMPI'] = patient_meta['EMPI'].astype(int).astype(str)

    # --- logging setup: write logs to outputs/logs/... and to console ---
    log_dir = Path(args.output) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"main_parallel_{args.model}_{args.prompt_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    handlers = [logging.FileHandler(log_file), logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", handlers=handlers)

    # Catch unhandled exceptions and log them
    def _handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    sys.excepthook = _handle_exception
    logger.info("Started run — model=%s prompt_version=%s", args.model, args.prompt_version)
    logger.info("Log file: %s", str(log_file))

    try:
        from hashlib import sha256
        prompts_src = Path(args.prompts)
        dest_folder = Path(args.output) / "recurrence_task" / args.model / str(args.prompt_version)
        dest_folder.mkdir(parents=True, exist_ok=True)
        if prompts_src.exists():
            prompt_bytes = prompts_src.read_bytes()
            prompt_hash = sha256(prompt_bytes).hexdigest()
            # copy canonical prompt content into the run folder with shorthash
            dest_prompt = dest_folder / f"prompt_{prompt_hash[:8]}.yaml"
            if not dest_prompt.exists():
                dest_prompt.write_bytes(prompt_bytes)
            # write metadata
            meta = {
                "prompt_file_original": str(prompts_src.resolve()),
                "prompt_file_copied": str(dest_prompt),
                "prompt_hash": prompt_hash,
                "prompt_version": args.prompt_version,
                "model": args.model,
                "saved_at": datetime.now().isoformat()
            }
            (dest_folder / "prompt_metadata.json").write_text(json.dumps(meta, indent=2))
            logger.info("Saved prompt to %s (sha256=%s)", dest_prompt, prompt_hash[:8])
        else:
            logger.warning("Prompts file not found: %s — skipping prompt save", prompts_src)
    except Exception as e:
        logger.warning("Failed to persist prompt file: %s", e)

    logger.info("Loading patient metadata from: %s", args.groundtruth)

    # Get patient IDs to process
    if args.patient_ids:
        patient_ids = [str(pid) for pid in args.patient_ids]
        logger.info("Processing %d specified patients", len(patient_ids))
    else:
        patient_ids = patient_meta["EMPI"].tolist()
        logger.info("Processing all %d patients", len(patient_ids))    

    if args.skip_existing:
        original_count = len(patient_ids)
        patient_ids_to_process = []
        already_processed = []
        
        for pid in patient_ids:
            # add date to outputs
            date = "1031" #datetime.now().strftime("%m%d")
            output_file = os.path.join(args.output, f"recurrence_task", args.model, args.prompt_version, str(pid), f"output.json")
            logger.debug("output file path: %s", output_file)

            if os.path.exists(output_file):
                already_processed.append(pid)
            else:
                patient_ids_to_process.append(pid)
        
        patient_ids = patient_ids_to_process
        logger.info("Found %d already processed patients (skipping)", len(already_processed))
        logger.info("Remaining patients to process: %d", len(patient_ids))

        if len(already_processed) > 0:
            logger.info("Already processed: %s", already_processed[:5])
    else:
        logger.info("Note: Use --skip-existing to skip already processed patients")

    
    # Determine processing mode
    use_parallel = args.parallel > 1

    logger.info("Configuration:")
    logger.info("  Model: %s", args.model)
    logger.info("  Data dir: %s", args.data_dir)
    logger.info("  Prompts: %s", args.prompts)
    logger.info("  Output: %s", args.output)
    logger.info("  Max VIS reports: %d", args.max_vis)
    logger.info("  Processing mode: %s", 'Parallel' if use_parallel else 'Sequential')
    if use_parallel:
        logger.info("  Parallel slots: %d", args.parallel)
    logger.info("  LlamaCPP port: %d", args.port)
    logger.info("")
    
    # Process patients
    start_time = datetime.now()
    sem = asyncio.Semaphore(args.parallel)

    async def _run_patient(pid):
        async with sem:
            logger.info("Processing patient: %s", pid)
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
                logger.error("ERROR processing patient %s: %s", pid, e)
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
    logger.info("Processing complete!")
    logger.info("Total patients: %d", len(patient_ids))
    logger.info("Total time: %s", elapsed)
    logger.info("Average per patient: %s", elapsed / len(patient_ids))
    logger.info("%s", '='*60)


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
        default="/vast/florian/carlotta/data/gt_automatic_eval_v1.csv",
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
