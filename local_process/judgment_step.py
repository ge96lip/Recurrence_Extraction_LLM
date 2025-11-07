#!/usr/bin/env python3
"""
judgment_step.py

Consolidate VIS-level first-pass model outputs into temporal recurrence episodes
and produce corrected labels / reasoning.
Output will be saved in each patient's folder. 

Usage:
    python judgment_step.py --csv /path/to/patients/ --surgery 2016-04

Inputs (per-patient CSV):
    Event Type,Event Date,Date extracted,Evidence Quote,VIS file,Certainty

Outputs:
    JSON with keys:
      - first_surgery_date
      - recurrence_episodes
      - corrected_labels
      - reasoning_summary
Also prints brief summary to stdout.
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


async def process_patient_judgment(
    empi: str,
    model_name: str,
    prompts_template: str,
    processor: TextProcessor,
    patient_meta: pd.DataFrame,
    output_root: str,
    prompt_version: str,
    max_events: int = None,
):

    print(f"patient_id: {empi}")

    patient_row = patient_meta[patient_meta["EMPI"] == str(empi)]
    if patient_row.empty:
        print(f"ERROR: Patient {empi} not found in patient_meta!")
        return None

    first_surgery_date = patient_row['INDEX_SURGERY'].iloc[0]
    if pd.isna(first_surgery_date) or not first_surgery_date:
        print(f"Missing first lung surgery date for patient {empi}")
        return None

    base_patient_folder = Path(output_root) / "recurrence_task" / model_name / "1031" / str(empi)
    json_path = base_patient_folder / "patient_level_evidence.json"
    csv_path = base_patient_folder / "patient_level_events.csv"
    if not json_path.exists() and not csv_path.exists():
        print(f"No patient_level_evidence.json / patient_level_events.csv for {empi} at {base_patient_folder}, skipping")
        return None

    # Load events (prefer JSON)
    parsed_events = None
    if json_path.exists():
        try:
            raw = json_path.read_text(encoding="utf-8")
            parsed_events = json.loads(raw)
            if isinstance(parsed_events, dict):
                for k in ("events", "timeline", "items"):
                    if k in parsed_events and isinstance(parsed_events[k], list):
                        parsed_events = parsed_events[k]
                        break
        except Exception as e:
            print(f"Warning: failed to parse JSON {json_path} for {empi}: {e}. Will try CSV fallback.")
            parsed_events = None

    
    if not parsed_events:
        print(f"No events loaded for {empi}, skipping")
        return None

    if max_events:
        parsed_events = parsed_events[:max_events]

    events_json_str = json.dumps(parsed_events, indent=2, ensure_ascii=False)
    events_json_block = "\n".join("  " + line for line in events_json_str.splitlines())

    user_prompt = prompts_template.replace("{first_lung_surgery_date}", str(first_surgery_date))
    user_prompt = user_prompt.replace("{events_json}", events_json_block)

    # Send exactly one prompt per patient (no intra-patient parallelism)
    texts = [{"id": f"{empi}", "text": user_prompt}]

    print(f"texts is: {texts}")

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
                print(f"Failed to parse JSON for {empi}")

        output = {
                "patient": empi,
                "surgery_date": str(first_surgery_date),
                "output": output_json if output_json else {"error": True}
            }
    except Exception as e:
        print(f"Exception for {empi}")
        output = {
            "patient": empi,
            "surgery_date": str(first_surgery_date),
            "output": {"error": str(e)}
        }
    

    # save output next to the patient events file
    out_folder = base_patient_folder
    out_folder.mkdir(parents=True, exist_ok=True)
    out_file = out_folder / f"judgment_{prompt_version}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved judgment for {empi} -> {out_file}")
    return {"empi": empi, "status": "ok", "out_file": str(out_file)}

async def run_in_parallel(
    empi_list,
    model_name,
    prompts_template,
    processor: TextProcessor,
    patient_meta,
    outputs_root,
    prompt_version,
    max_events,
    parallel_slots,
):
    sem = asyncio.Semaphore(parallel_slots)
    results = []

    async def worker(empi):
        async with sem:
            return await process_patient_judgment(
                empi=empi,
                model_name=model_name,
                prompts_template=prompts_template,
                processor=processor,
                patient_meta=patient_meta,
                output_root=outputs_root,
                prompt_version=prompt_version,
                max_events=max_events,
            )

    tasks = [asyncio.create_task(worker(empi)) for empi in empi_list]
    for t in asyncio.as_completed(tasks):
        res = await t
        results.append(res)
    return results


def parse_args():
    p = argparse.ArgumentParser(description="Run judgment step on patient-level events using LLM")
    p.add_argument("--model", default="Qwen3-Instruct", help="Model name (e.g. Qwen3-Instruct)")
    p.add_argument("--prompts", required=True, help="Path to judgment prompt YAML (e.g. prompts/judgment_prompt.yaml)")
    p.add_argument("--groundtruth", default="ground_truth_events.csv", help="Patient metadata CSV (contains EMPI and indexSurgery)")
    p.add_argument("--output", default="./outputs", help="Root outputs folder")
    p.add_argument("--prompt_version", default=datetime.now().strftime("%m%d"), help="Prompt/version tag")
    p.add_argument("--patient-ids", nargs="+", default=None, help="Specific patient EMPI(s) to process")
    p.add_argument("--max-events", type=int, default=300, help="Max events rows to include per patient")
    p.add_argument("--parallel", type=int, default=4, help="Parallel slots")
    p.add_argument("--port", type=int, default=8080, help="llama.cpp server port")
    return p.parse_args()

def main():
    args = parse_args()

    patient_meta = pd.read_csv(args.groundtruth, dtype=str)
    # ensure EMPI and INDEX_SURGERY exist
    if 'EMPI' not in patient_meta.columns or 'INDEX_SURGERY' not in patient_meta.columns:
        print("groundtruth CSV must contain EMPI and indexSurgery columns")
        return

    patient_meta['EMPI'] = patient_meta['EMPI'].astype(str)
    prompts_config = load_prompts_config(args.prompts)
    system_prompt = prompts_config.get('system_prompt', '')
    user_prompt_template = prompts_config.get('user_prompt', prompts_config.get('user_prompt_template', ''))

    shared_processor = TextProcessor(
        model_name=args.model,
        api_model=False,
        prompt="{report}",            # placeholder only; we pass full user_prompt as text per patient
        system_prompt=system_prompt,
        temperature=0.1,
        grammar="",
        n_predict=4096,
        chat_endpoint=True,
        debug=False,
        llamacpp_port=args.port
    )

    if args.patient_ids:
        empi_list = [str(x) for x in args.patient_ids]
    else:
        empi_list = patient_meta['EMPI'].astype(str).tolist()

    # only run for the patients who have the patient_level_events.csv file
    outputs_root = Path(args.output)
    available = []
    missing = []
    for empi in empi_list:
        folder = outputs_root / "recurrence_task" / args.model / str(args.prompt_version) / str(empi)
        if (folder / "patient_level_evidence.json").exists() or (folder / "patient_level_events.csv").exists():
            available.append(empi)
        else:
            missing.append(empi)

    if not available:
        print("No patients with patient_level_events.csv found. Exiting.")
        return

    if missing:
        print(f"Skipping {len(missing)} patients without patient_level_events.csv")
    empi_list = available

    print(f"Running judgment for {len(empi_list)} patients (parallel={args.parallel})")

    asyncio.run(run_in_parallel(
        empi_list=empi_list,
        model_name=args.model,
        prompts_template=user_prompt_template,
        processor=shared_processor,
        patient_meta=patient_meta,
        outputs_root=args.output,
        prompt_version=args.prompt_version,
        max_events=args.max_events,
        parallel_slots=args.parallel,
    ))

if __name__ == "__main__":
    main()
