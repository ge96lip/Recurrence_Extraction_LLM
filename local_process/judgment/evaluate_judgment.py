# judgment.json saved per patient as: /vast/florian/carlotta/LLMAIx/rec_task/outputs/recurrence_task/Qwen3-Instruct/{PROMPT_VERSION}/{PATIENT_ID}/judgment_{PROMPTVERSION}.json

# ground truth saved under: /vast/florian/carlotta/LLMAIx/rec_task/ground_truth_events.csv

# only evaluate the REC (+date) and NOREC for now 

"""
for every patient read json
1. if json error -> skip 
2. if json NOREC and groundtruth NOREC -> correct
3. if json NOREC and groundtruth REC -> incorrect
4. if json REC -> check date with groundtruth date note the difference in months 

"""
"""
Evaluate per-patient judgment JSONs against ground truth.

Usage:
  python ./judgment/evaluate_judgment.py --prompt_version 11073 \
    --outputs_root /vast/florian/carlotta/LLMAIx/rec_task/outputs/recurrence_task/Qwen3-Instruct \
    --ground_truth /vast/florian/carlotta/LLMAIx/rec_task/ground_truth_events.csv \
    --out_csv ./judgment/judgment_eval_summary_v2.csv
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outputs_root", required=True, help="Base folder for model outputs (per-prompt folders)")
    p.add_argument("--prompt_version", required=True, help="Prompt version folder name")
    p.add_argument("--ground_truth", required=True, help="CSV with ground truth events (EMPI, EVENT, EVENT_DATE)")
    p.add_argument("--out_csv", default="./judgment_eval_summary.csv", help="CSV to write per-patient summary")
    return p.parse_args()


def read_ground_truth(gt_path: str) -> Dict[str, Dict[str, str]]:
    """Return EMPI -> {'EVENT': 'REC'|'NOREC', 'EVENT_DATE': 'YYYY-MM' or ''}"""
    df = pd.read_csv(gt_path, dtype=str).fillna("")
    # normalize column names
    cols = {c.lower(): c for c in df.columns}
    empi_col = cols.get("empi") or cols.get("patientid") or cols.get("patient_id")
    event_col = cols.get("event")
    date_col = cols.get("event_date") or cols.get("date")
    if not (empi_col and event_col and date_col):
        raise RuntimeError(f"GT missing required columns. Found: {list(df.columns)}")
    df = df[[empi_col, event_col, date_col]].copy()
    df.columns = ["EMPI", "EVENT", "EVENT_DATE"]
    df["EVENT"] = df["EVENT"].str.upper().map({"REC": "REC", "SUSP": "SUSP", "NOREC": "NOREC"}).fillna(df["EVENT"])
    gt_map: Dict[str, Dict[str, str]] = {}
    for empi, group in df.groupby("EMPI"):
        group = group.copy()
        recs = group[group["EVENT"] == "REC"]
        if len(recs):
            # pick earliest parseable REC date
            dates = pd.to_datetime(recs["EVENT_DATE"], errors="coerce")
            if dates.notna().any():
                d = dates[dates.notna()].min()
                date_str = d.strftime("%Y-%m")
            else:
                date_str = str(recs["EVENT_DATE"].iloc[0]) or ""
            gt_map[str(empi)] = {"EVENT": "REC", "EVENT_DATE": date_str}
        else:
            gt_map[str(empi)] = {"EVENT": "NOREC", "EVENT_DATE": ""}
    return gt_map


def normalize_ym(s: Optional[str]) -> str:
    if not s:
        return ""
    try:
        dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt):
            return s
        return dt.strftime("%Y-%m")
    except Exception:
        return s or ""


def months_diff(ym1: str, ym2: str) -> Optional[int]:
    if not ym1 or not ym2:
        return None
    try:
        d1 = pd.to_datetime(ym1, errors="coerce")
        d2 = pd.to_datetime(ym2, errors="coerce")
        if pd.isna(d1) or pd.isna(d2):
            return None
        return abs((d2.year - d1.year) * 12 + (d2.month - d1.month))
    except Exception:
        return None


def extract_pred_from_output(output: Any) -> Tuple[Optional[str], Optional[str]]:
    """
    Heuristic extraction:
      - if output.get('error') -> return (None,None) -> will be skipped
      - if output.recurrence_prediction.has_recurrence -> REC and pick earliest recurrence_months or first timeline REC
      - else search timeline for entry with type == 'REC' or 'NOREC_PHASE'
      - fallback: try keys 'recurrence_prediction','recurrence','prediction' etc.
    Returns (event_label in {"REC","NOREC"} or None, date 'YYYY-MM' or '')
    """
    if output is None:
        return None, None
    if isinstance(output, dict):
        if output.get("error") or output.get("status") == "error":
            return None, None
        # recurrence_prediction section
        rp = output.get("recurrence_prediction")
        if isinstance(rp, dict):
            has = rp.get("has_recurrence")
            months = rp.get("recurrence_months") or rp.get("recurrence_dates") or rp.get("recurrence_month")
            if has is True or (months and len(months) > 0):
                # REC: choose first recurrence month
                if isinstance(months, list) and months:
                    return "REC", normalize_ym(months[0])
                if isinstance(months, str) and months:
                    return "REC", normalize_ym(months)
                return "REC", ""
        # timeline entries
        timeline = output.get("timeline")
        if isinstance(timeline, list):
            # prefer first REC entry
            for item in timeline:
                typ = (item.get("type") or "").upper()
                if typ == "REC":
                    d = item.get("date_or_interval") or item.get("date") or item.get("month")
                    # date_or_interval may be "YYYY-MM" or "YYYY-MM..YYYY-MM"; take first YYYY-MM
                    if isinstance(d, str) and ".." in d:
                        d = d.split("..", 1)[0]
                    return "REC", normalize_ym(d)
            # if no REC, determine if whole timeline is NOREC_PHASE -> NOREC
            for item in timeline:
                typ = (item.get("type") or "").upper()
                if "NOREC" in typ:
                    return "NOREC", ""
        # sometimes top-level keys
        for k in ("prediction", "recurrence", "label", "event"):
            v = output.get(k)
            if isinstance(v, str):
                vs = v.strip().upper()
                if "REC" in vs:
                    return "REC", ""
                if "NOREC" in vs or "NO REC" in vs:
                    return "NOREC", ""
        # scan values for indicators
        for v in output.values():
            if isinstance(v, str):
                vs = v.strip().upper()
                if "REC" == vs or "NOREC" == vs:
                    return ("REC", "") if vs == "REC" else ("NOREC", "")
        # recurse into nested structures
        for v in output.values():
            if isinstance(v, (dict, list)):
                ev, ed = extract_pred_from_output(v)
                if ev:
                    return ev, ed
    elif isinstance(output, list):
        for item in output:
            ev, ed = extract_pred_from_output(item)
            if ev:
                return ev, ed
    return None, None

def evaluate(outputs_root: str, prompt_version: str, gt_map: Dict[str, Dict[str, str]]):
    base = Path(outputs_root) / str(prompt_version)
    # collect all EMPI present in GT and in outputs (so we evaluate every patient)
    outputs_empi = set()
    if base.exists():
        for p in base.iterdir():
            if p.is_dir():
                outputs_empi.add(p.name)
    all_empi = sorted(set(list(gt_map.keys()) + list(outputs_empi)))

    rows: List[Dict[str, Any]] = []
    skipped = 0

    for empi in all_empi:
        # default values for missing GT or prediction
        gt = gt_map.get(empi, {"EVENT": "", "EVENT_DATE": ""})
        gt_event = gt["EVENT"]
        gt_date = normalize_ym(gt["EVENT_DATE"])

        jdir = base / empi
        jfile = jdir / f"judgment_{prompt_version}.json"
        if not jfile.exists():
            # fallback to any judgment_*.json inside folder
            if jdir.exists():
                candidates = sorted(jdir.glob("judgment_*.json"))
            else:
                candidates = []
            if candidates:
                jfile = candidates[0]

        pred_event: Optional[str] = ""
        pred_date: str = ""
        status = "MISSING"  # MISSING / PARSE_ERROR / ERROR / OK / NO_OUTPUT_EXTRACTED

        if not jfile.exists():
            # no prediction file for this patient
            status = "MISSING"
        else:
            try:
                jdata = json.loads(jfile.read_text(encoding="utf-8"))
            except Exception:
                status = "PARSE_ERROR"
                skipped += 1
            else:
                # expected structure: { "patient": "...", "surgery_date": "...", "output": { ... } }
                output = jdata.get("output") if isinstance(jdata, dict) else jdata
                if isinstance(output, dict) and output.get("error"):
                    status = "ERROR"
                else:
                    ev, ed = extract_pred_from_output(output)
                    if ev:
                        pred_event = ev
                        pred_date = normalize_ym(ed or "")
                        status = "OK"
                    else:
                        # no explicit REC/NOREC extracted, still consider OK but note missing extraction
                        pred_event = ""
                        pred_date = ""
                        status = "NO_OUTPUT_EXTRACTED"

        exists_correct = (pred_event == gt_event) if pred_event != "" else False
        month_diff = None
        if pred_event == "REC" and gt_event == "REC":
            month_diff = months_diff(gt_date, pred_date or "")

        rows.append({
            "EMPI": empi,
            "GT_EVENT": gt_event,
            "GT_DATE": gt_date,
            "PRED_EVENT": pred_event or "",
            "PRED_DATE": pred_date or "",
            "EXISTS_CORRECT": exists_correct,
            "MONTH_DIFF": month_diff,
            "STATUS": status,
            "JFILE": str(jfile) if jfile.exists() else ""
        })

    df = pd.DataFrame(rows)
    return df, skipped


def main():
    args = parse_args()
    gt_map = read_ground_truth(args.ground_truth)
    df, skipped = evaluate(args.outputs_root, args.prompt_version, gt_map)
    df.to_csv(args.out_csv, index=False)
    total = len(df)
    correct_norec = int(((df["GT_EVENT"] == "NOREC") & (df["PRED_EVENT"] == "NOREC")).sum())
    wrong_norec = int(((df["GT_EVENT"] == "REC") & (df["PRED_EVENT"] == "NOREC")).sum())
    rec_pairs = df[df["GT_EVENT"] == "REC"]
    mdiffs = rec_pairs["MONTH_DIFF"].dropna().astype(int) if not rec_pairs.empty else pd.Series(dtype=int)
    print(f"Evaluated: {total} patients, skipped (missing/errored): {skipped}")
    print(f"NOREC correct: {correct_norec}, REC missed (pred NOREC): {wrong_norec}")
    print(f"REC pairs with month diffs: {len(mdiffs)}, mean diff (months): {mdiffs.mean() if len(mdiffs) else 'N/A'}")
    print(f"Saved detailed CSV to: {args.out_csv}")


if __name__ == "__main__":
    main()