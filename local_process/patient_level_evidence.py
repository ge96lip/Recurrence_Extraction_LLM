#!/usr/bin/env python3
"""
Read patient_level_evidence.csv and write a cleaned, chronological JSON suitable
for the judgment prompt.

Usage:
    python preprocess_patient_events.py /path/to/patient_level_evidence.csv /path/to/output.json
"""
import os
from pathlib import Path
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict

PRIORITY_ORDER = {"NOREC": 0, "SUSP": 1, "REC": 2}
VERSION = "11073"
BASE_DIR = f'...{VERSION}/'

def _find_column(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        for col in df.columns:
            if col.strip().lower() == c.strip().lower():
                return col
    raise KeyError(f"None of the candidate columns found: {candidates}")


def _split_vis_field(vis_field: str) -> List[str]:
    if pd.isna(vis_field):
        return []
    parts = []
    for part in str(vis_field).split("|"):
        p = part.strip()
        if p:
            parts.append(p)
    # remove duplicates while preserving order
    seen = set()
    out = []
    for v in parts:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def process_patient_level_evidence(input_csv: str, output_json: str) -> None:
    p_in = Path(input_csv)
    p_out = Path(output_json)
    # handle empty csv files gracefully
    if p_in.stat().st_size == 0:
        print(f"Input CSV {p_in} is empty. Writing empty JSON to {p_out}.")
        no_events_reasoning = "No events recorded."
        p_out.parent.mkdir(parents=True, exist_ok=True)
        p_out.write_text(json.dumps([no_events_reasoning], indent=2, ensure_ascii=False))
        return
    df = pd.read_csv(p_in, dtype=str, keep_default_na=False)
    
    print(f"Processing {p_in}, {len(df)} rows")
    # map expected columns
    event_col = _find_column(df, ["Event Type", "EventType", "event_type"])
    event_date_col = _find_column(df, ["Event Date", "Event_DATE", "EVENT_DATE", "event_date"])
    extracted_date_col = None
    for cand in ["Date extracted", "Date_extracted", "Date Extracted", "DateExtracted", "DateExtracted"]:
        try:
            extracted_date_col = _find_column(df, [cand])
            break
        except KeyError:
            extracted_date_col = None
    evidence_col = None
    for cand in ["Evidence Quote", "evidence quote", "EvidenceQuote", "Evidence_Quote", "evidence_quote"]:
        try:
            evidence_col = _find_column(df, [cand])
            break
        except KeyError:
            evidence_col = None
    vis_col = None
    for cand in ["VIS file", "VIS_file", "VIS file(s)", "VIS", "VIS file(s)"]:
        try:
            vis_col = _find_column(df, [cand])
            break
        except KeyError:
            vis_col = None
    reasoning_col = None
    for cand in ["Reasoning", "reasoning", "Reason", "Notes"]:
        try:
            reasoning_col = _find_column(df, [cand])
            break
        except KeyError:
            reasoning_col = None
    certainty_col = None
    for cand in ["Certainty", "certainty"]:
        try:
            certainty_col = _find_column(df, [cand])
            break
        except KeyError:
            certainty_col = None

    # normalize dates to YYYY-MM strings (use event_date primarily)
    def to_ym(val: str):
        if val is None or val == "":
            return None
        try:
            dt = pd.to_datetime(val, errors="coerce")
            if pd.isna(dt):
                # try parse YYYY-MM strings directly
                s = str(val).strip()
                if len(s) >= 7 and s[4] == "-":
                    return s[:7]
                return None
            return dt.strftime("%Y-%m")
        except Exception:
            return None

    df["__event_type"] = df[event_col].str.strip().str.upper()
    df["__event_date"] = df[event_date_col].apply(to_ym)
    if extracted_date_col:
        df["__extracted_date"] = df[extracted_date_col].apply(to_ym)
    else:
        df["__extracted_date"] = df["__event_date"]

    df["__evidence"] = df[evidence_col] if evidence_col else ""
    df["__vis_list"] = df[vis_col].apply(_split_vis_field) if vis_col else [[]]
    df["__reasoning"] = df[reasoning_col] if reasoning_col else ""
    df["__certainty"] = df[certainty_col] if certainty_col else ""

    # drop rows without event_date (keep if there is extracted_date)
    df = df[df["__event_date"].notna()].copy()
    if df.empty:
        p_out.write_text("[]")
        return

    # group rows by event_date
    grouped = []
    date_map: Dict[str, List[Dict]] = {}
    for event_date, g in df.groupby("__event_date", sort=True):
        # combine rows by event_type for this date
        rows_by_type: Dict[str, Dict] = {}
        for _, row in g.iterrows():
            et_raw = (row["__event_type"] or "").strip().upper()
            et = et_raw if et_raw else "NOREC"
            if et not in rows_by_type:
                rows_by_type[et] = {
                    "evidence_quotes": [],
                    "vis": [],
                    "certainties": set(),
                    "extracted_dates": [],
                    "reasonings": set(),
                    "count": 0
                }
            rows_by_type[et]["evidence_quotes"].append(str(row["__evidence"]).strip())
            rows_by_type[et]["vis"].extend(row["__vis_list"] if isinstance(row["__vis_list"], list) else [])
            if row["__certainty"]:
                rows_by_type[et]["certainties"].add(str(row["__certainty"]).strip())
            if row["__extracted_date"]:
                rows_by_type[et]["extracted_dates"].append(row["__extracted_date"])
            if row.get("__reasoning"):
                rows_by_type[et]["reasonings"].add(str(row["__reasoning"]).strip())
            rows_by_type[et]["count"] += 1

        # create ordered entries for this date using PRIORITY_ORDER (NOREC, SUSP, REC)
        date_entries: List[Dict] = []
        for et in sorted(rows_by_type.keys(), key=lambda x: PRIORITY_ORDER.get(x, 99)):
            info = rows_by_type[et]
            vis_unique = []
            seen = set()
            for v in info["vis"]:
                if v and v not in seen:
                    seen.add(v)
                    vis_unique.append(v)
            evidence_joined = " | ".join([q for q in info["evidence_quotes"] if q])
            extracted_earliest = min(info["extracted_dates"]) if info["extracted_dates"] else event_date
            if info.get("reasonings"):
                reasoning_text = " | ".join(sorted(info["reasonings"]))
            else:
                reasoning_text = f"{info['count']} row(s); {len(vis_unique)} VIS files; certainties: {', '.join(sorted(info['certainties'])) if info['certainties'] else 'unknown'}"
            date_entries.append({
                "event_type": et,
                "event_date": event_date,
                "extracted_date": extracted_earliest,
                "evidence_quote": evidence_joined,
                "supporting_VIS": vis_unique,
                "reasoning": reasoning_text
            })

        date_map[event_date] = date_entries

    # Now merge consecutive NOREC dates only when the intervening dates contain exclusively NOREC entries.
    final_entries: List[Dict] = []
    buffer_norec: List[Dict] = []

    def flush_norec_buffer():
        if not buffer_norec:
            return
        start = buffer_norec[0]["event_date"]
        end = buffer_norec[-1]["event_date"]
        start_end = f"{start}..{end}" if start != end else start
        combined_quotes = " | ".join([e["evidence_quote"] for e in buffer_norec if e["evidence_quote"]])
        combined_vis = []
        seen = set()
        for e in buffer_norec:
            for v in e.get("supporting_VIS", []):
                if v not in seen:
                    seen.add(v)
                    combined_vis.append(v)
        # prefer explicit reasonings if present, else synthesize
        collected_reasonings = [e.get("reasoning") for e in buffer_norec if e.get("reasoning")]
        if collected_reasonings:
            combined_reasoning = " | ".join(collected_reasonings)
        else:
            combined_reasoning = f"Merged {len(buffer_norec)} NOREC entries spanning {start_end}; {len(combined_vis)} VIS files involved."
        final_entries.append({
            "event_type": "NOREC",
            "event_date": start_end,
            "extracted_date": buffer_norec[0]["extracted_date"],
            "evidence_quote": combined_quotes,
            "supporting_VIS": combined_vis,
            "reasoning": combined_reasoning
        })
        buffer_norec.clear()

    # iterate dates in chronological order
    for date in sorted(date_map.keys()):
        entries = date_map[date]
        # if all entries for this date are NOREC, buffer them for potential merging
        if all(e.get("event_type") == "NOREC" for e in entries):
            # append all NOREC entries for this date (there may be multiple rows aggregated)
            buffer_norec.extend(entries)
            continue

        # date contains at least one non-NOREC -> flush any buffered NOREC interval first,
        # then append this date's entries (which may include a NOREC entry but also other types).
        flush_norec_buffer()
        final_entries.extend(entries)

    # flush remaining NOREC buffer at end
    flush_norec_buffer()

    # final_entries already chronological because grouped iterated grouped by date ascending
    # write JSON
    p_out.parent.mkdir(parents=True, exist_ok=True)
    p_out.write_text(json.dumps(final_entries, indent=2, ensure_ascii=False))
    print(f"Wrote {len(final_entries)} entries to {p_out}")


if __name__ == "__main__":
    import sys
    print("Starting processing of patient level evidence files...")
    print(f"Base directory: {BASE_DIR}")
    for item in os.listdir(BASE_DIR):
        patient_dir = os.path.join(BASE_DIR, item, "patient_level_events.csv")
        output = os.path.join(BASE_DIR, item, "patient_level_evidence.json")
        process_patient_level_evidence(patient_dir, output)