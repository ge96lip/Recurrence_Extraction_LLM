import pandas as pd
from pathlib import Path
import re

# ----------------------------
# CONFIGURATION
# ----------------------------
INCLUDE_NOREC = False  # Set to True to include NO_RECURRENCE events in predictions

# ----------------------------
# Helpers
# ----------------------------
def to_ym_str(x):
    """
    Parse various date formats to 'YYYY-MM' string.
    Tries strict YYYY-MM first, then general parsing.
    Returns None if parsing fails.
    """
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    # Try strict YYYY-MM
    try:
        dt = pd.to_datetime(s, format="%Y-%m", errors="raise")
        return dt.strftime("%Y-%m")
    except Exception:
        pass
    # Try generic parse (handles e.g. 7/2/18, 2020-07-15, etc.)
    dt = pd.to_datetime(s, errors="coerce", dayfirst=False)
    if pd.isna(dt):
        # Try day-first as a last resort
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if pd.isna(dt):
        return None
    return dt.to_period("M").strftime("%Y-%m")


def extract_patient_id_from_path(path_obj: Path):
    # parent folder name is the patient/EMPI (e.g., 'patient_id')
    return path_obj.parent.name

def clean_event_type(x):
    """
    Normalize event types to the set {RECURRENCE, SUSPICIOUS, NO_RECURRENCE}.
    Maps variants to standard names.
    """
    if pd.isna(x):
        return None
    s = str(x).strip().upper()
    
    # Map variants to standard names
    if s in {"REC", "RECURRENCE"}:
        return "REC"
    elif s in {"SUSP", "SUSPICIOUS", "SUSPICIOUS_EVENT"}:
        return "SUSP"
    elif s in {"NOREC", "NO_RECURRENCE", "NO_RECURRENCE_EVENT"}:
        return "NOREC"

    # Unknown types -> None
    return None


def apply_event_priority(pred_df):
    """
    When multiple events exist for the same patient and date (within same month),
    keep only the highest priority event: RECURRENCE > SUSPICIOUS > NO_RECURRENCE
    """
    # Define priority mapping
    priority_map = {
        'REC': 1,
        'SUSP': 2,
        'NOREC': 3
    }
    
    # Add priority column
    pred_df['priority'] = pred_df['EventType'].map(priority_map)
    
    # Group by patient and date, keep the event with highest priority (lowest number)
    deduplicated = pred_df.sort_values('priority').groupby(['EMPI', 'Event_DATE']).first().reset_index()
    
    # Remove the temporary priority column
    deduplicated = deduplicated.drop(columns=['priority'])
    
    return deduplicated


# ----------------------------
# 1) Build PRED from per-patient CSVs
# ----------------------------
def build_pred():
    base = Path("/vast/florian/carlotta/LLMAIx/rec_task/outputs/recurrence_task/Qwen3-Instruct/1031")
    csv_paths = sorted(base.glob("*/patient_level_events.csv"))

    pred_rows = []
    for p in csv_paths:
        empi = extract_patient_id_from_path(p)
        df = pd.read_csv(p, dtype=str).rename(columns=lambda c: c.strip())
        # Expected columns: Event Type, Event Date, Date extracted, Evidence Quote, VIS file, Certainty
        if "Event Type" not in df.columns or "Event Date" not in df.columns:
            raise ValueError(f"Missing required columns in {p}")
        df["EventType"] = df["Event Type"].map(clean_event_type)
        df["Event_DATE"] = df["Event Date"].map(to_ym_str)
        
        df["patient_id"] = empi
        # Keep useful originals too
        keep_cols = [
            "patient_id",
            "EventType",
            "Event_DATE",
            "Date extracted",
            "Evidence Quote",
            "VIS file",
            "Certainty"
        ]
        # Some columns may be absent; select the intersection
        keep_cols = [c for c in keep_cols if c in df.columns]
        pred_rows.append(df[keep_cols])

    pred = pd.concat(pred_rows, ignore_index=True)
    
    # Sanity: drop rows without month or without event type
    pred = pred[~pred["Event_DATE"].isna()].copy()
    pred = pred[~pred["EventType"].isna()].copy()
    
    # Rename patient_id to EMPI to align with GT
    pred = pred.rename(columns={"patient_id": "EMPI"})
    
    print(f"Total events before filtering: {len(pred)}")
    print(f"Event distribution before filtering:\n{pred['EventType'].value_counts()}\n")
    
    # Apply priority deduplication BEFORE filtering
    # This ensures REC > SUSP > NOREC when multiple events exist for same patient-date
    pred = apply_event_priority(pred)
    print(f"Events after priority deduplication: {len(pred)}")
    print(f"Event distribution after deduplication:\n{pred['EventType'].value_counts()}\n")
    
    # Filter to REC and SUSP only (optional via configuration)
    if INCLUDE_NOREC:
        event_types_to_keep = ["REC", "SUSP", "NOREC"]
        print("Including NOREC events in predictions")
    else:
        event_types_to_keep = ["REC", "SUSP"]
        print("Filtering to REC and SUSP events only")

    pred = pred[pred["EventType"].isin(event_types_to_keep)].copy()
    
    print(f"Final events after filtering: {len(pred)}")
    print(f"Final event distribution:\n{pred['EventType'].value_counts()}\n")
    
    # Save
    pred.to_csv("predicted_events.csv", index=False)
    print(f"Saved predictions to predicted_events.csv")
    
    return pred


# ----------------------------
# 2) Build GT from your registry-like file
# ----------------------------
def build_gt(): 
    path="/Users/carlotta/Desktop/Code_MT/data/aux_vis/GroundTruth/gt_automatic_eval.xlsx"
    gt_raw = pd.read_excel(path, dtype=str).rename(columns=lambda c: c.strip())

    # Normalize column names we rely on:
    # Try to find columns by case-insensitive match
    def find_col(df, candidates):
        cols_lower = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand.lower() in cols_lower:
                return cols_lower[cand.lower()]
        raise ValueError(f"None of the candidate columns {candidates} found in GT file. Columns are: {list(df.columns)}")

    empi_col = find_col(gt_raw, ["EMPI", "empi", "PatientID", "patient_id"])
    etype_col = find_col(gt_raw, ["Event", "event", "EVENT"])
    edate_col = find_col(gt_raw, ["Event_DATE", "EVENT Date", "EVENT_DATE", "Date"])
    eindexsurgery_col = find_col(gt_raw, ["indexSurgery", "INDEXSURGERY", "Index_Surgery_Date"])    
    # Forward-fill the EMPI (and optionally other meta columns) to cover rows with blanks
    gt_raw[empi_col] = gt_raw[empi_col].ffill()
    gt_raw[eindexsurgery_col] = gt_raw[eindexsurgery_col].ffill()  

    # Keep only rows that actually represent an event (REC or SUSP) and have a date
    gt = gt_raw[[empi_col, eindexsurgery_col, etype_col, edate_col]].copy()
    gt.columns = ["EMPI", "INDEX_SURGERY", "EVENT", "EVENT_DATE"]
    gt["EVENT"] = gt["EVENT"].map(clean_event_type)
    gt["EVENT_DATE"] = gt["EVENT_DATE"].map(to_ym_str)
    gt["INDEX_SURGERY"] = gt["INDEX_SURGERY"].map(to_ym_str)
    # Map to standard names for consistency
    gt["EVENT"] = gt["EVENT"].map({
        "REC": "REC",
        "SUSP": "SUSP",
        "NOREC": "NOREC"
    })

    # Keep only REC and SUSP events (ground truth typically doesn't include NOREC)
    gt = gt[gt["EVENT"].isin(["REC", "SUSP"])].copy()
    gt = gt[~gt["EVENT_DATE"].isna()].copy()

    # Optional: drop duplicates
    gt = gt.drop_duplicates(subset=["EMPI", "EVENT", "EVENT_DATE"]).reset_index(drop=True)

    # Save if you want
    gt.to_csv("ground_truth_events.csv", index=False)

    print("gt shape:", gt.shape)
    print(f"First three GT rows:\n{gt.head(3)}\n")
    print(f"GT event distribution:\n{gt['EVENT'].value_counts()}\n")
    print("Any SUSPICIOUS events in GT?", (gt["EVENT"] == "SUSPICIOUS").any())
    
    return gt


if __name__ == '__main__':
    
    print("\n" + "="*60)
    print("BUILDING GROUND TRUTH")
    print("="*60)
    gt = build_gt()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Unique patients in ground truth: {gt['EMPI'].nunique()}")