import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

# 1. Load
pred = pd.read_csv("predicted_events.csv")
gt = pd.read_csv("ground_truth_events.csv")

print("PRED columns:", pred.columns.tolist())
print("GT columns:", gt.columns.tolist())
print("PRED sample rows:\n", pred.head())
print("GT sample rows:\n", gt.head())

# normalize same as script
pred['Event_DATE'] = pd.to_datetime(pred.get('Event_DATE'), errors='coerce')
gt['Event_DATE'] = pd.to_datetime(gt['EVENT_DATE'], errors='coerce')
gt['INDEX_SURGERY'] = pd.to_datetime(gt['INDEX_SURGERY'], errors='coerce')

gt['INDEXSURGERY_MONTH'] = gt['INDEX_SURGERY'].dt.to_period('M')

print(f"Total predictions before filtering: {len(pred)}")

# 3. Filter out predictions that match index surgery month
def filter_index_surgery_events(pred_df, gt_df):
    """
    Remove predicted events that occur in the same month as the patient's index surgery.
    """
    # Create a mapping of EMPI -> index surgery month
    surgery_months = gt_df[['EMPI', 'INDEXSURGERY_MONTH']].drop_duplicates().dropna()
    surgery_dict = dict(zip(surgery_months['EMPI'], surgery_months['INDEXSURGERY_MONTH']))
    
    # Convert pred Event_DATE to period for comparison
    pred_df['Event_MONTH'] = pred_df['Event_DATE'].dt.to_period('M')
    
    # Filter out events matching index surgery month
    filtered_pred = []
    removed_count = 0
    
    for _, row in pred_df.iterrows():
        empi = row['EMPI']
        event_month = row['Event_MONTH']
        
        # Check if this patient has an index surgery date
        if empi in surgery_dict:
            surgery_month = surgery_dict[empi]
            
            # Skip if event is in the same month as surgery
            if event_month <= (surgery_month + 1):
                removed_count += 1
                # print(f"Filtering out: EMPI={empi}, Event={row['EventType']}, Date={row['Event_DATE'].strftime('%Y-%m')} (matches surgery month)")
                continue
        
        filtered_pred.append(row)
    
    result = pd.DataFrame(filtered_pred)
    
    # Remove the temporary Event_MONTH column
    if 'Event_MONTH' in result.columns:
        result = result.drop(columns=['Event_MONTH'])
    
    print(f"Removed {removed_count} events matching index surgery month")
    print(f"Remaining predictions: {len(result)}")
    
    return result
pred = filter_index_surgery_events(pred, gt)

# 3. Apply priority logic to predictions: REC > SUSP > NOREC
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

# Apply priority deduplication to predictions
pred = apply_event_priority(pred)

# 4. Group by patient and match events
def match_events(gt_df, pred_df, tolerance_months=2):
    """
    Match ground truth events with predicted events within a tolerance window.
    Predictions have already been deduplicated by priority (REC > SUSP > NOREC).
    Returns matches DataFrame and detailed false positives.
    """
    results = []
    false_positives_detail = []
    skipped_no_pred = []
    
    for patient in gt_df['EMPI'].unique():
        g = gt_df[gt_df['EMPI'] == patient]
        p = pred_df[pred_df['EMPI'] == patient]
        
        if p.empty:
            skipped_no_pred.append(patient)
            continue
        
        matched_preds = set()
        for _, g_row in g.iterrows():
            g_date, g_type = g_row['Event_DATE'], g_row['EVENT']
            # collect candidate predictions using month periods (robust month +/- tolerance)
            candidates = []
            if not (pd.isna(g_date)):
                g_period = g_date.to_period('M')
                for i, p_row in p.iterrows():
                    p_date = p_row['Event_DATE']
                    if pd.isna(p_date):
                        continue
                    p_period = p_date.to_period('M')
                    months_diff = abs((p_period.year - g_period.year) * 12 + (p_period.month - g_period.month))
                    if months_diff <= tolerance_months:
                        candidates.append((i, p_row, months_diff))

            if candidates:
                # If GT is REC, prefer REC predictions first; if any REC preds exist within tolerance,
                # match all REC preds and do NOT match SUSP preds for this GT.
                if g_type == 'REC':
                    rec_cands = [c for c in candidates if c[1]['EventType'] == 'REC']
                    use_cands = rec_cands if rec_cands else candidates  # fallback to any if no REC preds
                else:
                    # for non-REC GT, accept any candidate predictions
                    use_cands = candidates

                for i, p_row, months_diff in use_cands:
                    if i in matched_preds:
                        continue
                    matched_preds.add(i)
                    true_date_str = g_date.strftime('%Y-%m') if not pd.isna(g_date) else None
                    pred_date_str = p_row['Event_DATE'].strftime('%Y-%m') if not pd.isna(p_row['Event_DATE']) else None
                    results.append({
                        "patient": patient,
                        "true_type": g_type,
                        "pred_type": p_row['EventType'],
                        "delta_months": months_diff,
                        "true_date": true_date_str,
                        "pred_date": pred_date_str
                    })
            else:
                results.append({"patient": patient, "true_type": g_type, "pred_type": "NONE", "true_date": (g_date.strftime('%Y-%m') if not pd.isna(g_date) else None), "pred_date": None})
        # false positives - store detailed information
        for i, p_row in p.iterrows():
            if i not in matched_preds:
                results.append({"patient": patient, "true_type": "NONE", "pred_type": p_row['EventType']})
                
                # Store detailed false positive information
                false_positives_detail.append({
                    "patient": patient,
                    "pred_type": p_row['EventType'],
                    "pred_date": p_row['Event_DATE'].strftime('%Y-%m'),
                    "evidence": p_row.get('Evidence Quote', 'N/A'),
                    "vis_file": p_row.get('VIS file', 'N/A'),
                    "certainty": p_row.get('Certainty', 'N/A')
                })
    
    return pd.DataFrame(results), pd.DataFrame(false_positives_detail)

matches, false_positives = match_events(gt, pred)
confusion = matches.groupby(['true_type', 'pred_type']).size().reset_index(name='count')
print(confusion)

fn = matches[(matches['true_type'] != 'NONE') & (matches['pred_type'] == 'NONE')].copy()
print("\n" + "="*80)
print(f"FALSE NEGATIVES (true event with no matching prediction): {len(fn)}")
print("="*80)
if len(fn) > 0:
    fn_display = fn[['patient', 'true_type', 'true_date']].copy()
    print(fn_display.to_string(index=False))
    fn.to_csv("false_negatives.csv", index=False)
    print(f"Full false negatives saved to: false_negatives.csv")

# Filter and display false positives
fp_rec = false_positives[false_positives['pred_type'] == 'REC']
fp_susp = false_positives[false_positives['pred_type'] == 'SUSP']

print("\n" + "="*80)
print(f"FALSE POSITIVES - RECURRENCE (true_type=NONE, pred_type=REC): {len(fp_rec)} entries")
print("="*80)
if len(fp_rec) > 0:
    # Display only patient, pred_type, and pred_date
    fp_rec_display = fp_rec[['patient', 'pred_type', 'pred_date']].copy()
    print(fp_rec_display.to_string(index=False))
    
    # Save full details to CSV for detailed review
    fp_rec.to_csv("false_positives_RECURRENCE.csv", index=False)
    print(f"\nFull details saved to: false_positives_RECURRENCE.csv")

print("\n" + "="*80)
print(f"FALSE POSITIVES - SUSPICIOUS (true_type=NONE, pred_type=SUSP): {len(fp_susp)} entries")
print("="*80)
if len(fp_susp) > 0:
    # Display only patient, pred_type, and pred_date
    fp_susp_display = fp_susp[['patient', 'pred_type', 'pred_date']].copy()
    print(fp_susp_display.to_string(index=False))
    
    # Save full details to CSV for detailed review
    fp_susp.to_csv("false_positives_SUSPICIOUS.csv", index=False)
    print(f"\nFull details saved to: false_positives_SUSPICIOUS.csv")

rec_vs_susp = matches[(matches['true_type'] == 'REC') & (matches['pred_type'] == 'SUSP')].copy()
print("\n" + "="*80)
print(f"GT=REC but PRED=SUSP: {len(rec_vs_susp)} entries")
print("="*80)
if len(rec_vs_susp) > 0:
    display_cols = ['patient', 'true_type', 'true_date', 'pred_type', 'pred_date']
    # Some columns may be missing in matches; select intersection for safe printing
    display_cols = [c for c in display_cols if c in rec_vs_susp.columns]
    print(rec_vs_susp[display_cols].to_string(index=False))
    #rec_vs_susp.to_csv("rec_true_REC_pred_SUSP.csv", index=False)
    #print(f"Full details saved to: rec_true_REC_pred_SUSP.csv")

"""# Optional: Show sample evidence for inspection
print("\n" + "="*80)
print("SAMPLE FALSE POSITIVE EVIDENCE (first 5 of each type)")
print("="*80)
if len(fp_rec) > 0:
    print("\n--- RECURRENCE False Positives ---")
    for idx, row in fp_rec.head(5).iterrows():
        print(f"\nPatient: {row['patient']} | Date: {row['pred_date']} | Certainty: {row['certainty']}")
        print(f"Evidence: {row['evidence'][:200]}...")  # First 200 chars

if len(fp_susp) > 0:
    print("\n--- SUSPICIOUS False Positives ---")
    for idx, row in fp_susp.head(5).iterrows():
        print(f"\nPatient: {row['patient']} | Date: {row['pred_date']} | Certainty: {row['certainty']}")
        print(f"Evidence: {row['evidence'][:200]}...")  # First 200 chars"""

# check dtypes and uniqueness
"""print("pred EMPI present?", 'EMPI' in pred.columns)
print("unique EMPI types:", pred['EMPI'].dtype if 'EMPI' in pred.columns else None, gt['EMPI'].dtype)

# inspect problematic patients
for empi in [patientA, patientB, patientC]:
    print("\n--- DEBUG EMPI:", empi, '---')
    print("GT rows:")
    print(gt[gt['EMPI'] == empi][['EMPI','INDEX_SURGERY','EVENT','EVENT_DATE']].to_string(index=False))
    print("PRED rows:")
    if 'EMPI' in pred.columns:
        print("\nPRED there")
        # print(pred[pred['EMPI'] == empi][['EMPI','EventType','Event_DATE','Evidence Quote']].to_string(index=False))
    else:
        print("No EMPI column in predictions - cannot match by patient")

    # show exact date differences between any pred and gt rows for this EMPI
    g_rows = gt[gt['EMPI'] == empi]
    p_rows = pred[pred['EMPI'] == empi] if 'EMPI' in pred.columns else pd.DataFrame()
    for _, gr in g_rows.iterrows():
        for _, pr in p_rows.iterrows():
            d_g = pd.to_datetime(gr['EVENT_DATE'], errors='coerce')
            d_p = pd.to_datetime(pr['Event_DATE'], errors='coerce')
            print("GT date ->", d_g, "PRED date ->", d_p, "delta days:",
                  None if pd.isna(d_g) or pd.isna(d_p) else abs((d_p - d_g).days))"""