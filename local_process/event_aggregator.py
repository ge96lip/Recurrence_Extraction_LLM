import os
import json
import csv
from datetime import datetime
from collections import defaultdict
from datetime import date

# --- Configuration ---
# The base path where all patient folders are located
BASE_DIR = f'/vast/florian/carlotta/LLMAIx/rec_task/outputs/recurrence_task/Qwen3-Instruct/1031/'   #{date.today().strftime("%m%d")}'
INPUT_FILENAME = 'output.json'
OUTPUT_FILENAME = 'patient_level_events.csv' # Changed filename to reflect aggregation

def clean_date_to_month(date_str):
    """
    Converts various date formats (including M/D/YYYY, M/YYYY, etc.) to a YYYY-MM string for grouping.
    Returns None if the date is invalid or missing.
    """
    if not date_str:
        return None

    date_str = str(date_str).strip()
    if not date_str or date_str.lower() == 'null':
        return None
    
    date_part = date_str.split(' ')[0] # Isolate the date part

    # List of expected formats to try parsing, ordered from most specific to least specific
    formats_to_try = [
        # Full Dates (Day/Month/Year required)
        '%Y-%m-%d',     # Standard (2022-05-06)
        '%m-%d-%Y',     # Dash-separated US (05-06-2022)
        '%m/%d/%Y',     # Slash-separated US (05/06/2022)
        '%m/%d/%y',     # Slash-separated US short year (05/06/22)
    ]
    
    # Partial Dates (Month/Year only) - Try these last, as they are less specific
    formats_to_try_partial = [
        '%Y-%m',        # Standard (2022-05)
        '%m/%Y',        # Slash-separated Month/Year (07/2023)
    ]

    dt = None
    for fmt in formats_to_try:
        try:
            dt = datetime.strptime(date_part, fmt)
            break
        except ValueError:
            continue
    
    # If full date parsing failed, try partial date parsing
    if dt is None:
        for fmt in formats_to_try_partial:
            try:
                dt = datetime.strptime(date_part, fmt)
                break
            except ValueError:
                continue

    if dt is None:
        # If all formats fail, print warning and return None
        print(f"Warning: Could not parse date '{date_part}'. Skipping date.")
        return None
    
    # Return the date in the required YYYY-MM format
    return dt.strftime('%Y-%m')

def process_patient_data(patient_dir):
    """
    Reads the JSON, extracts, formats, and aggregates events by YYYY-MM and type.
    Handles new structure with recurrence, suspicious_event, and no_recurrence_event.
    """
    json_path = os.path.join(patient_dir, INPUT_FILENAME)
    patient_id = os.path.basename(patient_dir)
    
    # Aggregated events are stored in a dict where the key is (Type, YYYY-MM)
    aggregated_events = defaultdict(lambda: {
        'Event Type': None,
        'Event Date': None,         # YYYY-MM format
        'Date extracted': set(),    # Set of VIS dates (YYYY-MM-DD or YYYY-MM)
        'Evidence Quote': set(),    # Set of evidence quotes
        'VIS file': set(),          # Set of vis_id's
        'Reasoning': set(),        # Set of reasoning texts
        'Certainty': set(),         # Set of certainty levels
    })

    if not os.path.exists(json_path):
        print(f"Skipping {patient_id}: {INPUT_FILENAME} not found.")
        return []

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Skipping {patient_id}: JSON decoding failed.")
        return []

    for entry in data:
        output = entry.get('output', {})
        vis_date_raw = entry.get('vis_date')
        vis_id = entry.get('vis_id')
        # strip the id from vis_id e.g. 101295110_VIS_25 becomes VIS_25
        if vis_id and '_' in vis_id:
            vis_id = vis_id.split('_', 1)[1]

        if output.get('error'):
            print(f"Skipping {vis_id}: LLM error reported.")
            continue

        vis_date_formatted = clean_date_to_month(vis_date_raw) # YYYY-MM format
        certainty = output.get('certainty', 'unknown')

        # Priority: recurrence > suspicious_event > no_recurrence_event > no_event
        # (based on classification logic: recurrence takes precedence)

        # --- Extract RECURRENCE events ---
        if output.get('recurrence') is True:
            event_type = 'REC'
            event_date_coarse = clean_date_to_month(output.get('recurrence_date'))
            evidence = output.get('recurrence_evidence')

            if event_date_coarse:
                key = (event_type, event_date_coarse)
                event_entry = aggregated_events[key]

                event_entry['Event Type'] = event_type
                event_entry['Event Date'] = event_date_coarse

                if vis_date_formatted:
                    event_entry['Date extracted'].add(vis_date_formatted)
                event_entry['VIS file'].add(vis_id)
                if evidence:
                    event_entry['Evidence Quote'].add(evidence.replace('\n', ' ').strip())
                event_entry['Certainty'].add(certainty)

        # --- Extract SUSPICIOUS events (only if recurrence is not true) ---
        elif output.get('suspicious_event') is True:
            event_type = 'SUSP'
            event_date_coarse = clean_date_to_month(output.get('suspicious_event_date'))
            evidence = output.get('suspicious_event_evidence')

            if event_date_coarse:
                key = (event_type, event_date_coarse)
                event_entry = aggregated_events[key]

                event_entry['Event Type'] = event_type
                event_entry['Event Date'] = event_date_coarse

                if vis_date_formatted:
                    event_entry['Date extracted'].add(vis_date_formatted)
                event_entry['VIS file'].add(vis_id)
                if evidence:
                    event_entry['Evidence Quote'].add(evidence.replace('\n', ' ').strip())
                event_entry['Certainty'].add(certainty)

        # --- Extract NO_RECURRENCE events ---
        elif output.get('no_recurrence_event') is True:
            event_type = 'NOREC'
            # For NO_RECURRENCE, use the VIS date as the event date
            event_date_coarse = vis_date_formatted
            evidence = output.get('no_recurrence_evidence')

            if event_date_coarse:
                key = (event_type, event_date_coarse)
                event_entry = aggregated_events[key]

                event_entry['Event Type'] = event_type
                event_entry['Event Date'] = event_date_coarse

                if vis_date_formatted:
                    event_entry['Date extracted'].add(vis_date_formatted)
                event_entry['VIS file'].add(vis_id)
                if evidence:
                    event_entry['Evidence Quote'].add(evidence.replace('\n', ' ').strip())
                event_entry['Certainty'].add(certainty)
        

        # --- Handle NO_EVENT (no extraction needed, just log) ---
        elif output.get('no_event') is True:
            # Optionally log or skip - no event to aggregate
            continue            

        event_entry['Reasoning'].add(output.get('reasoning', '').replace('\n', ' ').strip())

    # Convert the dictionary values into a list for sorting and output
    final_events = list(aggregated_events.values())

    # Sort events chronologically by 'Event Date' (YYYY-MM)
    final_events.sort(key=lambda x: x['Event Date'] or '0000-00')

    # Reformat sets into joined strings for CSV output
    for event in final_events:
        # Join sets and sort content for deterministic output
        event['Date extracted'] = ' | '.join(sorted(event['Date extracted']))
        event['Evidence Quote'] = ' | '.join(sorted(event['Evidence Quote']))
        event['VIS file'] = ' | '.join(sorted(event['VIS file']))
        event['Certainty'] = ' | '.join(sorted(event['Certainty']))

    return final_events

def write_events_to_csv(patient_dir, events):
    """Writes the sorted list of aggregated events to a CSV file."""
    output_path = os.path.join(patient_dir, OUTPUT_FILENAME)
    
    if not events:
        print(f"No events to write for {os.path.basename(patient_dir)}. Skipping CSV creation.")
        return

    fieldnames = ['Event Type', 'Event Date', 'Date extracted', 'Evidence Quote', 'VIS file', 'Certainty'] 

    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(events)
        print(f"Successfully wrote {len(events)} aggregated events to {output_path}")
    except Exception as e:
        print(f"Error writing CSV for {os.path.basename(patient_dir)}: {e}")

def main():
    """Main function to loop through all patients and process data."""
    print(f"Starting event aggregation from base directory: {BASE_DIR}")

    # Iterate through subdirectories (patient IDs)
    for item in os.listdir(BASE_DIR):
        patient_dir = os.path.join(BASE_DIR, item)

        if os.path.isdir(patient_dir):
            events = process_patient_data(patient_dir)

            if events:
                write_events_to_csv(patient_dir, events)
            else:
                print(f"No valid events found for patient {item}.")

    print("Aggregation complete.")

if __name__ == '__main__':
    # NOTE: This script assumes the patient folders and JSON files exist in the BASE_DIR.
    main()