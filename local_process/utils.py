import pandas as pd
import json
import os
import regex as re
import glob
from typing import List, Dict, Optional
import pyarrow as pa
import pyarrow.parquet as pq

def extract_last_json_from_text(content: str):
    # Match balanced curly braces (naive but effective for model outputs)
    matches = list(re.finditer(r'\{(?:[^{}]|(?R))*\}', content, re.DOTALL))
    if not matches:
        raise ValueError("No JSON object found in LLM output.")
    last_json_str = matches[-1].group(0)
    try:
        # Optionally: clean up trailing commas or other formatting issues
        cleaned = re.sub(r',\s*}', '}', last_json_str)
        return json.loads(cleaned)
    except Exception as e:
        raise ValueError(f"Failed to parse last JSON: {e}")
    
def load_descriptions_from_txt(file_path: str) -> List[str]:
    with open(file_path, 'r') as file:
        descriptions = [line.strip() for line in file if line.strip()]
    return descriptions

def find_patient_bucket(dir_path: str, patient_id: int) -> Optional[int]:
    print("dir_path:", dir_path)
    bucket_dirs = [d for d in os.listdir(dir_path) if d.startswith('patient_bucket=')]
    print(f"Found {len(bucket_dirs)} bucket directories")
    
    for bucket_dir in bucket_dirs:
        bucket_path = os.path.join(dir_path, bucket_dir)
        parquet_files = glob.glob(os.path.join(bucket_path, '*.parquet'))
        
        for file in parquet_files:
            try:
                # Read just a sample to check for the patient
                df_sample = pd.read_parquet(file)
                if 'patient_id' in df_sample.columns:
                    # Handle object type conversion safely
                    try:
                        patient_ids_numeric = pd.to_numeric(df_sample['patient_id'], errors='coerce')
                        if int(patient_id) in patient_ids_numeric.values:
                            bucket_num = bucket_dir.replace('patient_bucket=', '')
                            return int(bucket_num)
                    except:
                        # Fallback: compare as strings
                        if str(patient_id) in df_sample['patient_id'].astype(str).values:
                            bucket_num = bucket_dir.replace('patient_bucket=', '')
                            return int(bucket_num)
            except Exception as e:
                print(f"Error reading {file}: {e}")

    return None

def load_from_parquet(input_dir: str, patient_id: int) -> List[Dict[str, str]]:
    """Read all entries for a patient from the parquet files in the input directory"""
    print(f"input_dir: {input_dir}, patient_id: {patient_id}")
    bucket = find_patient_bucket(input_dir, patient_id)
    # bucket = 116  # for testing
    if bucket is None:
        print(f"Patient ID {patient_id} not found in any bucket")
        return []
    print(f"Patient ID {patient_id} found in bucket {bucket}")
    bucket_path = f'/Users/carlotta/Desktop/Code_MT/data/timeline_ds_STS/patient_bucket={bucket}'

    parquet_files = glob.glob(os.path.join(bucket_path, "*.parquet"))
    if not parquet_files:
        print(f"No parquet files found in bucket {bucket}")
        return pd.DataFrame()
    
    try:
        dfs = []
        for file in parquet_files:
            df = pd.read_parquet(file)
            print(f"Loaded {len(df)} records from {file}")
            # Filter for the specific patient
            if 'patient_id' in df.columns:
                empi_col = 'patient_id'
                try:
                    df[empi_col] = pd.to_numeric(df[empi_col], errors='coerce')
                    patient_data = df[df[empi_col] == int(patient_id)]
                except:
                    # Fallback: compare as strings
                    patient_data = df[df[empi_col].astype(str) == str(patient_id)]
                print(f"Found {len(patient_data)} records for patient {patient_id} in {file}")
                if not patient_data.empty:
                    print("Appending patient data")
                    dfs.append(patient_data)
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            print(f"Total records found for patient {patient_id}: {len(combined_df)}")
            # Don't filter by note types here - return all data for the patient
            return combined_df
        
    except Exception as e:
        print(f"Error processing files for patient {patient_id}: {e}")
        return pd.DataFrame()
    
def extract_report_description(meta_json):
    try:
        meta = json.loads(meta_json) if isinstance(meta_json, str) else meta_json
        return meta.get("Report_Description", None)
    except Exception:
        return None
    
def extract_single_reports(data_dir, patient_id, modality):
    if modality == 'RAD': 
        chest_related_descriptions = load_descriptions_from_txt('chest_related_RAD_Report_Descriptions.txt')

    patient_data = load_from_parquet(data_dir, patient_id)

    # return list of single report entities 
    if 'note_date' in patient_data.columns:
        patient_data = patient_data.sort_values(by='note_date')
    else:
        print("note_date column not found in patient data")
    if patient_data.empty:
        print(f"No data found for patient {patient_id}")
        return []
    else:
        # return list of single report entities
        mod_data = patient_data[patient_data['modality'].isin([modality])]
        mod_data["Report_Description"] = mod_data["meta_json"].apply(extract_report_description)
        if modality == 'RAD':
            # Filter to only keep rows where Report_Description is in chest_related_descriptions
            mod_data = mod_data[mod_data["Report_Description"].isin(chest_related_descriptions)]
        # return list of dicts with note_date and text

        report_list = [
            {
                "id": f"{patient_id}_{modality}_{i+1}",
                "report_description": mod_data["Report_Description"].iloc[i],
                "note_date": row["note_date"],
                "text": row["text"],
            }
            for i, (_, row) in enumerate(mod_data.iterrows())
        ]
        print(f"Total {modality} reports for patient {patient_id}: {len(report_list)}")
    return report_list

def extract_mod_reports(data_dir, patient_id, modality):
    if modality == 'RAD': 
        chest_related_descriptions = load_descriptions_from_txt('chest_related_RAD_Report_Descriptions.txt')

    patient_data = load_from_parquet(data_dir, patient_id)
    # sort chronologically by note_date
    if 'note_date' in patient_data.columns:
        patient_data = patient_data.sort_values(by='note_date')
    else:
        print("note_date column not found in patient data")
    if patient_data.empty:
        print(f"No data found for patient {patient_id}")
        return ""
    else:
        mod_data = patient_data[patient_data['modality'].isin([modality])]
        print(f"Total {modality} reports for patient {patient_id}: {len(mod_data)}")
        if modality == 'RAD':
            mod_data["Report_Description"] = mod_data["meta_json"].apply(extract_report_description)

            # Filter to only keep rows where Report_Description is in chest_related_descriptions
            mod_data = mod_data[mod_data["Report_Description"].isin(chest_related_descriptions)]
        # convert to a single string add ---Report: i & Date:  --- between reports
        full_history = "\n\n".join(
            f"---Report: {i+1} - Date: {row['note_date']} ---\n{row['text']}"
            for i, (_, row) in enumerate(mod_data.iterrows())
        )
        # save full history to a text file
        #with open(f"patient_{patient_id}_{modality}_reports.txt", "w") as f:
            #f.write(full_history)
    return full_history
# only load PAT reports which are Chest related

# only load OPN reports which are Chest related

"""if __name__ == "__main__":
    data_dir = '/Users/carlotta/Desktop/Code_MT/data/timeline_ds_STS'
    patient_id = 104352354 # bucket: 116
    extract_chest_radreports(patient_id, data_dir)"""