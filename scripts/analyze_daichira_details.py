import json
import os
import pandas as pd
import hashlib
import re
from tqdm import tqdm

def calculate_hash(text):
    """Calculates SHA256 hash for a given text."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def process_record(record, dataset_name, non_renderable_map):
    """Extracts information from a single JSON record."""
    user_content = ""
    assistant_content = ""
    for msg in record.get('messages', []):
        if msg.get('role') == 'user':
            user_content = msg.get('content', '')
        elif msg.get('role') == 'assistant':
            assistant_content = msg.get('content', '')
    
    if not user_content or not assistant_content:
        return None

    full_content = user_content + assistant_content
    
    return {
        'dataset': dataset_name,
        'id': record.get('id'),
        'full_hash': calculate_hash(full_content),
        'user_hash': calculate_hash(user_content),
        'task': record.get('task', 'unknown'),
        'output_format': non_renderable_map.get(record.get('category'), 'unknown'),
        'input_len': len(user_content),
        'output_len': len(assistant_content),
    }

def parse_concatenated_json(file_content):
    """Parses a file with multiple concatenated, pretty-printed JSON objects."""
    # This is a robust way to handle files with concatenated JSON objects.
    # It looks for '}' followed by whitespace and then '{' to split objects.
    json_objects_str = re.split(r'}\s*{', file_content)
    
    records = []
    # Re-add the braces that were removed by the split
    for i, s in enumerate(json_objects_str):
        if i == 0 and not s.startswith('{'):
            s = '{' + s
        if i == len(json_objects_str) - 1 and not s.endswith('}'):
            s = s + '}'
        if i > 0:
            s = '{' + s
        if i < len(json_objects_str) - 1:
            s = s + '}'
        
        try:
            records.append(json.loads(s))
        except json.JSONDecodeError:
            # This might happen for the last empty split, etc.
            continue
    return records


def analyze_daichira_details(sft_path, output_csv_path):
    """
    Performs a detailed analysis of daichira datasets, handling JSON, JSONL, and concatenated JSON formats.
    """
    dataset_paths = {
        "3k-mix": os.path.join(sft_path, "daichira__structured-3k-mix-sft", "train.jsonl"),
        "5k-mix": os.path.join(sft_path, "daichira__structured-5k-mix-sft", "train.jsonl"),
        "hard-4k": os.path.join(sft_path, "daichira__structured-hard-sft-4k", "train.jsonl"),
    }

    non_renderable_formats_map = {
        "C_JSON": "JSON", "C_XML": "XML", "C_YAML": "YAML", 
        "C_TOML": "TOML", "C_CSV": "CSV"
    }

    all_records_data = []
    print("Loading and parsing records from daichira datasets (Final Attempt)...")

    for name, file_path in dataset_paths.items():
        if not os.path.exists(file_path):
            print(f"Warning: File not found for {name} at {file_path}")
            continue
        
        print(f"Processing {name}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            # Special handling for the '3k-mix' file which is concatenated JSON
            if name == '3k-mix':
                try:
                    content = f.read()
                    json_records = parse_concatenated_json(content)
                    for record in tqdm(json_records, desc=f"  - Reading {name}"):
                        processed = process_record(record, name, non_renderable_formats_map)
                        if processed:
                            all_records_data.append(processed)
                except Exception as e:
                    print(f"Error processing the concatenated JSON file {file_path}: {e}")
                    continue
            # Standard JSONL processing for other files
            else:
                for line in tqdm(f, desc=f"  - Reading {name}"):
                    try:
                        record = json.loads(line)
                        processed = process_record(record, name, non_renderable_formats_map)
                        if processed:
                            all_records_data.append(processed)
                    except json.JSONDecodeError:
                        continue
    
    if not all_records_data:
        print("No valid records found to analyze.")
        return

    df = pd.DataFrame(all_records_data)
    
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"\nDetailed analysis data saved to {output_csv_path}")

    print("\n--- daichira Datasets: Final Analysis Summary ---")

    print("\n1. Duplication Analysis:")
    total_records = len(df)
    unique_records = df['full_hash'].nunique()
    print(f"  - Total records (sum of all files): {total_records}")
    print(f"  - Unique records (by content): {unique_records}")
    print(f"  - Overall duplication rate: {1 - unique_records / total_records:.2%}")

    df_3k = df[df['dataset'] == '3k-mix']
    df_5k = df[df['dataset'] == '5k-mix']
    
    if not df_3k.empty and not df_5k.empty:
        overlap_3k_in_5k = df_3k['full_hash'].isin(df_5k['full_hash']).sum()
        print(f"  - Records from '3k-mix' also found in '5k-mix': {overlap_3k_in_5k} / {len(df_3k)} ({overlap_3k_in_5k/len(df_3k):.2%})")
    else:
        print("  - '3k-mix' or '5k-mix' dataset was not processed correctly, cannot compute overlap.")

    print("\n2. Per-Dataset Characteristics:")
    summary = df.groupby('dataset').agg(
        record_count=('id', 'count'),
        avg_input_len=('input_len', 'mean'),
        avg_output_len=('output_len', 'mean')
    ).round(1)
    print(summary.to_string())

    print("\n3. Task Type Distribution by Dataset:")
    task_dist = pd.crosstab(df['dataset'], df['task'])
    print(task_dist.to_string())

    print("\n4. Output Format Distribution by Dataset:")
    format_dist = pd.crosstab(df['dataset'], df['output_format'])
    print(format_dist.to_string())
    
    print("\n5. Complexity Analysis (Average Lengths):")
    complexity_summary = df.groupby('dataset')[['input_len', 'output_len']].describe().round(1)
    print(complexity_summary[[('input_len', 'mean'), ('input_len', 'std'), ('output_len', 'mean'), ('output_len', 'std')]].to_string())

if __name__ == '__main__':
    SFT_DATA_PATH = '/home/junta_takahashi/matsuo_llm2025_mainCompe/data/raw/SFT'
    OUTPUT_CSV = '/home/junta_takahashi/matsuo_llm2025_mainCompe/outputs/daichira_detailed_analysis.csv'
    analyze_daichira_details(SFT_DATA_PATH, OUTPUT_CSV)