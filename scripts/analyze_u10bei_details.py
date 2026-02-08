
import glob
import json
import os
import pandas as pd
import hashlib
from tqdm import tqdm
import warnings

# Suppress pandas performance warnings that can be noisy
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def calculate_hash(text):
    """Calculates SHA256 hash for a given text."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def process_u10bei_record(record, dataset_name):
    """Extracts information from a single u-10bei JSON record."""
    user_content = ""
    assistant_content = ""
    system_content = ""

    # Extract content from messages
    for msg in record.get('messages', []):
        role = msg.get('role')
        content = msg.get('content', '')
        if role == 'user':
            user_content = content
        elif role == 'assistant':
            assistant_content = content
        elif role == 'system':
            system_content = content
            
    if not user_content or not assistant_content:
        return None

    # Use system prompt + user prompt for a more unique user-side hash
    full_user_prompt = system_content + user_content
    full_content = full_user_prompt + assistant_content
    
    has_cot = assistant_content.strip().startswith('Approach:')
    
    metadata = record.get('metadata', {})
    
    return {
        'dataset': dataset_name,
        'full_hash': calculate_hash(full_content),
        'user_hash': calculate_hash(full_user_prompt),
        'has_cot': has_cot,
        'output_format': metadata.get('format', 'unknown').upper(),
        'task_type': metadata.get('type', 'unknown'),
        'complexity': metadata.get('complexity', 'unknown'),
    }

def analyze_u10bei_details(sft_path, output_csv_path):
    """
    Performs a detailed analysis of u-10bei datasets.
    """
    search_pattern = os.path.join(sft_path, "u-10bei*", "train.jsonl")
    jsonl_files = glob.glob(search_pattern)

    all_records_data = []
    print(f"Found {len(jsonl_files)} u-10bei train.jsonl files to analyze.")
    print("Loading and parsing records...")

    for file_path in jsonl_files:
        dataset_name = os.path.basename(os.path.dirname(file_path))
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"  - Reading {dataset_name}"):
                try:
                    record = json.loads(line)
                    processed = process_u10bei_record(record, dataset_name)
                    if processed:
                        all_records_data.append(processed)
                except json.JSONDecodeError:
                    continue
    
    if not all_records_data:
        print("No valid records found in u-10bei datasets.")
        return

    df = pd.DataFrame(all_records_data)
    
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"\nDetailed analysis data saved to {output_csv_path}")

    print("\n--- u-10bei Datasets: Deep Analysis Summary ---")

    # 1. Record Counts
    print("\n1. Record Count per Dataset:")
    print(df['dataset'].value_counts().to_string())

    # 2. Duplication Analysis
    print("\n2. Duplication Analysis (Super-set/Sub-set relations):")
    datasets = sorted(df['dataset'].unique(), reverse=True)
    hashes_by_dataset = {ds: set(df[df['dataset'] == ds]['full_hash']) for ds in datasets}

    for i in range(len(datasets)):
        for j in range(i + 1, len(datasets)):
            ds1 = datasets[i]
            ds2 = datasets[j] 
            
            overlap = len(hashes_by_dataset[ds1].intersection(hashes_by_dataset[ds2]))
            
            if overlap > 0:
                if hashes_by_dataset[ds2].issubset(hashes_by_dataset[ds1]):
                    print(f"  - '{ds2}' is a COMPLETE SUBSET of '{ds1}' ({overlap} records).")
                else:
                    print(f"  - '{ds1}' and '{ds2}' have a partial overlap of {overlap} records.")

    # 3. CoT Presence
    print("\n3. CoT (Chain of Thought) Presence by Dataset:")
    cot_dist = df.groupby('dataset')['has_cot'].value_counts(normalize=True).unstack(fill_value=0).applymap('{:.2%}'.format)
    print(cot_dist.to_string())

    # 4. Complexity Distribution
    print("\n4. Complexity Distribution by Dataset:")
    complexity_dist = pd.crosstab(df['dataset'], df['complexity'])
    print(complexity_dist.to_string())

    # 5. Output Format Distribution
    print("\n5. Output Format Distribution by Dataset:")
    format_dist = pd.crosstab(df['dataset'], df['output_format'])
    print(format_dist.to_string())


if __name__ == '__main__':
    SFT_DATA_PATH = '/home/junta_takahashi/matsuo_llm2025_mainCompe/data/raw/SFT'
    OUTPUT_CSV = '/home/junta_takahashi/matsuo_llm2025_mainCompe/outputs/u10bei_detailed_analysis.csv'
    analyze_u10bei_details(SFT_DATA_PATH, OUTPUT_CSV)
