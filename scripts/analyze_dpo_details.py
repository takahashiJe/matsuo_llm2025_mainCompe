
import json
import os
import pandas as pd
from tqdm import tqdm
import re

def analyze_dpo_details(dpo_file_path, output_dir):
    """
    Performs a detailed analysis of the DPO dataset.
    """
    if not os.path.exists(dpo_file_path):
        print(f"Error: DPO file not found at {dpo_file_path}")
        return

    all_records_data = []
    raw_examples = []
    max_examples_to_print = 3

    print("Loading and parsing DPO dataset...")
    with open(dpo_file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing DPO file"):
            try:
                record = json.loads(line)
                
                prompt = record.get('prompt', '')
                chosen = record.get('chosen', '')
                rejected = record.get('rejected', '')

                if not all([prompt, chosen, rejected]):
                    continue

                # --- Extract basic info ---
                has_cot_chosen = chosen.strip().startswith('Approach:')
                has_cot_rejected = rejected.strip().startswith('Approach:')
                
                # Infer output format from prompt
                output_format = 'unknown'
                format_match = re.search(r'expert in (\w+)', prompt, re.IGNORECASE)
                if not format_match:
                    format_match = re.search(r'output (\w+)', prompt, re.IGNORECASE)
                if format_match:
                    format_str = format_match.group(1).upper()
                    if format_str in ['JSON', 'XML', 'YAML', 'TOML', 'CSV']:
                        output_format = format_str

                all_records_data.append({
                    'has_cot_chosen': has_cot_chosen,
                    'has_cot_rejected': has_cot_rejected,
                    'output_format': output_format,
                })

                # --- Store raw examples for qualitative analysis ---
                if len(raw_examples) < max_examples_to_print:
                    raw_examples.append({
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected
                    })

            except json.JSONDecodeError:
                continue
    
    if not all_records_data:
        print("No valid records found to analyze.")
        return

    df = pd.DataFrame(all_records_data)
    
    # --- Save detailed data to CSV ---
    output_csv_path = os.path.join(output_dir, "dpo_analysis.csv")
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"\nDetailed analysis data saved to {output_csv_path}")

    # --- Print Summary Statistics ---
    print("\n--- DPO Dataset: Deep Analysis Summary ---")
    
    print(f"\n1. Total Records Analyzed: {len(df)}")

    print("\n2. CoT (Chain of Thought) Presence:")
    chosen_cot_pct = df['has_cot_chosen'].mean() * 100
    rejected_cot_pct = df['has_cot_rejected'].mean() * 100
    print(f"  - 'chosen' responses with CoT: {chosen_cot_pct:.2f}%")
    print(f"  - 'rejected' responses with CoT: {rejected_cot_pct:.2f}%")

    print("\n3. Output Format Distribution:")
    print(df['output_format'].value_counts().to_string())

    # --- Print Qualitative Examples ---
    print("\n--- Qualitative Examples (chosen vs. rejected) ---")
    for i, example in enumerate(raw_examples, 1):
        print(f"\n--- Example {i} ---")
        print(f"--- PROMPT ---\n{example['prompt']}\n")
        print(f"--- CHOSEN ---\n{example['chosen']}\n")
        print(f"--- REJECTED ---\n{example['rejected']}\n")
        
        # Add a brief automated comparison
        chosen_output = example['chosen'].split('Output:', 1)[-1]
        rejected_output = example['rejected'].split('Output:', 1)[-1]
        print("--- QUICK COMPARISON ---")
        if "```" in rejected_output and "```" not in chosen_output:
            print("- Observation: 'rejected' contains markdown code fences, 'chosen' does not.")
        if len(example['chosen']) < len(example['rejected']):
            print("- Observation: 'chosen' response is more concise than 'rejected'.")
        print("-" * 20)


if __name__ == '__main__':
    DPO_FILE = '/home/junta_takahashi/matsuo_llm2025_mainCompe/data/raw/DPO/u-10bei__dpo-dataset-qwen-cot/train.jsonl'
    OUTPUT_DIR = '/home/junta_takahashi/matsuo_llm2025_mainCompe/outputs/reports'
    analyze_dpo_details(DPO_FILE, OUTPUT_DIR)
