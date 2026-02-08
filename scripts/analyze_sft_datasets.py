
import glob
import json
import os
import pandas as pd
from tqdm import tqdm

def analyze_sft_datasets(sft_path, output_csv_path):
    """
    Analyzes SFT datasets focusing on non-renderable tasks.

    Args:
        sft_path (str): Path to the SFT data directory.
        output_csv_path (str): Path to save the analysis CSV.
    """
    jsonl_files = glob.glob(os.path.join(sft_path, "**", "*.jsonl"), recursive=True)
    
    non_renderable_formats = {"JSON", "XML", "YAML", "TOML", "CSV"}
    renderable_formats_map = {
        "C_HTML": "HTML", "C_SVG": "SVG", "C_MERMAID": "Mermaid", 
        "C_MATPLOTLIB": "Matplotlib", "C_REACT": "React", "C_ANGULAR": "Angular",
        "C_VUE": "Vue", "C_VEGA": "Vega", "C_TYPST": "Typst", 
        "C_CANVAS": "Canvas", "C_LATEX": "LaTeX"
    }
    non_renderable_formats_map = {
        "C_JSON": "JSON", "C_XML": "XML", "C_YAML": "YAML", 
        "C_TOML": "TOML", "C_CSV": "CSV"
    }

    analysis_data = []

    print(f"Found {len(jsonl_files)} jsonl files to analyze.")

    for file_path in tqdm(jsonl_files, desc="Analyzing files"):
        dataset_name = file_path.split(os.sep)[-2]
        
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    
                    output_format = None
                    has_cot = False
                    task_type = 'unknown'
                    complexity = 'unknown'

                    # --- Data Extraction Logic ---
                    if 'u-10bei' in dataset_name:
                        if 'metadata' in record and 'format' in record['metadata']:
                            output_format = record['metadata']['format'].upper()
                        
                        # Check for CoT in assistant message
                        if 'messages' in record:
                            for msg in record['messages']:
                                if msg.get('role') == 'assistant' and 'content' in msg.get('content', ''):
                                    if msg['content'].strip().startswith('Approach:'):
                                        has_cot = True
                                    break # Only check first assistant message
                        
                        if 'metadata' in record:
                            task_type = record['metadata'].get('type', 'unknown')
                            complexity = record['metadata'].get('complexity', 'unknown')

                    elif 'daichira' in dataset_name:
                        has_cot = False # daichira datasets are assumed to be clean
                        if 'category' in record:
                            category = record['category']
                            if category in non_renderable_formats_map:
                                output_format = non_renderable_formats_map[category]
                            elif category in renderable_formats_map:
                                output_format = renderable_formats_map[category]
                        
                        task_type = record.get('task', 'unknown')
                        # Complexity is not explicitly defined in daichira, use 'seed' as a proxy
                        if 'dummy_hard' in record.get('seed', ''):
                            complexity = 'hard'
                        elif 'mix' in dataset_name:
                            complexity = 'mix'

                    # --- Store Non-Renderable Data ---
                    if output_format in non_renderable_formats:
                        analysis_data.append({
                            'dataset_name': dataset_name,
                            'output_format': output_format,
                            'has_cot': has_cot,
                            'task_type': task_type,
                            'complexity': complexity,
                        })

                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from a line in {file_path}")
                    continue

    if not analysis_data:
        print("No non-renderable data found to analyze.")
        return

    df = pd.DataFrame(analysis_data)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"Analysis complete. Results saved to {output_csv_path}")

    # --- Print Summary Statistics ---
    print("\n--- Analysis Summary ---")
    
    # Overall distribution by dataset
    print("\n1. Record Count per Dataset (Non-renderable only):")
    print(df['dataset_name'].value_counts().to_string())
    
    # Distribution of formats
    print("\n2. Output Format Distribution (All Datasets):")
    print(df['output_format'].value_counts().to_string())
    
    # CoT presence
    print("\n3. CoT Presence (u-10bei datasets only):")
    if not df[df['dataset_name'].str.contains('u-10bei')].empty:
        print(df[df['dataset_name'].str.contains('u-10bei')]['has_cot'].value_counts().to_string())
    else:
        print("No u-10bei data found.")

    # Task Type distribution
    print("\n4. Task Type Distribution (All Datasets):")
    print(df['task_type'].value_counts().to_string())

    # Complexity distribution
    print("\n5. Complexity Distribution (All Datasets):")
    print(df['complexity'].value_counts().to_string())

    # Crosstab: Dataset vs. Format
    print("\n6. Crosstab: Dataset vs. Output Format:")
    crosstab_df = pd.crosstab(df['dataset_name'], df['output_format'])
    print(crosstab_df.to_string())


if __name__ == '__main__':
    SFT_DATA_PATH = '/home/junta_takahashi/matsuo_llm2025_mainCompe/data/raw/SFT'
    OUTPUT_CSV = '/home/junta_takahashi/matsuo_llm2025_mainCompe/outputs/sft_non_renderable_analysis.csv'
    analyze_sft_datasets(SFT_DATA_PATH, OUTPUT_CSV)
