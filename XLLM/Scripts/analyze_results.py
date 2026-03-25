#!/usr/bin/env python3
"""
Script to compute the average score across multiple CSV files.
Each CSV may contain: index, prompt, response, parent, results, scores, step.
If a file is empty or has missing scores, fill them with 0.
"""

import pandas as pd
import numpy as np
import os
import glob

def calculate_average_score(csv_directory):
    """
    Calculate the overall average score from multiple CSV files.
    If a file is empty or has missing scores, treat them as 0.
    """
    csv_pattern = os.path.join(csv_directory, "*.csv")
    csv_files = glob.glob(csv_pattern)
    
    print(f"Found {len(csv_files)} CSV files in {csv_directory}")
    if not csv_files:
        print("⚠️ No CSV files found — default average score = 0.")
        return 0.0

    all_scores = []

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            # Case 1: Empty CSV file
            if df.empty:
                print(f"⚠️ {os.path.basename(csv_file)} is empty — using score=0.")
                all_scores.append(0.0)
                continue

            # Case 2: Missing 'scores' column
            if 'scores' not in df.columns:
                print(f"⚠️ {os.path.basename(csv_file)} missing 'scores' column — using score=0.")
                all_scores.append(0.0)
                continue

            # Case 3: Has scores but may contain NaN or invalid entries
            df['scores'] = pd.to_numeric(df['scores'], errors='coerce').fillna(0.0)
            print(df)
            file_mean = float(np.mean(df['scores']))
            all_scores.append(file_mean)
            print(f"✅ Processed {os.path.basename(csv_file)} | mean score={file_mean:.4f}")

        except Exception as e:
            print(f"❌ Error reading {csv_file}: {e}")
            all_scores.append(0.0)

    # Compute global average (including zero-filled cases)
    avg_score = float(np.mean(all_scores)) if all_scores else 0.0

    print(f"\n✅ Total files considered: {len(csv_files)}")
    print(f"✅ Global average score (fillna=0): {avg_score:.4f}")

    return avg_score

def main():
    csv_directory = "/projects/p32013/neurons/XLLM/Results/Qwen/Qwen3-4B-Instruct-2507/GPTFuzzer-2/"
    print("Calculating average score (treat empty files as 0)...")

    avg_score = calculate_average_score(csv_directory)

    # Save result to CSV
    output_file = "average_score_summary.csv"
    pd.DataFrame([{"average_score": avg_score}]).to_csv(output_file, index=False)

    print(f"\n📄 Results saved to: {output_file}")

if __name__ == "__main__":
    main()
