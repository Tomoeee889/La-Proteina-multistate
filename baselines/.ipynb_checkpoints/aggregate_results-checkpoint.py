"""
Aggregate analysis results from all 6 experiments into a single comparison table.

Usage:
    python baselines/aggregate_results.py
"""

import pandas as pd
from pathlib import Path

BASELINES_DIR = Path("baselines")

# Список всех экспериментов: (имя папки, задача, режим)
EXPERIMENTS = [
    ("task1_unconditional_default",       "Task1_unconditional", "default"),
    ("task1_unconditional_sim_eps",       "Task1_unconditional", "sim_eps"),
    ("task2_conditional_default",         "Task2_conditional_small", "default"),
    ("task2_conditional_sim_eps",         "Task2_conditional_small", "sim_eps"),
    ("task3_conditional_large_default",   "Task3_conditional_large", "default"),
    ("task3_conditional_large_sim_eps",   "Task3_conditional_large", "sim_eps"),
]


def load_summary(exp_dir_name):
    """Load analysis_summary.csv from experiment directory."""
    summary_path = BASELINES_DIR / exp_dir_name / "analysis_summary.csv"
    if not summary_path.exists():
        print(f"WARNING: {summary_path} not found, skipping")
        return None
    df = pd.read_csv(summary_path, header=[0, 1], index_col=0)
    return df


def main():
    all_rows = []
    
    for exp_dir, task, mode in EXPERIMENTS:
        summary = load_summary(exp_dir)
        if summary is None:
            continue
        
        # Flatten multi-level columns
        for noise_scale, row in summary.iterrows():
            row_dict = {
                'task': task,
                'mode': mode,
                'noise_scale': noise_scale,
            }
            for col in summary.columns:
                metric, stat = col
                row_dict[f"{metric}_{stat}"] = row[col]
            all_rows.append(row_dict)
    
    if not all_rows:
        print("No data found!")
        return
    
    combined = pd.DataFrame(all_rows)
    
    # Save full table
    output_csv = BASELINES_DIR / "all_experiments_comparison.csv"
    combined.to_csv(output_csv, index=False)
    print(f"\nFull comparison table saved: {output_csv}")
    
    # Print compact comparison: task × mode × noise_scale → key metrics
    print("\n" + "=" * 100)
    print("COMPARISON TABLE")
    print("=" * 100)
    
    for task in ["Task1_unconditional", "Task2_conditional_small", "Task3_conditional_large"]:
        print(f"\n>>> {task}")
        print("-" * 100)
        
        task_df = combined[combined['task'] == task]
        
        # Select key columns
        key_cols = ['noise_scale', 'mode']
        available_cols = [c for c in [
            'full_rmsd_mean', 'full_rmsd_std', 'full_rmsd_median',
            'seq_identity_mean', 'seq_identity_std',
            'scaffold_rmsd_mean', 'scaffold_rmsd_std',
            'motif_rmsd_mean', 'motif_rmsd_std',
            'scaffold_seq_identity_mean',
            'motif_seq_identity_mean',
        ] if c in task_df.columns]
        
        display_df = task_df[key_cols + available_cols].sort_values(['mode', 'noise_scale'])
        
        # Round floats
        for col in display_df.columns:
            if col not in ['task', 'mode', 'noise_scale']:
                display_df[col] = display_df[col].round(3)
        
        print(display_df.to_string(index=False))


if __name__ == "__main__":
    main()