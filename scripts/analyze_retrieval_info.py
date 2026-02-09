#!/usr/bin/env python3
"""
Analyze retrieval_info.csv to compute perturbation selection percentages.

For each unique (cell_type, pert_name) pair, this script:
1. Extracts all selected perturbations across all cells
2. Counts how many times each perturbation was selected
3. Calculates the percentage of appearance
4. Saves results to a CSV file

Usage:
    python analyze_retrieval_info.py <input_csv> [--output-dir <dir>]
"""

import argparse
import csv
import os
from collections import defaultdict, Counter
from pathlib import Path


def parse_selected_perturbations(selected_perts_str):
    """
    Parse the pipe-separated selected perturbations string.
    
    Args:
        selected_perts_str: String like "GENE1|GENE2|GENE3" or "none"
    
    Returns:
        List of perturbation names
    """
    if selected_perts_str == "none" or not selected_perts_str:
        return []
    return [p.strip() for p in selected_perts_str.split('|') if p.strip()]


def analyze_retrieval_info(input_csv_path, output_dir=None):
    """
    Analyze retrieval information and create per-condition summaries.
    
    Args:
        input_csv_path: Path to retrieval_info.csv
        output_dir: Directory to save output CSV files (defaults to same dir as input)
    """
    input_path = Path(input_csv_path)
    
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read the CSV and group by (cell_type, pert_name)
    condition_data = defaultdict(list)
    
    print(f"Reading {input_path}...")
    with open(input_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cell_type = row['cell_type']
            pert_name = row['pert_name']
            selected_perts = parse_selected_perturbations(row['selected_perturbations'])
            
            condition_key = (cell_type, pert_name)
            condition_data[condition_key].extend(selected_perts)
    
    print(f"Found {len(condition_data)} unique (cell_type, pert_name) pairs")
    
    # Process each condition
    for (cell_type, pert_name), all_selected_perts in condition_data.items():
        # Count occurrences of each selected perturbation
        pert_counts = Counter(all_selected_perts)
        
        # Calculate total selections
        total_selections = len(all_selected_perts)
        
        if total_selections == 0:
            print(f"  Skipping {cell_type}/{pert_name}: no perturbations selected")
            continue
        
        # Prepare output data
        output_rows = []
        for pert, count in pert_counts.most_common():
            percentage = (count / total_selections) * 100
            output_rows.append({
                'perturbation': pert,
                'count': count,
                'percentage': f"{percentage:.2f}",
                'total_selections': total_selections
            })
        
        # Create output filename
        safe_cell_type = cell_type.replace('/', '_').replace(' ', '_')
        safe_pert_name = pert_name.replace('/', '_').replace(' ', '_')
        output_filename = f"{safe_cell_type}_{safe_pert_name}_selection_stats.csv"
        output_path = output_dir / output_filename
        
        # Write CSV
        with open(output_path, 'w', newline='') as f:
            fieldnames = ['perturbation', 'count', 'percentage', 'total_selections']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)
        
        print(f"  Created {output_filename} ({len(output_rows)} unique perturbations)")
    
    # Also create a summary CSV with all conditions
    summary_path = output_dir / "all_conditions_summary.csv"
    print(f"\nCreating summary file: {summary_path}")
    
    with open(summary_path, 'w', newline='') as f:
        fieldnames = ['cell_type', 'pert_name', 'selected_perturbation', 'count', 'percentage', 'total_selections']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for (cell_type, pert_name), all_selected_perts in sorted(condition_data.items()):
            pert_counts = Counter(all_selected_perts)
            total_selections = len(all_selected_perts)
            
            if total_selections == 0:
                continue
            
            for pert, count in pert_counts.most_common():
                percentage = (count / total_selections) * 100
                writer.writerow({
                    'cell_type': cell_type,
                    'pert_name': pert_name,
                    'selected_perturbation': pert,
                    'count': count,
                    'percentage': f"{percentage:.2f}",
                    'total_selections': total_selections
                })
    
    print(f"\nAnalysis complete!")
    print(f"Output saved to: {output_dir}")
    print(f"  - Individual files: {len(condition_data)} CSV files")
    print(f"  - Summary file: all_conditions_summary.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze retrieval_info.csv to compute perturbation selection percentages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze with output in same directory as input
    python analyze_retrieval_info.py experiments/replogle/hepg2_rag32_sparsity_0.1/eval_last.ckpt/retrieval_info.csv
    
    # Specify custom output directory
    python analyze_retrieval_info.py retrieval_info.csv --output-dir ./analysis_results
        """
    )
    
    parser.add_argument(
        'input_csv',
        type=str,
        help='Path to retrieval_info.csv file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save output CSV files (default: same as input file)'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_csv):
        print(f"Error: Input file not found: {args.input_csv}")
        return 1
    
    analyze_retrieval_info(args.input_csv, args.output_dir)
    return 0


if __name__ == '__main__':
    exit(main())
