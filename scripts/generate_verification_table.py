#!/usr/bin/env python3
"""
Script to process verification logs and results and generate LaTeX table
showing verification outcomes by property across all runs.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

def find_json_files(base_path: str, directories: List[str]) -> List[Path]:
    """Find all JSON files in specified directories."""
    json_files = []
    for directory in directories:
        dir_path = Path(base_path) / directory
        if dir_path.exists():
            json_files.extend(dir_path.glob("*.json"))
    return json_files

def extract_verification_data(file_path: Path) -> Dict:
    """Extract verification data from JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Determine solver from filename or content
        filename = file_path.name.lower()
        if 'z3' in filename:
            solver = 'Z3'
        elif 'cvc5' in filename:
            solver = 'CVC5'
        else:
            # Try to get from content
            solver = data.get('verification_session', {}).get('verifier', 'UNKNOWN')

        results = {}

        # Check for z3_solver_details or cvc5_solver_details
        solver_details = data.get('z3_solver_details') or data.get('cvc5_solver_details')

        if solver_details:
            for constraint_name, constraint_data in solver_details.items():
                if isinstance(constraint_data, dict):
                    # Get result
                    result = constraint_data.get('z3_result') or constraint_data.get('cvc5_result', '')
                    if result:
                        result = result.upper()
                        if result == 'SAT':
                            results[constraint_name] = 'SAT'
                        elif result == 'UNSAT':
                            results[constraint_name] = 'UNSAT'
                        elif result == 'UNKNOWN':
                            results[constraint_name] = 'UNKNOWN'

        return {
            'solver': solver,
            'results': results,
            'file': file_path.name
        }

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def aggregate_results(verification_data: List[Dict]) -> Dict:
    """Aggregate verification results by solver and property."""
    # Structure: {property: {solver: {result: count}}}
    aggregated = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for data in verification_data:
        if not data:
            continue

        solver = data['solver']
        for property_name, result in data['results'].items():
            aggregated[property_name][solver][result] += 1

    return aggregated

def generate_latex_table(aggregated: Dict) -> str:
    """Generate LaTeX table from aggregated results."""

    # Get all properties and sort them
    properties = sorted(aggregated.keys())

    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Verification outcomes by property across all runs}")
    latex.append("\\label{tab:verdicts_by_property}")
    latex.append("\\begin{tabular}{lcccccc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Property} & \\multicolumn{3}{c}{\\textbf{CVC5}} & \\multicolumn{3}{c}{\\textbf{Z3}} \\\\")
    latex.append("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}")
    latex.append("& SAT & UNSAT & UNK & SAT & UNSAT & UNK \\\\")
    latex.append("\\midrule")

    # Add rows for each property
    for prop in properties:
        cvc5_sat = aggregated[prop].get('CVC5', {}).get('SAT', 0)
        cvc5_unsat = aggregated[prop].get('CVC5', {}).get('UNSAT', 0)
        cvc5_unk = aggregated[prop].get('CVC5', {}).get('UNKNOWN', 0)

        z3_sat = aggregated[prop].get('Z3', {}).get('SAT', 0)
        z3_unsat = aggregated[prop].get('Z3', {}).get('UNSAT', 0)
        z3_unk = aggregated[prop].get('Z3', {}).get('UNKNOWN', 0)

        # Format property name (replace underscores with spaces and capitalize)
        prop_display = prop.replace('_', ' ').title()

        latex.append(f"{prop_display} & {cvc5_sat} & {cvc5_unsat} & {cvc5_unk} & {z3_sat} & {z3_unsat} & {z3_unk} \\\\")

    # Add totals row
    latex.append("\\midrule")

    total_cvc5_sat = sum(aggregated[p].get('CVC5', {}).get('SAT', 0) for p in properties)
    total_cvc5_unsat = sum(aggregated[p].get('CVC5', {}).get('UNSAT', 0) for p in properties)
    total_cvc5_unk = sum(aggregated[p].get('CVC5', {}).get('UNKNOWN', 0) for p in properties)

    total_z3_sat = sum(aggregated[p].get('Z3', {}).get('SAT', 0) for p in properties)
    total_z3_unsat = sum(aggregated[p].get('Z3', {}).get('UNSAT', 0) for p in properties)
    total_z3_unk = sum(aggregated[p].get('Z3', {}).get('UNKNOWN', 0) for p in properties)

    latex.append(f"\\textbf{{Total}} & {total_cvc5_sat} & {total_cvc5_unsat} & {total_cvc5_unk} & {total_z3_sat} & {total_z3_unsat} & {total_z3_unk} \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    return "\n".join(latex)

def main():
    # Base path
    base_path = Path(__file__).parent.parent

    # Directories to search
    directories = ['logs', 'logs_backup', 'results', 'results_backup']

    print(f"Searching for JSON files in: {base_path}")
    print(f"Directories: {directories}")

    # Find all JSON files
    json_files = find_json_files(base_path, directories)
    print(f"\nFound {len(json_files)} JSON files")

    # Extract verification data
    verification_data = []
    for file_path in json_files:
        data = extract_verification_data(file_path)
        if data:
            verification_data.append(data)

    print(f"Successfully processed {len(verification_data)} files")

    # Aggregate results
    aggregated = aggregate_results(verification_data)

    print(f"\nProperties found: {sorted(aggregated.keys())}")

    # Generate LaTeX table
    latex_table = generate_latex_table(aggregated)

    # Save to file
    output_file = base_path / "verification_outcomes_table.tex"
    with open(output_file, 'w') as f:
        f.write(latex_table)

    print(f"\nLaTeX table saved to: {output_file}")
    print("\n" + "="*80)
    print("Generated LaTeX Table:")
    print("="*80)
    print(latex_table)
    print("="*80)

    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 80)
    for prop in sorted(aggregated.keys()):
        print(f"\n{prop.replace('_', ' ').title()}:")
        for solver in ['CVC5', 'Z3']:
            if solver in aggregated[prop]:
                sat = aggregated[prop][solver].get('SAT', 0)
                unsat = aggregated[prop][solver].get('UNSAT', 0)
                unk = aggregated[prop][solver].get('UNKNOWN', 0)
                print(f"  {solver}: SAT={sat}, UNSAT={unsat}, UNKNOWN={unk}")

if __name__ == "__main__":
    main()
