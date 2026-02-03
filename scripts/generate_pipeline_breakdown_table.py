#!/usr/bin/env python3
"""Generate verification breakdown by pipeline stage."""

import json
import os
from collections import defaultdict
from pathlib import Path


def analyze_logs():
    """Analyze verification logs by pipeline stage."""
    base_dir = Path(__file__).parent.parent
    logs_dir = base_dir / "logs"

    # Pipeline stages mapping
    stages = {
        "data": "Transformação de Dados",
        "labels": "Transformação de Labels",
        "parameters_pre": "Parâmetros (Pré-Perturbação)",
        "parameters_post": "Parâmetros (Pós-Perturbação)",
        "model_integrity": "Integridade do Modelo",
    }

    # Results structure: stage -> solver -> result_type -> count
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    # Constraint results by stage
    constraint_results = defaultdict(lambda: defaultdict(lambda: {"satisfied": [], "violated": []}))

    for json_file in logs_dir.glob("*.json"):
        filename = json_file.name

        # Determine solver
        if filename.startswith("cvc5"):
            solver = "CVC5"
        elif filename.startswith("z3"):
            solver = "Z3"
        else:
            continue

        # Determine stage
        stage = None
        if "data_data" in filename or "data_input" in filename or "data_output" in filename:
            stage = "data"
        elif "labels" in filename:
            stage = "labels"
        elif "parameters_pre" in filename:
            stage = "parameters_pre"
        elif "parameters_post" in filename:
            stage = "parameters_post"
        elif "model_integrity" in filename:
            stage = "model_integrity"

        if not stage:
            continue

        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            # Get verification result
            ver_result = data.get("verification_result", data)

            # Get constraint results
            constraint_res = ver_result.get("constraint_results", {})
            satisfied = constraint_res.get("satisfied", [])
            violated = constraint_res.get("violated", [])

            results[stage][solver]["satisfied"] += len(satisfied)
            results[stage][solver]["violated"] += len(violated)
            results[stage][solver]["total_files"] += 1

            # Store individual constraints
            for c in satisfied:
                constraint_results[stage][solver]["satisfied"].append(c)
            for c in violated:
                constraint_results[stage][solver]["violated"].append(c)

        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

    return results, constraint_results, stages


def print_summary(results, constraint_results, stages):
    """Print detailed summary."""
    print("=" * 80)
    print("ANÁLISE DE VERIFICAÇÃO POR ETAPA DO PIPELINE")
    print("=" * 80)
    print()

    # Summary table data
    table_data = []

    for stage_key, stage_name in stages.items():
        print(f"\n{'='*60}")
        print(f"📊 {stage_name.upper()}")
        print(f"{'='*60}")

        cvc5_sat = results[stage_key]["CVC5"]["violated"]  # SAT = violação encontrada
        cvc5_unsat = results[stage_key]["CVC5"]["satisfied"]  # UNSAT = propriedade verificada
        z3_sat = results[stage_key]["Z3"]["violated"]
        z3_unsat = results[stage_key]["Z3"]["satisfied"]

        print(f"\n  CVC5:")
        print(f"    SAT (violações detectadas):    {cvc5_sat}")
        print(f"    UNSAT (propriedades OK):       {cvc5_unsat}")
        print(f"    Arquivos analisados:           {results[stage_key]['CVC5']['total_files']}")

        print(f"\n  Z3:")
        print(f"    SAT (violações detectadas):    {z3_sat}")
        print(f"    UNSAT (propriedades OK):       {z3_unsat}")
        print(f"    Arquivos analisados:           {results[stage_key]['Z3']['total_files']}")

        # Constraints breakdown
        print(f"\n  Constraints Satisfeitas (UNSAT):")
        cvc5_satisfied = set(constraint_results[stage_key]["CVC5"]["satisfied"])
        z3_satisfied = set(constraint_results[stage_key]["Z3"]["satisfied"])
        print(f"    CVC5: {sorted(cvc5_satisfied) if cvc5_satisfied else 'Nenhuma'}")
        print(f"    Z3:   {sorted(z3_satisfied) if z3_satisfied else 'Nenhuma'}")

        print(f"\n  Constraints Violadas (SAT - contraexemplo encontrado):")
        cvc5_violated = set(constraint_results[stage_key]["CVC5"]["violated"])
        z3_violated = set(constraint_results[stage_key]["Z3"]["violated"])
        print(f"    CVC5: {sorted(cvc5_violated) if cvc5_violated else 'Nenhuma'}")
        print(f"    Z3:   {sorted(z3_violated) if z3_violated else 'Nenhuma'}")

        table_data.append({
            "stage": stage_name,
            "cvc5_sat": cvc5_sat,
            "cvc5_unsat": cvc5_unsat,
            "z3_sat": z3_sat,
            "z3_unsat": z3_unsat,
        })

    return table_data


def generate_latex_table(table_data):
    """Generate LaTeX table by pipeline stage."""
    latex = r"""\begin{table}[htbp]
\centering
\caption{Resultados de verificação por etapa do pipeline}
\label{tab:verdicts_by_pipeline_stage}
\begin{tabular}{lcccc}
\toprule
\textbf{Etapa do Pipeline} & \multicolumn{2}{c}{\textbf{CVC5}} & \multicolumn{2}{c}{\textbf{Z3}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
& SAT & UNSAT & SAT & UNSAT \\
\midrule
"""

    total_cvc5_sat = 0
    total_cvc5_unsat = 0
    total_z3_sat = 0
    total_z3_unsat = 0

    for row in table_data:
        latex += f"{row['stage']} & {row['cvc5_sat']} & {row['cvc5_unsat']} & {row['z3_sat']} & {row['z3_unsat']} \\\\\n"
        total_cvc5_sat += row['cvc5_sat']
        total_cvc5_unsat += row['cvc5_unsat']
        total_z3_sat += row['z3_sat']
        total_z3_unsat += row['z3_unsat']

    latex += r"""\midrule
\textbf{Total} & """ + f"{total_cvc5_sat} & {total_cvc5_unsat} & {total_z3_sat} & {total_z3_unsat}" + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def main():
    results, constraint_results, stages = analyze_logs()
    table_data = print_summary(results, constraint_results, stages)

    # Generate LaTeX
    latex = generate_latex_table(table_data)

    print("\n" + "=" * 80)
    print("TABELA LATEX - POR ETAPA DO PIPELINE")
    print("=" * 80)
    print(latex)

    # Save LaTeX
    base_dir = Path(__file__).parent.parent
    output_file = base_dir / "verification_pipeline_breakdown.tex"
    with open(output_file, "w") as f:
        f.write(latex)
    print(f"\nTabela salva em: {output_file}")


if __name__ == "__main__":
    main()
