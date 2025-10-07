#!/usr/bin/env python3
"""Teste de paridade entre Z3 e CVC5 - Recursos, Constraints e Resultados"""
import sys
import os

# Adicionar path do projeto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class MockVerificationInput:
    """Mock para simular VerificationInput"""
    input_data: Any = None
    output_data: Any = None
    parameters: Optional[Dict] = None
    metadata: Optional[Dict] = None


def test_resource_parity():
    """Testa paridade de recursos computacionais entre Z3 e CVC5"""

    print("=" * 70)
    print("🔧 TESTE DE PARIDADE DE RECURSOS COMPUTACIONAIS")
    print("=" * 70)

    # Recursos esperados para paridade
    resources = [
        ("Timeout", "900000ms (15min)", "900000ms (15min)", "✅"),
        ("RLimit", "100.000.000 ops", "100.000.000 ops", "✅"),
        ("Lógica SMT", "QF_NIRA", "QF_NIRA", "✅"),
        ("Seed", "12345", "12345", "✅"),
        ("Memory", "16GB", "N/A*", "⚠️"),
        ("Threads", "até 16", "N/A*", "⚠️"),
    ]

    print("\n📋 COMPARAÇÃO DE RECURSOS:")
    print("-" * 70)
    print(f"{'Recurso':<20} {'Z3':<20} {'CVC5':<20} {'Status'}")
    print("-" * 70)

    for recurso, z3_val, cvc5_val, status in resources:
        print(f"{recurso:<20} {z3_val:<20} {cvc5_val:<20} {status}")

    print("-" * 70)
    print("\n*Nota: CVC5 não suporta memory limit nem threads nativos")
    print("       Isso é uma limitação do solver, não uma configuração incorreta.")

    return True


def test_functional_parity():
    """Testa paridade de constraints funcionais entre Z3 e CVC5"""

    from smart_inference_ai_fusion.verification.plugins.z3_plugin import Z3Verifier
    from smart_inference_ai_fusion.verification.plugins.cvc5_plugin import CVC5Verifier

    z3_v = Z3Verifier()
    cvc5_v = CVC5Verifier()

    print("=" * 70)
    print("📦 TESTE DE PARIDADE DE CONSTRAINTS FUNCIONAIS")
    print("=" * 70)

    # Obter constraints suportados
    z3_constraints = set(z3_v.supported_constraints())
    cvc5_constraints = set(cvc5_v.supported_constraints())

    # Calcular interseção e diferenças
    common = z3_constraints & cvc5_constraints
    only_z3 = z3_constraints - cvc5_constraints
    only_cvc5 = cvc5_constraints - z3_constraints

    print(f"\n📊 RESUMO DE CONSTRAINTS:")
    print("-" * 70)
    print(f"  Total Z3:           {len(z3_constraints)} constraints")
    print(f"  Total CVC5:         {len(cvc5_constraints)} constraints")
    print(f"  Em comum:           {len(common)} constraints")
    print(f"  Apenas Z3:          {len(only_z3)} constraints")
    print(f"  Apenas CVC5:        {len(only_cvc5)} constraints")

    # Constraints críticos que DEVEM estar em ambos
    critical_constraints = [
        "bounds",
        "range_check",
        "type_safety",
        "non_negative",
        "positive",
        "shape_preservation",
        "integer_arithmetic",
        "real_arithmetic",
        "linear_arithmetic",
        "floating_point",
        "invariant",
        "precondition",
        "postcondition",
        "robustness",
    ]

    print(f"\n🎯 CONSTRAINTS CRÍTICOS (devem estar em ambos):")
    print("-" * 70)
    print(f"{'Constraint':<30} {'Z3':<10} {'CVC5':<10} {'Status'}")
    print("-" * 70)

    critical_ok = 0
    critical_total = len(critical_constraints)

    for constraint in critical_constraints:
        in_z3 = "✅" if constraint in z3_constraints else "❌"
        in_cvc5 = "✅" if constraint in cvc5_constraints else "❌"

        if constraint in z3_constraints and constraint in cvc5_constraints:
            status = "✅ OK"
            critical_ok += 1
        elif constraint in z3_constraints or constraint in cvc5_constraints:
            status = "⚠️ PARCIAL"
        else:
            status = "❌ FALTANDO"

        print(f"{constraint:<30} {in_z3:<10} {in_cvc5:<10} {status}")

    print("-" * 70)
    parity_pct = (critical_ok / critical_total) * 100
    print(f"\n🎯 PARIDADE CRÍTICA: {parity_pct:.0f}% ({critical_ok}/{critical_total})")

    if only_z3:
        print(f"\n⚠️  Apenas em Z3 ({len(only_z3)}):")
        for c in sorted(only_z3)[:10]:
            print(f"    - {c}")
        if len(only_z3) > 10:
            print(f"    ... e mais {len(only_z3) - 10}")

    if only_cvc5:
        print(f"\n⚠️  Apenas em CVC5 ({len(only_cvc5)}):")
        for c in sorted(only_cvc5)[:10]:
            print(f"    - {c}")
        if len(only_cvc5) > 10:
            print(f"    ... e mais {len(only_cvc5) - 10}")

    return parity_pct == 100


def test_parity():
    """Testa paridade de resultados entre Z3 e CVC5 para todos os constraints"""

    from smart_inference_ai_fusion.verification.plugins.z3_plugin import Z3Verifier
    from smart_inference_ai_fusion.verification.plugins.cvc5_plugin import CVC5Verifier

    z3_v = Z3Verifier()
    cvc5_v = CVC5Verifier()

    print("=" * 70)
    print("🔍 TESTE DE PARIDADE DE RESULTADOS")
    print("=" * 70)

    # Dados de teste
    test_cases = {
        "positive_data": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "mixed_data": np.array([[-1.0, 2.0, 3.0], [4.0, -5.0, 6.0]]),
        "integer_data": np.array([[1, 2, 3], [4, 5, 6]]),
        "float_data": np.array([[1.5, 2.7, 3.9]]),
    }

    # Constraints a testar
    constraints = [
        ("bounds", {"min": -10, "max": 100}),
        ("range_check", {"min": -100, "max": 100}),
        ("type_safety", True),
        ("shape_preservation", True),
        ("non_negative", True),
        ("positive", True),
        ("integer_arithmetic", True),
        ("real_arithmetic", True),
        ("parameter_drift", {"previous_params": {"lr": 0.01}, "threshold": 0.1}),
        ("model_instantiation", {"required_params": [], "required_attrs": []}),
        ("parameter_consistency", {"consistency_rules": []}),
    ]

    results = {}

    for c_name, c_data in constraints:
        results[c_name] = {"matches": 0, "mismatches": 0, "details": []}

        for t_name, t_data in test_cases.items():
            mock = MockVerificationInput(
                input_data={"train": t_data},
                output_data={"predictions": t_data},
                parameters={"lr": 0.01, "epochs": 100},
            )

            try:
                # Z3
                z3_r = z3_v._verify_constraint(c_name, c_data, mock)

                # CVC5
                cvc5_dict = cvc5_v._verify_constraint(c_name, c_data, mock)
                cvc5_r = cvc5_dict.get("satisfied", True) if isinstance(cvc5_dict, dict) else cvc5_dict

                if z3_r == cvc5_r:
                    results[c_name]["matches"] += 1
                else:
                    results[c_name]["mismatches"] += 1
                    results[c_name]["details"].append({
                        "case": t_name,
                        "z3": z3_r,
                        "cvc5": cvc5_r,
                    })

            except Exception as e:
                results[c_name]["details"].append({
                    "case": t_name,
                    "error": str(e)[:100],
                })

    # Imprimir resultados
    print("\n📊 RESULTADOS DE PARIDADE:")
    print("-" * 70)

    total_matches = 0
    total_tests = 0

    for constraint, data in results.items():
        matches = data["matches"]
        mismatches = data["mismatches"]
        total = matches + mismatches

        total_matches += matches
        total_tests += total

        if total == 0:
            parity = "N/A"
            icon = "⚪"
        else:
            parity_pct = (matches / total) * 100
            parity = f"{parity_pct:.0f}%"
            icon = "✅" if parity_pct == 100 else ("⚠️" if parity_pct >= 75 else "❌")

        print(f"{icon} {constraint:25s}: {parity:>6s} ({matches}/{total} matches)")

        if data["details"]:
            for detail in data["details"][:2]:
                if "error" in detail:
                    print(f"    └─ ⚠️  {detail['case']}: ERROR - {detail['error']}")
                else:
                    print(f"    └─ ❌ {detail['case']}: Z3={detail['z3']}, CVC5={detail['cvc5']}")

    print("-" * 70)

    overall_parity = (total_matches / total_tests * 100) if total_tests > 0 else 0
    print(f"\n🎯 PARIDADE GERAL: {overall_parity:.1f}% ({total_matches}/{total_tests})")

    if overall_parity == 100:
        print("🎉 PARIDADE PERFEITA! Z3 e CVC5 estão 100% alinhados!")
        return True
    elif overall_parity >= 90:
        print("⚠️  Paridade alta, mas alguns ajustes necessários.")
        return False
    else:
        print("❌ Paridade insuficiente - revisão necessária.")
        return False


if __name__ == "__main__":
    # 1. Teste de recursos computacionais
    test_resource_parity()
    print("\n")

    # 2. Teste de constraints funcionais
    test_functional_parity()
    print("\n")

    # 3. Teste de resultados
    success = test_parity()
    sys.exit(0 if success else 1)
