"""Extensão do sistema de report para incluir resultados de verificação formal."""

import json
from typing import Dict, Any, Optional
from smart_inference_ai_fusion.utils.report import report_data, ReportMode, generate_experiment_filename
from smart_inference_ai_fusion.utils.logging import logger
from smart_inference_ai_fusion.verification.core.plugin_interface import VerificationResult


def report_verification_results(
    verification_result: VerificationResult,
    model_name: str,
    dataset_name: str,
    transformation_name: Optional[str] = None
) -> None:
    """Reporta resultados detalhados da verificação formal.
    
    Args:
        verification_result: Resultado da verificação Z3
        model_name: Nome do modelo
        dataset_name: Nome do dataset  
        transformation_name: Nome da transformação (opcional)
    """
    # Criar estrutura detalhada dos resultados
    verification_details = {
        "verification_info": {
            "verifier": verification_result.verifier_name,
            "status": verification_result.status.value,
            "execution_time_seconds": verification_result.execution_time,
            "message": verification_result.message,
            "success": verification_result.success
        },
        "constraints_analysis": {
            "total_checked": len(verification_result.constraints_checked),
            "total_satisfied": len(verification_result.constraints_satisfied),
            "total_violated": len(verification_result.constraints_violated),
            "constraints_checked": verification_result.constraints_checked,
            "constraints_satisfied": verification_result.constraints_satisfied,
            "constraints_violated": verification_result.constraints_violated
        },
        "context": {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "transformation_name": transformation_name or "pipeline_verification"
        },
        "solver_details": verification_result.details
    }
    
    # 1. Reportar no console/log com detalhes
    _report_console_verification(verification_details)
    
    # 2. Salvar resultado detalhado em JSON (results/)
    _save_verification_results(verification_details, model_name, dataset_name, transformation_name)
    
    # 3. Adicionar ao log estruturado (logs/)
    _log_verification_details(verification_details)


def _report_console_verification(details: Dict[str, Any]) -> None:
    """Reporta verificação no console com formatação bonita."""
    info = details["verification_info"]
    constraints = details["constraints_analysis"]
    context = details["context"]
    
    # Header
    status_icon = "✅" if info["success"] else "❌"
    report_data(
        f"🔍 {status_icon} FORMAL VERIFICATION - {info['verifier']}",
        mode=ReportMode.PRINT
    )
    
    # Context
    report_data(
        f"   📊 Contexto: {context['model_name']} → {context['dataset_name']}",
        mode=ReportMode.PRINT
    )
    
    if context["transformation_name"] != "pipeline_verification":
        report_data(
            f"   🔧 Transformation: {context['transformation_name']}",
            mode=ReportMode.PRINT
        )
    
    # Status e timing
    report_data(
        f"   ⚡ Status: {info['status'].upper()} ({info['execution_time_seconds']:.3f}s)",
        mode=ReportMode.PRINT
    )
    
    # Constraints summary
    if constraints["total_checked"] > 0:
        report_data(
            f"   📋 Constraints: {constraints['total_satisfied']}/{constraints['total_checked']} satisfied",
            mode=ReportMode.PRINT
        )
        
        # List satisfied constraints
        if constraints["constraints_satisfied"]:
            satisfied_list = ", ".join(constraints["constraints_satisfied"])
            report_data(
                f"   ✅ Satisfied: {satisfied_list}",
                mode=ReportMode.PRINT
            )
        
        # List violated constraints (if any)
        if constraints["constraints_violated"]:
            violated_list = ", ".join(constraints["constraints_violated"])
            report_data(
                f"   ❌ Violated: {violated_list}",
                mode=ReportMode.PRINT
            )
    
    # Mensagem adicional do solver
    if info["message"]:
        report_data(
            f"   💬 Z3: {info['message']}",
            mode=ReportMode.PRINT
        )


def _save_verification_results(details: Dict[str, Any], model_name: str, dataset_name: str, transformation_name: Optional[str]) -> None:
    """Salva resultados detalhados da verificação em JSON."""
    # Gerar nome do arquivo
    suffix = f"verification-{transformation_name}" if transformation_name else "verification-pipeline"
    filename = generate_experiment_filename(
        model_class=type(model_name, (), {}),  # Fake class for filename generation
        dataset_name=dataset_name,
        suffix=suffix
    )
    
    # Salvar no diretório results/
    report_data(
        content=details,
        mode=ReportMode.JSON_RESULT,
        name_output=filename
    )


def _log_verification_details(details: Dict[str, Any]) -> None:
    """Adiciona detalhes de verificação ao log estruturado."""
    # Gerar nome do log
    context = details["context"]
    log_name = f"verification-{context['model_name']}-{context['dataset_name']}"
    
    # Log estruturado
    log_entry = {
        "timestamp": details["verification_info"].get("timestamp"),
        "verification_result": details
    }
    
    report_data(
        content=log_entry,
        mode=ReportMode.JSON_LOG,
        name_output=log_name
    )


def enhance_experiment_results_with_verification(
    experiment_results: Dict[str, Any],
    verification_results: list[VerificationResult]
) -> Dict[str, Any]:
    """Adiciona resultados de verificação aos resultados do experimento.
    
    Args:
        experiment_results: Resultados existentes do experimento
        verification_results: Lista de resultados de verificação
        
    Returns:
        Dict com resultados expandidos incluindo verificação
    """
    if not verification_results:
        return experiment_results
    
    # Adicionar seção de verificação formal
    verification_summary = {
        "formal_verification": {
            "enabled": True,
            "total_verifications": len(verification_results),
            "successful_verifications": sum(1 for vr in verification_results if vr.success),
            "failed_verifications": sum(1 for vr in verification_results if not vr.success),
            "total_execution_time": sum(vr.execution_time for vr in verification_results),
            "verifiers_used": list(set(vr.verifier_name for vr in verification_results)),
            "detailed_results": []
        }
    }
    
    # Adicionar detalhes de cada verificação
    for vr in verification_results:
        verification_detail = {
            "verifier": vr.verifier_name,
            "status": vr.status.value,
            "execution_time": vr.execution_time,
            "constraints_checked": len(vr.constraints_checked),
            "constraints_satisfied": len(vr.constraints_satisfied),
            "constraints_violated": len(vr.constraints_violated),
            "message": vr.message,
            "success": vr.success
        }
        verification_summary["formal_verification"]["detailed_results"].append(verification_detail)
    
    # Atualizar resultados do experimento
    enhanced_results = experiment_results.copy()
    enhanced_results.update(verification_summary)
    
    return enhanced_results


def report_solver_sat_unsat_details(solver_details: Dict[str, Any], constraint_name: str) -> None:
    """Reporta detalhes específicos de SAT/UNSAT do solver Z3.
    
    Args:
        solver_details: Detalhes do solver (modelo, core insatisfazível, etc.)
        constraint_name: Nome do constraint sendo verificado
    """
    if not solver_details:
        return
    
    # Reportar resultado SAT/UNSAT
    if "satisfiable" in solver_details:
        sat_status = "SAT" if solver_details["satisfiable"] else "UNSAT"
        report_data(
            f"   🧮 Z3 Result for '{constraint_name}': {sat_status}",
            mode=ReportMode.PRINT
        )
    
    # Reportar modelo (se SAT)
    if solver_details.get("model"):
        report_data(
            f"   📝 Z3 Model: {solver_details['model']}",
            mode=ReportMode.PRINT
        )
    
    # Reportar core insatisfazível (se UNSAT)
    if solver_details.get("unsat_core"):
        report_data(
            f"   🔍 Z3 Unsat Core: {solver_details['unsat_core']}",
            mode=ReportMode.PRINT
        )
    
    # Reportar estatísticas do solver
    if solver_details.get("statistics"):
        stats = solver_details["statistics"]
        report_data(
            f"   📊 Z3 Stats: {stats}",
            mode=ReportMode.PRINT
        )
