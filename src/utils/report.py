import json
import os
from utils.types import ReportMode


def report_data(
    data: dict,
    mode: ReportMode = ReportMode.PRINT,
    file_path: str = None,
    label: str = None
) -> None:
    """
    Relata um dicionário de dados (métricas, logs, configs) em diferentes formatos.

    Args:
        data (dict): Dicionário contendo os dados a serem reportados.
        mode (ReportMode): Modo de saída. Pode ser ReportMode.PRINT ou ReportMode.JSON.
        file_path (str, opcional): Caminho para salvar o arquivo JSON (usado apenas se mode for JSON).
        label (str, opcional): Rótulo para exibição no modo PRINT (ex: 'Métricas', 'Parâmetros').

    Raises:
        ValueError: Se o modo informado for inválido ou se faltar `file_path` no modo JSON.
    """
    if mode == ReportMode.PRINT:
        if label:
            print(f"=== {label} ===")
        for key, value in data.items():
            print(f"{key}: {value}")

    elif mode == ReportMode.JSON:
        if not file_path:
            raise ValueError("É necessário fornecer 'file_path' para salvar como JSON.")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    else:
        raise ValueError(f"Modo inválido: {mode}")
