import json
import os

def report_data(data: dict, mode: str = 'print', file_path: str = None, label: str = None):
    """
    Relata qualquer dicionário (métricas, logs, configs) em formato print, json ou retorno.

    Args:
        data: dicionário de dados
        mode: 'print', 'json' ou 'dict'
        file_path: caminho para salvar JSON (se modo for 'json')
        label: nome opcional para print (ex: 'Métricas', 'Log de Parâmetros')
    """
    if mode == 'print':
        if label:
            print(f"=== {label} ===")
        for key, value in data.items():
            print(f"{key}: {value}")
    elif mode == 'json' and file_path:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
    elif mode == 'dict':
        return data
    else:
        raise ValueError("Modo inválido: use 'print', 'json' ou 'dict'")
