"""Report data utility for the inference framework."""

import os
import json
import re
from datetime import datetime, timezone
from typing import Any, Optional
from utils.types import ReportMode
from utils.logging import logger


def get_iso_timestamp():
    """Returns current UTC time as a filename-safe ISO string (YYYY-MM-DDTHH-MM-SSZ)."""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H-%M-%SZ")

def generate_experiment_filename(
    model_class: Any,
    dataset_name: Any,
    suffix: Optional[str] = None,
    max_len: int = 100
) -> str:
    """Generates a standardized, filesystem-safe filename for experiment outputs or logs.

    Args:
        model_class (Any): The model class or instance (e.g., GaussianNBModel or SVC).
        dataset_name (Any): The dataset identifier (e.g., str, Enum, or dataset class).
        suffix (Optional[str]): Optional extra info to append (e.g., 'seed42', 'fold1',
            'scenario-a').
        max_len (int): Maximum length for the generated filename (default: 100 characters).

    Returns:
        str: A cleaned, lowercased string in the format 'modelname-datasetname[-suffix]'.

    Example:
        >>> generate_experiment_filename(GaussianNBModel, SklearnDatasetName.DIGITS,
            suffix='seed-42')
        'gaussiannbmodel-digits-seed-42'
    """
    if hasattr(model_class, "__name__"):
        model_name = model_class.__name__
    else:
        model_name = model_class.__class__.__name__
    if hasattr(dataset_name, "name"):
        dataset = dataset_name.name
    elif hasattr(dataset_name, "value"):
        dataset = dataset_name.value
    else:
        dataset = str(dataset_name)
    parts = [model_name, dataset]
    if suffix:
        parts.append(str(suffix))
    filename = "-".join(parts)
    # Remove invalid filename characters, lowercase, and truncate
    filename = re.sub(r"[^\w\-]", "_", filename).lower()
    filename = filename[:max_len]
    return filename

def report_data(
    content: dict | str,
    mode: ReportMode,
    name_output: str = None
) -> None:
    """Handles reporting, logging, and persisting of experiment results or logs.

    Args:
        content (dict or str): The message or data to output (printed or saved).
        mode (ReportMode): Output mode: PRINT (console/log), JSON_LOG (logs/), 
            or JSON_RESULT (results/).
        name_output (str, optional): Base file name (without extension) for file output modes. 
            The function will add the date and '.json' extension automatically.

    Raises:
        ValueError: If a file output mode is selected and name_output is not provided,
            or if an unknown mode is given.

    Example:
        >>> report_data({'acc': 0.87}, mode=ReportMode.PRINT)
        >>> report_data(metrics_dict, mode=ReportMode.JSON_RESULT, name_output='GaussianNB-digits')
    """
    if mode == ReportMode.PRINT:
        logger.info("%s", content)
    elif mode in {ReportMode.JSON_LOG, ReportMode.JSON_RESULT}:
        if not name_output:
            raise ValueError("name_output is required for file output modes.")
        name_output = re.sub(r"[^\w\-]", "_", name_output)
        timestamp = get_iso_timestamp()
        folder = "logs" if mode == ReportMode.JSON_LOG else "results"
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{name_output}-{timestamp}.json")
        with open(path, "w", encoding="utf-8") as f:
            if isinstance(content, str):
                f.write(content)
            else:
                json.dump(content, f, indent=2)
        logger.info("Saved %s to %s", mode.value, path)
    else:
        logger.error("Invalid mode: %s", mode)
        logger.error("content: %s", content)
        logger.error("name_output: %s", name_output)
        raise ValueError(f"Unknown mode: {mode}")
