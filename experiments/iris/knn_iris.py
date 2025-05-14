from models.knn_model import KNNModel
from core.experiment import Experiment
from inference.inference_engine import InferenceEngine
from inference.param_runner import apply_param_inference
from utils.report import report_data, ReportMode
from datasets.factory import DatasetFactory
from utils.types import DatasetSourceType,SklearnDatasetName

def run_knn_without_inference():
    print("\n=== KNN SEM INFERÊNCIA ===")
    
    base_params = {"n_neighbors": 3, "weights": "uniform", "algorithm": "auto"}
    model = KNNModel(base_params)
    dataset = DatasetFactory.create(DatasetSourceType.SKLEARN, name=SklearnDatasetName.IRIS)
    experiment = Experiment(model, dataset)
    metrics = experiment.run()
    report_data(metrics, mode=ReportMode.PRINT)

def run_knn_with_inference():
    print("\n=== KNN COM INFERÊNCIA (data + param) ===")

    base_params = {"n_neighbors": 3, "weights": "uniform", "algorithm": "auto"}

    model, param_log = apply_param_inference(
        model_class=KNNModel,
        base_params=base_params,
        seed=42,
        ignore_rules={"weights"}
    )

    dataset = DatasetFactory.create(DatasetSourceType.SKLEARN, name=SklearnDatasetName.IRIS)
    config = {
        'noise_level': 0.2,
        'truncate_decimals': 1,
        'quantize_bins': 5,
        'cast_to_int': False,
        'shuffle_fraction': 0.1,
        'scale_range': (0.8, 1.2),
        'zero_out_fraction': 0.05,
        'insert_nan_fraction': 0.05,
        'outlier_fraction': 0.05,
        'add_dummy_features': 2,
        'duplicate_features': 2,
        'feature_selective_noise': (0.3, [0, 2]),
        'remove_features': [1, 3],
        'feature_swap': [0, 2]
    }

    inference = InferenceEngine(config)
    experiment = Experiment(model, dataset, inference=inference)
    metrics = experiment.run()

    report_data(metrics, mode=ReportMode.PRINT)
    report_data(param_log, mode=ReportMode.JSON, file_path='results/knn_param_log.json')

def run():
    run_knn_without_inference()
    run_knn_with_inference()

if __name__ == "__main__":
    run()
