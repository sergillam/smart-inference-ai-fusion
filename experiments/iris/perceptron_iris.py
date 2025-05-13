# experiments/perceptron_iris.py
from models.perceptron_model import PerceptronModel
from datasets.iris_loader import IrisLoader
from core.experiment import Experiment
from inference.inference_engine import InferenceEngine
from inference.param_runner import apply_param_inference
from utils.report import report_data

def run_perceptron_without_inference():
    print("\n=== Perceptron SEM INFERÊNCIA ===")
    
    base_params = {"max_iter": 1000, "tol": 1e-3}
    model = PerceptronModel(base_params)
    dataset = IrisLoader()
    experiment = Experiment(model, dataset)
    metrics = experiment.run()
    report_data(metrics, mode='print')

def run_perceptron_with_inference():
    print("\n=== Perceptron COM INFERÊNCIA ===")

    base_params = {"max_iter": 1000, "tol": 1e-3}
    model, param_log = apply_param_inference(
        model_class=PerceptronModel,
        base_params=base_params,
        seed=42,
        ignore_rules={"tol"}
    )

    dataset = IrisLoader()

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

    report_data(metrics, mode='print')
    report_data(param_log, mode='json', file_path='results/perceptron_param_log.json')

def run():
    run_perceptron_without_inference()
    run_perceptron_with_inference()

if __name__ == "__main__":
    run()
