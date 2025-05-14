from models.perceptron_model import PerceptronModel
from core.experiment import Experiment
from utils.report import report_data, ReportMode
from datasets.factory import DatasetFactory
from utils.types import DatasetSourceType, SklearnDatasetName, DatasetNoiseConfig
from inference.pipeline.inference_pipeline import InferencePipeline

def run_perceptron_without_inference():
    print("\n=== Perceptron SEM INFERÊNCIA ===")
    
    base_params = {"max_iter": 1000, "tol": 1e-3}
    model = PerceptronModel(base_params)
    dataset = DatasetFactory.create(DatasetSourceType.SKLEARN, name=SklearnDatasetName.WINE)

    X_train, X_test, y_train, y_test = dataset.load_data()
    
    experiment = Experiment(model, dataset)
    metrics = experiment.run(X_train, X_test, y_train, y_test)

    report_data(metrics, mode=ReportMode.PRINT)

def run_perceptron_with_inference():
    print("\n=== Perceptron COM INFERÊNCIA (data + param + label) ===")

    base_params = {"max_iter": 1000, "tol": 1e-3}
    dataset = DatasetFactory.create(DatasetSourceType.SKLEARN, name=SklearnDatasetName.WINE)

    dataset_noise_config = DatasetNoiseConfig(
        noise_level=0.2,
        truncate_decimals=1,
        quantize_bins=5,
        cast_to_int=False,
        shuffle_fraction=0.1,
        scale_range=(0.8, 1.2),
        zero_out_fraction=0.05,
        insert_nan_fraction=0.05,
        outlier_fraction=0.05,
        add_dummy_features=2,
        duplicate_features=2,
        feature_selective_noise=(0.3, [0, 2]),
        remove_features=[1, 3],
        feature_swap=[0, 2],
        label_noise_fraction=0.1
    )

    pipeline = InferencePipeline(dataset_noise_config=dataset_noise_config)

    model, param_log = pipeline.apply_param_inference(
        model_class=PerceptronModel,
        base_params=base_params,
        seed=42,
        ignore_rules={"tol"}
    )

    X_train, X_test, y_train, y_test = dataset.load_data()

    X_train, X_test = pipeline.apply_data_inference(X_train, X_test)

    y_train, y_test = pipeline.apply_label_inference(y_train, y_test)

    experiment = Experiment(model, dataset)
    metrics = experiment.run(X_train, X_test, y_train, y_test)

    report_data(metrics, mode=ReportMode.PRINT)
    report_data(param_log, mode=ReportMode.JSON, file_path='results/perceptron_param_log-wine.json')

def run():
    run_perceptron_without_inference()
    run_perceptron_with_inference()

if __name__ == "__main__":
    run()
