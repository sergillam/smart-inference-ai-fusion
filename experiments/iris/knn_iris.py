from models.knn_model import KNNModel
from core.experiment import Experiment
from utils.report import report_data, ReportMode
from datasets.factory import DatasetFactory
from utils.types import DatasetSourceType, SklearnDatasetName,DatasetNoiseConfig
from inference.pipeline.inference_pipeline import InferencePipeline

def run_knn_without_inference():
    print("\n=== KNN SEM INFERÊNCIA ===")

    base_params = {"n_neighbors": 3, "weights": "uniform", "algorithm": "auto"}
    model = KNNModel(base_params)
    dataset = DatasetFactory.create(DatasetSourceType.SKLEARN, name=SklearnDatasetName.IRIS)

    X_train, X_test, y_train, y_test = dataset.load_data()
    
    experiment = Experiment(model, dataset)
    metrics = experiment.run(X_train, X_test, y_train, y_test)
    
    report_data(metrics, mode=ReportMode.PRINT)

def run_knn_with_inference():
    print("\n=== KNN COM INFERÊNCIA (data + param + label) ===")

    base_params = {"n_neighbors": 3, "weights": "uniform", "algorithm": "auto"}
    dataset = DatasetFactory.create(DatasetSourceType.SKLEARN, name=SklearnDatasetName.IRIS)

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

    pipeline = InferencePipeline(
        dataset_noise_config=dataset_noise_config
    )

    model, param_log = pipeline.apply_param_inference(
        model_class=KNNModel,
        base_params=base_params,
        seed=42,
        ignore_rules={"weights"}
    )

    X_train, X_test, y_train, y_test = dataset.load_data()

    X_train, X_test = pipeline.apply_data_inference(X_train, X_test)

    y_train, y_test = pipeline.apply_label_inference(y_train, y_test)

    experiment = Experiment(model, dataset)
    metrics = experiment.run(X_train, X_test, y_train, y_test)

    report_data(metrics, mode=ReportMode.PRINT)
    report_data(param_log, mode=ReportMode.JSON, file_path='results/knn_param_log-iris.json')


def run():
    run_knn_without_inference()
    run_knn_with_inference()

if __name__ == "__main__":
    run()
