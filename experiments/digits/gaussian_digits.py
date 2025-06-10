from models.gaussian_model import GaussianNBModel
from core.experiment import Experiment
from utils.report import report_data, ReportMode
from datasets.factory import DatasetFactory
from utils.types import (
    DatasetSourceType,
    SklearnDatasetName,
    DataNoiseConfig,        # Configuração para inferência nos dados (X)
    ParameterNoiseConfig,   # Configuração para inferência nos parâmetros do modelo
    LabelNoiseConfig        # Configuração para inferência nos rótulos (y)
)
from inference.pipeline.inference_pipeline import InferencePipeline
from inference.engine.param_runner import ParameterInferenceEngine


def run_gaussian_without_inference():
    print("\n=== GaussianNB SEM INFERÊNCIA ===")
    base_params = {"var_smoothing": 1e-9}
    model = GaussianNBModel(**base_params)
    dataset = DatasetFactory.create(DatasetSourceType.SKLEARN, name=SklearnDatasetName.DIGITS)
    X_train, X_test, y_train, y_test = dataset.load_data()
    experiment = Experiment(model, dataset)
    metrics = experiment.run(X_train, X_test, y_train, y_test)
    report_data(metrics, mode=ReportMode.PRINT)

def run_gaussian_with_inference():
    print("\n=== GaussianNB COM INFERÊNCIA (data + param + label) ===")

    base_params = {"var_smoothing": 1e-9}
    dataset = DatasetFactory.create(DatasetSourceType.SKLEARN, name=SklearnDatasetName.DIGITS)

    # Separando configs
    data_noise_config = DataNoiseConfig(
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
        conditional_noise=(0, 5.0, 0.2),
        random_missing_block_fraction=0.1,
        distribution_shift_fraction=0.1,
        cluster_swap_fraction=0.1,
        group_outlier_cluster_fraction=0.1,
        temporal_drift_std=0.5,
    )
    label_noise_config = LabelNoiseConfig(
        label_noise_fraction=0.1,
        flip_near_border_fraction=0.1,
        confusion_matrix_noise_level=0.1,
        partial_label_fraction=0.1,
        swap_within_class_fraction=0.1,
    )
    param_noise_config = ParameterNoiseConfig(
        integer_noise=True,
        boolean_flip=True,
        string_mutator=True,
        semantic_mutation=True,
        scale_hyper=True,
        cross_dependency=True,
        random_from_space=True,
        bounded_numeric=True,
        type_cast_perturbation=True,
        enum_boundary_shift=True,
    )

    X_train, X_test, y_train, y_test = dataset.load_data()

    pipeline = InferencePipeline(
        data_noise_config=data_noise_config,
        label_noise_config=label_noise_config,
        X_train=X_train
    )

    param_engine = ParameterInferenceEngine(config=param_noise_config)
    perturbed_params = param_engine.apply(base_params)
    param_log = param_engine.export_log()
    print("🔧 Parâmetros perturbados:", perturbed_params)

    model = GaussianNBModel(**perturbed_params)

    X_train, X_test = pipeline.apply_data_inference(X_train, X_test)
    model.fit(X_train, y_train)

    y_train, y_test = pipeline.apply_label_inference(
        y_train, y_test,
        model=model,
        X_train=X_train,
        X_test=X_test
    )

    experiment = Experiment(model, dataset)
    metrics = experiment.run(X_train, X_test, y_train, y_test)

    report_data(metrics, mode=ReportMode.PRINT)
    report_data(param_log, mode=ReportMode.JSON, file_path='results/gaussian_param_log-digits.json')

def run():
    run_gaussian_without_inference()
    run_gaussian_with_inference()

if __name__ == "__main__":
    run()
