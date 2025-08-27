"""Experiment script for Random Forest Regressor on the digits dataset."""
from core.experiment import Experiment
from datasets.factory import DatasetFactory
from inference.engine.param_runner import ParameterInferenceEngine
from inference.pipeline.inference_pipeline import InferencePipeline
from src.models.random_forest_regressor_model import RandomForestRegressorModel
from src.utils.report import generate_experiment_filename
from utils.preprocessing import filter_sklearn_params
from utils.report import report_data, ReportMode
from utils.types import (
    DataNoiseConfig,
    DatasetSourceType,
    LabelNoiseConfig,
    ParameterNoiseConfig,
    SklearnDatasetName
)
# pylint: disable=duplicate-code
def run_rf_regressor_without_inference():
    """Run the Random Forest Regressor baseline (no inference or perturbations).

    Loads the Digits dataset, trains a RandomForest model, evaluates it,
    and reports results via report_data.
    """
    model_class = RandomForestRegressorModel
    dataset_name = SklearnDatasetName.DIGITS
    name_output = generate_experiment_filename(model_class, dataset_name)

    report_data("=== RandomForestRegressor WITHOUT INFERENCE ===", mode=ReportMode.PRINT)
    base_params = {"n_estimators": 100, "random_state": 42}
    filtered_params = filter_sklearn_params(base_params, RandomForestRegressorModel)
    model = model_class(**filtered_params)
    dataset = DatasetFactory.create(DatasetSourceType.SKLEARN, name=dataset_name)
    X_train, X_test, y_train, y_test = dataset.load_data()
    experiment = Experiment(model, dataset)
    metrics = experiment.run(X_train, X_test, y_train, y_test)

    report_data("Evaluation metrics (no inference):", mode=ReportMode.PRINT)
    report_data(metrics, mode=ReportMode.PRINT)
    report_data(metrics, mode=ReportMode.JSON_RESULT, name_output=f"{name_output}-no-inference")

def run_rf_regressor_with_inference():
    """Runs the RF Regressor experiment with a full inference pipeline.

    This includes applying data, parameter, and label perturbations before
    training and evaluation to test model robustness.
    """
    model_class = RandomForestRegressorModel
    dataset_name = SklearnDatasetName.DIGITS
    name_output = generate_experiment_filename(model_class, dataset_name)

    report_data("=== RandomForestRegressor WITH INFERENCE (data + param + label) ===",
                mode=ReportMode.PRINT)

    base_params = {"n_estimators": 100, "random_state": 42}
    dataset = DatasetFactory.create(DatasetSourceType.SKLEARN, name=dataset_name)

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
        # flip_near_border_fraction=0.1,  # Excluded for regression: depends on predict_proba
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
    report_data(f"Perturbed parameters: {perturbed_params}", mode=ReportMode.PRINT)
    report_data(param_log, mode=ReportMode.JSON_LOG, name_output=f"{name_output}-param-perturb")

    filtered_params = filter_sklearn_params(perturbed_params, RandomForestRegressorModel)
    model = RandomForestRegressorModel(**filtered_params)

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

    report_data("Evaluation metrics (with inference):", mode=ReportMode.PRINT)
    report_data(metrics, mode=ReportMode.PRINT)
    report_data(metrics, mode=ReportMode.JSON_RESULT, name_output=f"{name_output}-with-inference")

def run():
    """Runs the complete experiment suite for the Random Forest Regressor.

    This executes both the baseline (without inference) and the full
    (with inference) experimental runs.
    """
    run_rf_regressor_without_inference()
    run_rf_regressor_with_inference()
