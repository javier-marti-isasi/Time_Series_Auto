from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Any, Dict

from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from clearml import Dataset, Task
from clearml.automation import HyperParameterOptimizer
from clearml.automation.optuna import OptimizerOptuna
from clearml.automation.parameters import DiscreteParameterRange, UniformIntegerParameterRange, UniformParameterRange

from utils.utils_feature_engineering import (
    ForecastConfig,
    add_time_decay_weights,
    build_feature_columns,
)
from utils.utils_model_training import (
    cast_categorical_features,
    evaluate_catboost,
    resolve_temporal_split_periods,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

PROJECT_NAME = "Time Series Auto"
EXPERIMENT_NAME = "Demo CatBoost Horizon 08"
OPTIMIZER_EXPERIMENT_NAME = f"{EXPERIMENT_NAME} HPO"
PARAMS_MODE = "custom"  # "default", "aggresive" or "custom"
RUN_REMOTELY = True
TRAINING_EXECUTION_QUEUE = "default"
PARAMS_OPTIMIZATION = False
HPO_EXECUTION_QUEUE = "default"
HPO_TOTAL_MAX_JOBS = 20
HPO_MAX_CONCURRENT_JOBS = 2


def load_processed_for_training_dataset(
    input_path: Path,
    config: ForecastConfig,
) -> pd.DataFrame:
    modeling_table = pd.read_parquet(input_path)
    modeling_table[config.period_col] = pd.to_datetime(modeling_table[config.period_col])
    return modeling_table.sort_values(config.group_cols + [config.period_col]).reset_index(drop=True)


def build_demo_config() -> ForecastConfig:
    base_config = ForecastConfig(
        dataset_name="Competition",
        aggregation_level="month",
        use_hwes=False,
    )
    return base_config.with_horizon(1)


def build_demo_params(config: ForecastConfig) -> Dict[str, Any]:

    default_params = {
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "iterations": 300,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": config.random_seed,
        "verbose": 100,
        "allow_writing_files": False,
        "early_stopping_rounds": 50,
    }

    aggresive_params = {
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "iterations": 1000,
        "learning_rate": 0.01,
        "depth": 8,
        "l2_leaf_reg": 5.0,
        "random_seed": config.random_seed,
        "verbose": 100,
        "allow_writing_files": False,
        "early_stopping_rounds": 100,
    }

    custom_params = {
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "iterations": 300,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": config.random_seed,
        "verbose": 100,
        "allow_writing_files": False,
        "early_stopping_rounds": 200,
    }

    if PARAMS_MODE == "default":
        return default_params
    if PARAMS_MODE == "aggresive":
        return aggresive_params
    return custom_params


def run_catboost_hyperparameter_optimization(base_task_id: str) -> None:
    optimizer_task = Task.init(
        project_name=PROJECT_NAME,
        task_name=OPTIMIZER_EXPERIMENT_NAME,
        task_type=Task.TaskTypes.optimizer,
    )
    optimizer_task.connect(
        {
            "base_task_id": base_task_id,
            "execution_queue": HPO_EXECUTION_QUEUE,
            "total_max_jobs": HPO_TOTAL_MAX_JOBS,
            "max_concurrent_jobs": HPO_MAX_CONCURRENT_JOBS,
        },
        name="hpo",
    )

    optimizer = HyperParameterOptimizer(
        base_task_id=base_task_id,
        hyper_parameters=[
            UniformIntegerParameterRange(
                "catboost_params/iterations",
                min_value=300,
                max_value=1000,
                step_size=100,
            ),
            UniformParameterRange(
                "catboost_params/learning_rate",
                min_value=0.005,
                max_value=0.05,
            ),
            UniformIntegerParameterRange(
                "catboost_params/depth",
                min_value=4,
                max_value=10,
            ),
            UniformParameterRange(
                "catboost_params/l2_leaf_reg",
                min_value=1.0,
                max_value=10.0,
            ),
            UniformIntegerParameterRange(
                "catboost_params/early_stopping_rounds",
                min_value=50,
                max_value=200,
                step_size=50,
            ),
            DiscreteParameterRange("config/params_optimization", values=[0]),
        ],
        objective_metric_title="valid_mae",
        objective_metric_series="catboost",
        objective_metric_sign="min",
        optimizer_class=OptimizerOptuna,
        execution_queue=HPO_EXECUTION_QUEUE,
        max_number_of_concurrent_tasks=HPO_MAX_CONCURRENT_JOBS,
        total_max_jobs=HPO_TOTAL_MAX_JOBS,
        max_iteration_per_job=1,
    )
    optimizer.start()
    optimizer.wait()
    optimizer.stop()


def train_catboost_demo_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    cat_features: list[str],
    config: ForecastConfig,
    params: Dict[str, Any],
):
    from catboost import CatBoostRegressor, Pool

    X_train = cast_categorical_features(train_df[feature_cols], cat_features)
    y_train = train_df[config.active_target_scaled_col]
    train_pool = Pool(
        X_train,
        y_train,
        cat_features=cat_features,
        weight=train_df.get("sample_weight"),
    )

    X_valid = cast_categorical_features(valid_df[feature_cols], cat_features)
    y_valid = valid_df[config.active_target_scaled_col]
    valid_pool = Pool(
        X_valid,
        y_valid,
        cat_features=cat_features,
    )

    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    return model


def prepare_training_data(config: ForecastConfig) -> Dict[str, Any]:
    dataset = Dataset.get(
        dataset_name="competition_processed_for_training_month_h1",
        dataset_project="Time Series Auto",
        dataset_version="1.0.2",
        alias="competition_month_h1_input",
    )
    local_dataset_path = Path(dataset.get_local_copy())
    parquet_files = list(local_dataset_path.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet found in dataset {dataset.id}")
    input_path = parquet_files[0]
    modeling_table = load_processed_for_training_dataset(input_path=input_path, config=config)

    _, valid_periods, test_periods = resolve_temporal_split_periods(
        modeling_table=modeling_table,
        config=config,
    )

    feature_cols, cat_features = build_feature_columns(
        modeling_table,
        config=config,
        use_scaled_demand_features=True,
    )

    eligible_table = modeling_table.dropna(
        subset=[config.active_target_col, config.active_target_scaled_col]
    ).copy()

    train_df = eligible_table[
        ~eligible_table[config.period_col].isin(valid_periods | test_periods)
    ].copy()
    valid_df = eligible_table[
        eligible_table[config.period_col].isin(valid_periods)
    ].copy()
    test_df = eligible_table[
        eligible_table[config.period_col].isin(test_periods)
    ].copy()

    train_df = add_time_decay_weights(
        train_df,
        config,
        reference_week=train_df[config.period_col].max(),
    )

    return {
        "input_path": input_path,
        "input_dataset_id": dataset.id,
        "input_dataset_name": dataset.name,
        "input_dataset_version": "1.0.1",
        "modeling_table": modeling_table,
        "feature_cols": feature_cols,
        "cat_features": cat_features,
        "train_df": train_df,
        "valid_df": valid_df,
        "test_df": test_df,
    }


def report_metrics_to_clearml(task: Task, metrics_df: pd.DataFrame, split_name: str) -> None:
    logger = task.get_logger()

    for row in metrics_df.to_dict(orient="records"):
        model_name = str(row["model"])
        horizon = int(row["horizon"])

        for metric_name in ["mae", "rmse", "wape", "bias"]:
            logger.report_scalar(
                title=f"{split_name}_{metric_name}",
                series=model_name,
                value=float(row[metric_name]),
                iteration=horizon,
            )


def report_real_vs_pred_plot(
    task: Task,
    prediction_frame: pd.DataFrame,
    config: ForecastConfig,
    max_groups: int = 10,
    random_seed: int = 42,
) -> None:
    group_frame = prediction_frame.loc[:, config.group_cols].drop_duplicates().sample(
        n=min(max_groups, prediction_frame[config.group_cols].drop_duplicates().shape[0]),
        random_state=random_seed,
    )
    plot_frame = prediction_frame.merge(group_frame, on=config.group_cols, how="inner")
    plot_frame = plot_frame.sort_values(config.group_cols + [config.period_col]).reset_index(drop=True)

    figure = go.Figure()
    for group_values, group_data in plot_frame.groupby(config.group_cols, sort=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        group_label = " | ".join(
            f"{group_col}={group_value}"
            for group_col, group_value in zip(config.group_cols, group_values)
        )
        figure.add_trace(
            go.Scatter(
                x=group_data[config.period_col],
                y=group_data[config.target_col],
                mode="lines+markers",
                name=f"real | {group_label}",
                legendgroup=group_label,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=group_data[config.period_col],
                y=group_data["prediction_catboost"],
                mode="lines+markers",
                name=f"pred | {group_label}",
                legendgroup=group_label,
            )
        )

    figure.update_layout(
        title="Real vs Pred by Random Product-Store Groups",
        xaxis_title=config.period_col,
        yaxis_title=config.target_col,
    )

    task.get_logger().report_plotly(
        title="Real vs Pred by Group",
        series="random_groups",
        figure=figure,
        iteration=config.horizon_periods,
    )


def report_real_vs_pred_scatter(
    task: Task,
    prediction_frame: pd.DataFrame,
    config: ForecastConfig,
) -> None:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=prediction_frame[config.target_col],
            y=prediction_frame["prediction_catboost"],
            mode="markers",
            name="all_predictions",
            marker={"opacity": 0.5},
        )
    )
    figure.update_layout(
        title="Scatter Real vs Pred",
        xaxis_title="real",
        yaxis_title="pred",
    )

    task.get_logger().report_plotly(
        title="Scatter Real vs Pred",
        series="all_test_predictions",
        figure=figure,
        iteration=config.horizon_periods,
    )


def report_worst_predictions_table(
    task: Task,
    prediction_frame: pd.DataFrame,
    config: ForecastConfig,
    top_n: int = 20,
) -> None:
    table_frame = prediction_frame.copy()
    table_frame["abs_error"] = (
        table_frame["prediction_catboost"] - table_frame[config.target_col]
    ).abs()
    selected_columns = (
        config.group_cols
        + [config.period_col, config.target_col, "prediction_catboost", "abs_error"]
    )
    table_frame = table_frame.loc[:, selected_columns]
    table_frame = table_frame.sort_values("abs_error", ascending=False).head(top_n).reset_index(drop=True)

    task.get_logger().report_table(
        title="Worst Test Predictions",
        series="top_absolute_errors",
        iteration=config.horizon_periods,
        table_plot=table_frame,
    )


def report_debug_sample_worst_group(
    task: Task,
    prediction_frame: pd.DataFrame,
    config: ForecastConfig,
) -> None:
    debug_frame = prediction_frame.copy()
    debug_frame["abs_error"] = (
        debug_frame["prediction_catboost"] - debug_frame[config.target_col]
    ).abs()

    worst_group = (
        debug_frame.groupby(config.group_cols, as_index=False)["abs_error"]
        .mean()
        .sort_values("abs_error", ascending=False)
        .head(1)
    )

    worst_group_frame = debug_frame.merge(
        worst_group[config.group_cols],
        on=config.group_cols,
        how="inner",
    ).sort_values(config.period_col)

    if worst_group_frame.empty:
        return

    group_label = " | ".join(
        f"{group_col}={worst_group_frame.iloc[0][group_col]}"
        for group_col in config.group_cols
    )

    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(
        worst_group_frame[config.period_col],
        worst_group_frame[config.target_col],
        marker="o",
        label="real",
    )
    axis.plot(
        worst_group_frame[config.period_col],
        worst_group_frame["prediction_catboost"],
        marker="o",
        label="pred",
    )
    axis.set_title(f"Debug Sample Worst Group: {group_label}")
    axis.set_xlabel(config.period_col)
    axis.set_ylabel(config.target_col)
    axis.legend()
    figure.autofmt_xdate()

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(figure)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")

    task.get_logger().report_image(
        title="Debug Sample Worst Group",
        series=group_label,
        iteration=config.horizon_periods,
        image=image,
    )


def main() -> None:
    config = build_demo_config()
    params = build_demo_params(config)

    # Punto clave ClearML 1:
    # Inicializamos la tarea al principio para que ClearML empiece a registrar
    # ejecucion, entorno, paquetes instalados y consola del experimento.
    task = Task.init(
        project_name=PROJECT_NAME,
        task_name=EXPERIMENT_NAME,
        output_uri=True,
    )

    # Punto clave ClearML 2:
    # `connect` guarda la configuracion e hiperparametros en la UI.
    # Asi luego puedes reproducir el experimento desde ClearML.
    connected_config = task.connect(
        {
            "dataset_name": config.dataset_name,
            "aggregation_level": config.aggregation_level,
            "horizon": config.horizon_periods,
            "test_size_periods": config.test_size_periods,
            "valid_size_periods": config.valid_size_periods,
            "group_cols": config.group_cols,
            "params_mode": PARAMS_MODE,
            "run_remotely": int(RUN_REMOTELY),
            "training_execution_queue": TRAINING_EXECUTION_QUEUE,
            "params_optimization": int(PARAMS_OPTIMIZATION),
            "hpo_execution_queue": HPO_EXECUTION_QUEUE,
        },
        name="config",
    )
    params = task.connect(params, name="catboost_params")

    if bool(int(connected_config["run_remotely"])):
        task.set_base_docker(
            docker_image="python:3.11-bullseye",
            docker_arguments="--ipc=host",
        )
        task.execute_remotely(
            queue_name=str(connected_config["training_execution_queue"]),
            exit_process=True,
        )

    if bool(int(connected_config["params_optimization"])):
        base_task_id = task.id
        task.close()
        run_catboost_hyperparameter_optimization(base_task_id=base_task_id)
        return

    prepared_data = prepare_training_data(config)
    task.connect(
        {
            "dataset_id": prepared_data["input_dataset_id"],
            "dataset_name": prepared_data["input_dataset_name"],
            "dataset_version": prepared_data["input_dataset_version"],
            "dataset_path": str(prepared_data["input_path"]),
        },
        name="input_dataset",
    )

    model = train_catboost_demo_model(
        train_df=prepared_data["train_df"],
        valid_df=prepared_data["valid_df"],
        feature_cols=prepared_data["feature_cols"],
        cat_features=prepared_data["cat_features"],
        config=config,
        params=params,
    )

    valid_metrics, _ = evaluate_catboost(
        model=model,
        test_df=prepared_data["valid_df"],
        feature_cols=prepared_data["feature_cols"],
        cat_features=prepared_data["cat_features"],
        config=config,
    )
    test_metrics, test_predictions = evaluate_catboost(
        model=model,
        test_df=prepared_data["test_df"],
        feature_cols=prepared_data["feature_cols"],
        cat_features=prepared_data["cat_features"],
        config=config,
    )

    mae_value = float(test_metrics.loc[0, "mae"])
    rmse_value = float(test_metrics.loc[0, "rmse"])
    wape_value = float(test_metrics.loc[0, "wape"])
    bias_value = float(test_metrics.loc[0, "bias"])

    task.connect(
        {
            "mae": f"{mae_value:.4f}",
            "rmse": f"{rmse_value:.4f}",
            "wape": f"{wape_value:.4f}",
            "bias": f"{bias_value:.4f}",
        },
        name="metrics",
    )

    task.set_user_properties(
        properties={
            "final_mae": f"{mae_value:.4f}",
            "final_rmse": f"{rmse_value:.4f}",
            "final_wape": f"{wape_value:.4f}",
            "final_bias": f"{bias_value:.4f}",
        }
    )

    # Punto clave ClearML 3:
    # Reportamos metricas manualmente para ver su evolucion en la UI.
    report_metrics_to_clearml(task, valid_metrics, split_name="valid")
    report_metrics_to_clearml(task, test_metrics, split_name="test")
    report_real_vs_pred_plot(task, test_predictions, config)
    report_real_vs_pred_scatter(task, test_predictions, config)
    report_worst_predictions_table(task, test_predictions, config)
    report_debug_sample_worst_group(task, test_predictions, config)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = RESULTS_DIR / (
        f"{config.dataset_slug}_clearml_demo_catboost_{config.aggregation_level}_t_plus_{config.horizon_periods}.cbm"
    )
    predictions_path = RESULTS_DIR / (
        f"{config.dataset_slug}_clearml_demo_predictions_{config.aggregation_level}_t_plus_{config.horizon_periods}.csv"
    )

    model.save_model(model_path)
    test_predictions.to_csv(predictions_path, sep=";", index=False, encoding="utf-8-sig")

    # Punto clave ClearML 4:
    # Subimos artefactos explicitamente para que queden guardados en el experimento:
    # modelo entrenado, predicciones y metricas finales.
    task.upload_artifact(name="catboost_model", artifact_object=str(model_path))
    task.upload_artifact(name="test_predictions", artifact_object=str(predictions_path))
    task.upload_artifact(name="valid_metrics", artifact_object=valid_metrics)
    task.upload_artifact(name="test_metrics", artifact_object=test_metrics)

    print(f"Input dataset: {prepared_data['input_path']}")
    print(f"Train rows: {len(prepared_data['train_df'])}")
    print(f"Valid rows: {len(prepared_data['valid_df'])}")
    print(f"Test rows: {len(prepared_data['test_df'])}")
    print(f"Model saved to: {model_path}")
    print(f"Predictions saved to: {predictions_path}")
    print("\nValidation metrics:")
    print(valid_metrics)
    print("\nTest metrics:")
    print(test_metrics)


if __name__ == "__main__":
    main()
