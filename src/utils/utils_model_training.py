from __future__ import annotations

from math import ceil
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.utils_feature_engineering import (
    EPS,
    ForecastConfig,
    add_time_decay_weights,
    build_feature_columns,
    build_modeling_table,
)


def get_active_target_columns(config: ForecastConfig) -> Tuple[str, str]:
    return config.active_target_col, config.active_target_scaled_col


def _normalize_split_percentage(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None

    value = float(value)
    if value <= 0:
        return None
    if value > 1:
        value = value / 100.0
    if value >= 1:
        raise ValueError("Split percentages must be lower than 100%.")

    return value


def resolve_split_sizes(
    num_periods: int,
    config: ForecastConfig = ForecastConfig(),
    test_size_periods: Optional[int] = None,
    valid_size_periods: Optional[int] = None,
) -> Tuple[int, int]:
    test_percentage = _normalize_split_percentage(config.per_test_size)
    valid_percentage = _normalize_split_percentage(config.per_valid_size)

    resolved_test_size = test_size_periods if test_size_periods is not None else config.test_size_periods
    resolved_valid_size = valid_size_periods if valid_size_periods is not None else config.valid_size_periods

    if test_percentage is not None:
        resolved_test_size = max(1, ceil(num_periods * test_percentage))
    if valid_percentage is not None:
        resolved_valid_size = max(1, ceil(num_periods * valid_percentage))

    max_reserved_periods = max(num_periods - 1, 0)
    if resolved_test_size + resolved_valid_size > max_reserved_periods:
        overflow = resolved_test_size + resolved_valid_size - max_reserved_periods

        if valid_percentage is not None and resolved_valid_size > 1:
            reduction = min(overflow, resolved_valid_size - 1)
            resolved_valid_size -= reduction
            overflow -= reduction

        if overflow > 0 and test_percentage is not None and resolved_test_size > 1:
            reduction = min(overflow, resolved_test_size - 1)
            resolved_test_size -= reduction
            overflow -= reduction

        if overflow > 0:
            raise ValueError(
                "The requested train/valid/test split leaves no periods for train. "
                f"num_periods={num_periods}, test_size={resolved_test_size}, "
                f"valid_size={resolved_valid_size}."
            )

    return resolved_test_size, resolved_valid_size


def resolve_temporal_split_periods(
    modeling_table: pd.DataFrame,
    config: ForecastConfig = ForecastConfig(),
    test_size_periods: Optional[int] = None,
    valid_size_periods: Optional[int] = None,
) -> Tuple[set, set, set]:
    active_target_col, active_target_scaled_col = get_active_target_columns(config)
    data = modeling_table.dropna(subset=[active_target_col, active_target_scaled_col]).copy()
    unique_periods = sorted(data[config.period_col].unique())
    test_size_periods, valid_size_periods = resolve_split_sizes(
        num_periods=len(unique_periods),
        config=config,
        test_size_periods=test_size_periods,
        valid_size_periods=valid_size_periods,
    )
    min_required_periods = test_size_periods + valid_size_periods + 1

    period_label = "months" if config.aggregation_level == "month" else "weeks"

    if len(unique_periods) < min_required_periods:
        raise ValueError(
            f"Not enough {period_label} for the requested split. "
            f"Available {period_label}: {len(unique_periods)}. "
            f"Required at least: {min_required_periods}."
        )

    test_periods = set(unique_periods[-test_size_periods:])
    valid_periods = set(
        unique_periods[-(test_size_periods + valid_size_periods):-test_size_periods]
    )
    train_periods = set(unique_periods) - test_periods - valid_periods

    return train_periods, valid_periods, test_periods


def temporal_train_valid_test_split(
    modeling_table: pd.DataFrame,
    config: ForecastConfig = ForecastConfig(),
    test_size_periods: Optional[int] = None,
    valid_size_periods: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    active_target_col, active_target_scaled_col = get_active_target_columns(config)
    data = modeling_table.dropna(subset=[active_target_col, active_target_scaled_col]).copy()
    train_periods, valid_periods, test_periods = resolve_temporal_split_periods(
        modeling_table=modeling_table,
        config=config,
        test_size_periods=test_size_periods,
        valid_size_periods=valid_size_periods,
    )

    train_df = data[~data[config.period_col].isin(test_periods | valid_periods)].copy()
    valid_df = data[data[config.period_col].isin(valid_periods)].copy()
    test_df = data[data[config.period_col].isin(test_periods)].copy()

    train_df = add_time_decay_weights(
        train_df,
        config,
        reference_week=train_df[config.period_col].max(),
    )

    return train_df, valid_df, test_df


def cast_categorical_features(
    X: pd.DataFrame,
    cat_features: Sequence[str],
) -> pd.DataFrame:
    X = X.copy()

    for col in cat_features:
        if col in X.columns:
            X[col] = X[col].astype("object")
            X[col] = X[col].where(X[col].notna(), "__MISSING__")
            X[col] = X[col].astype(str)

    return X


def train_catboost_model(
    train_df: pd.DataFrame,
    valid_df: Optional[pd.DataFrame],
    feature_cols: Sequence[str],
    cat_features: Sequence[str],
    config: ForecastConfig = ForecastConfig(),
    params: Optional[Dict[str, Any]] = None,
):
    from catboost import CatBoostRegressor, Pool

    default_params = {
        "loss_function": "MAE", #"RMSE",
        "eval_metric": "MAE", #"RMSE",
        "iterations": 2000,
        "learning_rate": 0.03,
        "depth": 8,
        "l2_leaf_reg": 5.0,
        "random_seed": config.random_seed,
        "verbose": 200,
        "allow_writing_files": False,
        "early_stopping_rounds": 100,
    }

    moderate_params = {
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "iterations": 3000,
        "learning_rate": 0.04,
        "depth": 10,
        "l2_leaf_reg": 3.0,
        "random_seed": config.random_seed,
        "verbose": 200,
        "allow_writing_files": False,
        "early_stopping_rounds": 150,
    }

    aggressive_params = {
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "iterations": 4000,
        "learning_rate": 0.05,
        "depth": 10,
        "l2_leaf_reg": 2.0,
        "random_seed": config.random_seed,
        "verbose": 200,
        "allow_writing_files": False,
        "early_stopping_rounds": 150,
    }

    params = moderate_params

    if params is not None:
        params.update(params)

    X_train = cast_categorical_features(train_df[list(feature_cols)], cat_features)
    y_train = train_df[config.active_target_scaled_col]

    train_pool = Pool(
        X_train,
        y_train,
        cat_features=list(cat_features),
        weight=train_df.get("sample_weight"),
    )

    valid_pool = None

    if valid_df is not None and len(valid_df) > 0:
        X_valid = cast_categorical_features(valid_df[list(feature_cols)], cat_features)
        y_valid = valid_df[config.active_target_scaled_col]
        valid_pool = Pool(
            X_valid,
            y_valid,
            cat_features=list(cat_features),
        )

    model = CatBoostRegressor(**params)

    if valid_pool is not None:
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    else:
        model.fit(train_pool)

    return model


def predict_with_catboost(
    model,
    data: pd.DataFrame,
    feature_cols: Sequence[str],
    cat_features: Sequence[str],
    config: ForecastConfig = ForecastConfig(),
    clip_negative: bool = True,
) -> pd.DataFrame:
    X = cast_categorical_features(data[list(feature_cols)], cat_features)
    pred_scaled = model.predict(X)
    pred = pred_scaled * data["scale_factor"].to_numpy()

    if clip_negative:
        pred = np.clip(pred, 0, None)

    output = build_prediction_base_output(data, config)

    output["prediction_catboost"] = pred
    return output


def build_prediction_base_output(
    data: pd.DataFrame,
    config: ForecastConfig = ForecastConfig(),
) -> pd.DataFrame:
    output = data[config.group_cols + [config.period_col, "scale_factor"]].copy()
    output["horizon"] = config.horizon_periods

    if config.aggregation_level == "month":
        output["forecast_period_start"] = output[config.period_col] + pd.offsets.MonthBegin(
            config.horizon_periods
        )
    else:
        output["forecast_period_start"] = (
            output[config.period_col] + pd.to_timedelta(7 * config.horizon_periods, unit="D")
        )

    if config.active_target_col in data.columns:
        output[config.target_col] = data[config.active_target_col].to_numpy()

    return output


def get_hwes_seasonal_periods(config: ForecastConfig = ForecastConfig()) -> int:
    return 12 if config.aggregation_level == "month" else 52


def compute_forecast_steps(
    train_end_period: pd.Timestamp,
    forecast_periods: pd.Series,
    config: ForecastConfig = ForecastConfig(),
) -> pd.Series:
    if config.aggregation_level == "month":
        train_period = pd.Period(train_end_period, freq="M")
        forecast_index = pd.to_datetime(forecast_periods).dt.to_period("M")
        return forecast_index.map(lambda period: period.ordinal - train_period.ordinal).astype(int)

    train_period = pd.Period(train_end_period, freq="W")
    forecast_index = pd.to_datetime(forecast_periods).dt.to_period("W")
    return forecast_index.map(lambda period: period.ordinal - train_period.ordinal).astype(int)


def regression_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> Dict[str, float]:
    y_true = pd.Series(y_true).astype(float)
    y_pred = pd.Series(y_pred).astype(float)
    error = y_pred - y_true
    abs_error = np.abs(error)

    return {
        "mae": float(abs_error.mean()),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "wape": float(abs_error.sum() / (np.abs(y_true).sum() + EPS)),
        "bias": float(error.sum() / (np.abs(y_true).sum() + EPS)),
    }


def evaluate_baselines(
    test_df: pd.DataFrame,
    config: ForecastConfig = ForecastConfig(),
) -> pd.DataFrame:
    rows = []
    baseline_period_label = "month" if config.aggregation_level == "month" else "week"
    rolling_baseline_window = config.baseline_moving_average_months if config.aggregation_level == "month" else config.baseline_moving_average_weeks
    baseline_defs = {
        f"baseline_previous_{baseline_period_label}": "lag_0",
        f"baseline_mean_last_{rolling_baseline_window}_{baseline_period_label}s": (
            f"rolling_mean_{rolling_baseline_window}"
        ),
    }

    for model_name, pred_col in baseline_defs.items():
        eval_df = test_df.dropna(subset=[config.active_target_col, pred_col]).copy()

        if len(eval_df) == 0:
            continue

        metrics = regression_metrics(
            eval_df[config.active_target_col],
            eval_df[pred_col].clip(lower=0),
        )

        rows.append({
            "model": model_name,
            **metrics,
        })

    return pd.DataFrame(rows)


def build_baseline_predictions(
    test_df: pd.DataFrame,
    config: ForecastConfig = ForecastConfig(),
    clip_negative: bool = True,
) -> pd.DataFrame:
    baseline_period_label = "month" if config.aggregation_level == "month" else "week"
    rolling_baseline_window = config.baseline_moving_average_months if config.aggregation_level == "month" else config.baseline_moving_average_weeks
    baseline_defs = {
        f"baseline_previous_{baseline_period_label}": "lag_0",
        f"baseline_mean_last_{rolling_baseline_window}_{baseline_period_label}s": (
            f"rolling_mean_{rolling_baseline_window}"
        ),
    }

    base_output = build_prediction_base_output(test_df, config)

    rows = []
    for predictor_name, pred_col in baseline_defs.items():
        if pred_col not in test_df.columns:
            continue

        predictor_output = base_output.copy()
        prediction = test_df[pred_col].to_numpy()
        if clip_negative:
            prediction = np.clip(prediction, 0, None)

        predictor_output["predictor"] = predictor_name
        predictor_output["prediction"] = prediction
        rows.append(predictor_output)

    if not rows:
        return pd.DataFrame(columns=list(base_output.columns) + ["predictor", "prediction"])

    return pd.concat(rows, ignore_index=True)


def evaluate_hwes(
    history_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ForecastConfig = ForecastConfig(),
    clip_negative: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    value_col = config.value_col if config.value_col in history_df.columns else "lag_0"
    base_output = build_prediction_base_output(test_df, config)
    base_output["prediction_hwes"] = np.nan
    seasonal_periods = get_hwes_seasonal_periods(config)

    for group_values, test_group in test_df.groupby(config.group_cols, sort=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        history_mask = np.logical_and.reduce([
            history_df[group_col].astype(str) == str(group_value)
            for group_col, group_value in zip(config.group_cols, group_values)
        ])
        history_group = history_df.loc[history_mask].copy().sort_values(config.period_col)
        if test_group.empty:
            continue

        if history_group.empty or value_col not in history_group.columns:
            continue

        test_group = test_group.sort_values(config.period_col).copy()

        for origin_row in test_group.itertuples(index=True):
            origin_period = pd.to_datetime(getattr(origin_row, config.period_col))
            history_until_origin = history_group[
                pd.to_datetime(history_group[config.period_col]) <= origin_period
            ].copy()
            y_train = (
                history_until_origin.sort_values(config.period_col)[value_col]
                .dropna()
                .astype(float)
                .reset_index(drop=True)
            )
            if y_train.empty:
                continue

            forecast_steps = compute_forecast_steps(
                train_end_period=origin_period,
                forecast_periods=pd.Series([base_output.at[origin_row.Index, "forecast_period_start"]]),
                config=config,
            )
            if forecast_steps.empty or int(forecast_steps.iloc[0]) <= 0:
                continue

            try:
                fitted_model = ExponentialSmoothing(
                    y_train,
                    trend="add",
                    seasonal="add",
                    seasonal_periods=seasonal_periods,
                    initialization_method="estimated",
                ).fit(optimized=True)
            except ValueError:
                continue

            forecast_values = pd.Series(fitted_model.forecast(steps=int(forecast_steps.iloc[0])))
            prediction = forecast_values.iloc[int(forecast_steps.iloc[0]) - 1]
            if clip_negative:
                prediction = max(float(prediction), 0.0)

            base_output.at[origin_row.Index, "prediction_hwes"] = prediction

    pred_df = base_output.copy()
    eval_df = pred_df.dropna(subset=[config.target_col, "prediction_hwes"]).copy()

    if eval_df.empty:
        metrics_df = pd.DataFrame(columns=["model", "horizon", "mae", "rmse", "wape", "bias"])
    else:
        metrics = regression_metrics(
            eval_df[config.target_col],
            eval_df["prediction_hwes"],
        )
        metrics_df = pd.DataFrame([{
            "model": "hwes",
            "horizon": config.horizon_periods,
            **metrics,
        }])

    return metrics_df, pred_df


def format_hwes_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    formatted = pred_df.copy()
    formatted["predictor"] = "hwes"
    formatted["prediction"] = formatted["prediction_hwes"]
    return formatted.drop(columns=["prediction_hwes"], errors="ignore")


def build_chronos_pipeline(config: ForecastConfig = ForecastConfig()):
    try:
        import torch
        from chronos import ChronosPipeline
    except ImportError as exc:
        raise ImportError(
            "Chronos dependencies are not installed. Install them with: pip install git+https://github.com/amazon-science/chronos-forecasting.git torch"
        ) from exc

    has_cuda = torch.cuda.is_available()
    should_use_cuda = config.use_gpu_chronos and has_cuda
    device_map = "cuda" if should_use_cuda else "cpu"
    print(f"Using device: {device_map}")

    print(torch.__version__)
    print('cuda:', torch.cuda.is_available())
    print('cuda version:', torch.version.cuda)
    print('device count:', torch.cuda.device_count())
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')

    torch_dtype = torch.bfloat16 if should_use_cuda else torch.float32
    return ChronosPipeline.from_pretrained(
        config.chronos_model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )


def evaluate_chronos(
    history_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ForecastConfig = ForecastConfig(),
    clip_negative: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    import torch

    value_col = config.value_col if config.value_col in history_df.columns else "lag_0"
    base_output = build_prediction_base_output(test_df, config)
    base_output["prediction_chronos"] = np.nan
    pipeline = build_chronos_pipeline(config)
    prediction_batches: Dict[int, list] = {}

    for group_values, test_group in test_df.groupby(config.group_cols, sort=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        history_mask = np.logical_and.reduce([
            history_df[group_col].astype(str) == str(group_value)
            for group_col, group_value in zip(config.group_cols, group_values)
        ])
        history_group = history_df.loc[history_mask].copy().sort_values(config.period_col)
        if test_group.empty:
            continue

        if history_group.empty or value_col not in history_group.columns:
            continue

        test_group = test_group.sort_values(config.period_col).copy()

        for origin_row in test_group.itertuples(index=True):
            origin_period = pd.to_datetime(getattr(origin_row, config.period_col))
            history_until_origin = history_group[
                pd.to_datetime(history_group[config.period_col]) <= origin_period
            ].copy()
            y_train = (
                history_until_origin.sort_values(config.period_col)[value_col]
                .dropna()
                .astype(float)
                .reset_index(drop=True)
            )
            if y_train.empty:
                continue

            forecast_steps = compute_forecast_steps(
                train_end_period=origin_period,
                forecast_periods=pd.Series([base_output.at[origin_row.Index, "forecast_period_start"]]),
                config=config,
            )
            if forecast_steps.empty or int(forecast_steps.iloc[0]) <= 0:
                continue

            prediction_length = int(forecast_steps.iloc[0])
            context = torch.tensor(y_train.to_numpy(dtype=np.float32))
            prediction_batches.setdefault(prediction_length, []).append((origin_row.Index, context))

    with torch.inference_mode():
        for prediction_length, batch_items in prediction_batches.items():
            contexts = [context for _, context in batch_items]
            forecast = pipeline.predict(
                contexts,
                prediction_length=prediction_length,
                num_samples=config.chronos_num_samples,
            )
            forecast_values = forecast.detach().cpu().numpy()
            predictions = np.median(forecast_values[:, :, prediction_length - 1], axis=1)
            if clip_negative:
                predictions = np.clip(predictions, 0, None)

            for (row_index, _), prediction in zip(batch_items, predictions):
                base_output.at[row_index, "prediction_chronos"] = float(prediction)

    pred_df = base_output.copy()
    eval_df = pred_df.dropna(subset=[config.target_col, "prediction_chronos"]).copy()

    if eval_df.empty:
        metrics_df = pd.DataFrame(columns=["model", "horizon", "mae", "rmse", "wape", "bias"])
    else:
        metrics = regression_metrics(
            eval_df[config.target_col],
            eval_df["prediction_chronos"],
        )
        metrics_df = pd.DataFrame([{
            "model": "chronos",
            "horizon": config.horizon_periods,
            **metrics,
        }])

    return metrics_df, pred_df


def format_chronos_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    formatted = pred_df.copy()
    formatted["predictor"] = "chronos"
    formatted["prediction"] = formatted["prediction_chronos"]
    return formatted.drop(columns=["prediction_chronos"], errors="ignore")


def evaluate_catboost(
    model,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
    cat_features: Sequence[str],
    config: ForecastConfig = ForecastConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pred_df = predict_with_catboost(
        model=model,
        data=test_df,
        feature_cols=feature_cols,
        cat_features=cat_features,
        config=config,
    )

    eval_df = pred_df.dropna(subset=[config.target_col, "prediction_catboost"]).copy()
    metrics = regression_metrics(
        eval_df[config.target_col],
        eval_df["prediction_catboost"],
    )

    metrics_df = pd.DataFrame([{
        "model": "catboost",
        "horizon": config.horizon_periods,
        **metrics,
    }])

    return metrics_df, pred_df


def format_catboost_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    formatted = pred_df.copy()
    formatted["predictor"] = "catboost"
    formatted["prediction"] = formatted["prediction_catboost"]
    return formatted.drop(columns=["prediction_catboost"], errors="ignore")


def run_full_training_pipeline(
    df: pd.DataFrame,
    config: ForecastConfig = ForecastConfig(),
    test_size_periods: Optional[int] = None,
    valid_size_periods: Optional[int] = None,
    catboost_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    modeling_table = build_modeling_table(df, config)

    train_df, valid_df, test_df = temporal_train_valid_test_split(
        modeling_table,
        config=config,
        test_size_periods=test_size_periods,
        valid_size_periods=valid_size_periods,
    )

    feature_cols, cat_features = build_feature_columns(
        modeling_table,
        config=config,
        use_scaled_demand_features=True,
    )

    model = train_catboost_model(
        train_df=train_df,
        valid_df=valid_df,
        feature_cols=feature_cols,
        cat_features=cat_features,
        config=config,
        params=catboost_params,
    )

    baseline_metrics = evaluate_baselines(test_df, config)
    catboost_metrics, catboost_predictions = evaluate_catboost(
        model=model,
        test_df=test_df,
        feature_cols=feature_cols,
        cat_features=cat_features,
        config=config,
    )
    hwes_metrics, hwes_predictions = evaluate_hwes(
        history_df=modeling_table,
        test_df=test_df,
        config=config,
    )

    metrics = (
        pd.concat([baseline_metrics, catboost_metrics, hwes_metrics], ignore_index=True)
        .sort_values("wape")
        .reset_index(drop=True)
    )

    return {
        "model": model,
        "config": config,
        "modeling_table": modeling_table,
        "train_df": train_df,
        "valid_df": valid_df,
        "test_df": test_df,
        "feature_cols": feature_cols,
        "cat_features": cat_features,
        "metrics": metrics,
        "catboost_predictions": catboost_predictions,
        "hwes_predictions": hwes_predictions,
    }


def predict_next_period(
    df: pd.DataFrame,
    model,
    feature_cols: Sequence[str],
    cat_features: Sequence[str],
    config: ForecastConfig = ForecastConfig(),
    target_group: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    modeling_table = build_modeling_table(df, config)

    latest_rows = (
        modeling_table.sort_values(config.group_cols + [config.period_col])
        .groupby(config.group_cols, as_index=False)
        .tail(1)
        .copy()
    )

    if target_group is not None:
        for group_col in config.group_cols:
            if group_col not in target_group:
                raise ValueError(f"missing group key: {group_col}")

            latest_rows = latest_rows[
                latest_rows[group_col].astype(str) == str(target_group[group_col])
            ].copy()

        if len(latest_rows) == 0:
            raise ValueError(f"group not found: {target_group}")

    predictions = predict_with_catboost(
        model=model,
        data=latest_rows,
        feature_cols=feature_cols,
        cat_features=cat_features,
        config=config,
    )

    predictions = predictions.rename(columns={
        config.period_col: "origin_period_start",
        "prediction_catboost": "forecast_value",
    })

    return predictions[
        [
            *config.group_cols,
            "origin_period_start",
            "forecast_period_start",
            "forecast_value",
            "scale_factor",
        ]
    ].reset_index(drop=True)
