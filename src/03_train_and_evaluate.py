from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from utils.utils_feature_engineering import (
    ForecastConfig,
    add_time_decay_weights,
    build_feature_columns,
)
from utils.utils_model_training import (
    build_baseline_predictions,
    evaluate_baselines,
    evaluate_catboost,
    evaluate_chronos,
    evaluate_hwes,
    predict_with_catboost,
    format_catboost_predictions,
    format_chronos_predictions,
    format_hwes_predictions,
    resolve_temporal_split_periods,
    train_catboost_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


def load_processed_for_training_dataset(
    input_path: Path | None = None,
    config: ForecastConfig = ForecastConfig(),
) -> pd.DataFrame:
    if input_path is None:
        raise ValueError("input_path is required to load processed_for_training datasets")
    resolved_input_path = input_path
    modeling_table = pd.read_parquet(resolved_input_path)
    modeling_table[config.period_col] = pd.to_datetime(modeling_table[config.period_col])
    return modeling_table.sort_values(config.group_cols + [config.period_col]).reset_index(drop=True)


def build_horizon_input_path(config: ForecastConfig, horizon: int) -> Path:
    return config.processed_for_training_output_path(horizon)


def build_output_paths(output_dir: Path, config: ForecastConfig) -> Dict[str, Path]:
    suffix = f"{config.aggregation_level}_h{config.max_horizon_periods}"
    return {
        "models_dir": output_dir / f"models_{suffix}",
        "metrics": output_dir / config.build_dataset_file_name("metrics", suffix=suffix, extension=".xlsx"),
        "predictions": output_dir / config.build_dataset_file_name("test_predictions", suffix=suffix, extension=".csv"),
        "predictions_consolidated": output_dir / config.build_dataset_file_name("test_predictions_multi_horizon", suffix=suffix, extension=".csv"),
        "train": output_dir / config.build_dataset_file_name("train_multi_horizon", suffix=suffix, extension=".csv"),
        "valid": output_dir / config.build_dataset_file_name("valid_multi_horizon", suffix=suffix, extension=".csv"),
        "test": output_dir / config.build_dataset_file_name("test_multi_horizon", suffix=suffix, extension=".csv"),
        "feature_metadata": output_dir / config.build_dataset_file_name("feature_metadata", suffix=suffix, extension=".json"),
        "feature_importance": output_dir / config.build_dataset_file_name("feature_importance", suffix=suffix, extension=".csv"),
    }


def build_catboost_prediction_output_path(
    output_dir: Path,
    config: ForecastConfig,
) -> Path:
    return output_dir / (
        f"{config.dataset_slug}_catboost_predictions_{config.aggregation_level}_h{config.max_horizon_periods}.csv"
    )


def save_feature_metadata(
    features_by_horizon: Dict[int, list[str]],
    cat_features_by_horizon: Dict[int, list[str]],
    forecast_horizons: list[int],
    output_path: Path,
) -> None:
    payload: Dict[str, Any] = {
        "features_by_horizon": features_by_horizon,
        "cat_features_by_horizon": cat_features_by_horizon,
        "forecast_horizons": forecast_horizons,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_prediction_export_feature_columns(
    frame: pd.DataFrame,
    feature_cols: list[str],
) -> list[str]:
    export_feature_cols: list[str] = []

    for col in feature_cols:
        candidate_cols = [col]
        if col.endswith("_scaled"):
            raw_col = col[: -len("_scaled")]
            candidate_cols.append(raw_col)
        else:
            scaled_col = f"{col}_scaled"
            candidate_cols.append(scaled_col)

        for candidate in candidate_cols:
            if candidate in frame.columns and candidate not in export_feature_cols:
                export_feature_cols.append(candidate)

    return export_feature_cols


def build_catboost_prediction_export(
    prediction_frame: pd.DataFrame,
    feature_cols: list[str],
    config: ForecastConfig,
    split_name: str,
) -> pd.DataFrame:
    export_feature_cols = build_prediction_export_feature_columns(
        prediction_frame,
        feature_cols,
    )
    available_feature_cols = [
        col for col in export_feature_cols if col in prediction_frame.columns
    ]
    base_cols = config.group_cols + [config.period_col]
    export_cols = list(dict.fromkeys(base_cols + available_feature_cols))

    output = prediction_frame.loc[:, export_cols].copy()
    output = output.rename(columns={config.period_col: config.date_col})
    output["train_val_test"] = split_name
    output["target_original"] = prediction_frame[config.target_col].to_numpy()
    output["target_scaled"] = (
        prediction_frame[config.target_col].to_numpy()
        / prediction_frame["scale_factor"].to_numpy()
    )
    output["prediction_catboost_scaled"] = (
        prediction_frame["prediction_catboost"].to_numpy()
        / prediction_frame["scale_factor"].to_numpy()
    )
    output["prediction_catboost"] = prediction_frame["prediction_catboost"].to_numpy()
    output["horizon"] = config.horizon_periods

    ordered_cols = list(
        dict.fromkeys(
            config.group_cols
            + [config.date_col, "train_val_test"]
            + available_feature_cols
            + [
                "target_original",
                "target_scaled",
                "prediction_catboost_scaled",
                "prediction_catboost",
                "horizon",
            ]
        )
    )
    return output.loc[:, ordered_cols]


def enrich_prediction_frame_with_source_data(
    prediction_frame: pd.DataFrame,
    source_frame: pd.DataFrame,
    config: ForecastConfig,
) -> pd.DataFrame:
    merge_keys = config.group_cols + [config.period_col, "scale_factor"]
    if config.target_col in prediction_frame.columns and config.target_col in source_frame.columns:
        merge_keys.append(config.target_col)

    source_columns = [
        col for col in source_frame.columns
        if col not in prediction_frame.columns or col in merge_keys
    ]
    source_subset = source_frame.loc[:, source_columns].copy()

    return prediction_frame.merge(
        source_subset,
        on=merge_keys,
        how="left",
    )


def build_consolidated_predictions_table(
    test_frames_by_horizon: dict[int, pd.DataFrame],
    predictions_by_horizon: dict[int, pd.DataFrame],
    feature_cols: list[str],
    config: ForecastConfig,
) -> pd.DataFrame:
    base_columns = list(dict.fromkeys(config.group_cols + [config.period_col] + feature_cols))
    base_frame = test_frames_by_horizon[min(test_frames_by_horizon)].copy()
    consolidated = (
        base_frame.loc[:, [col for col in base_columns if col in base_frame.columns]]
        .drop_duplicates(subset=config.group_cols + [config.period_col])
        .sort_values(config.group_cols + [config.period_col])
        .reset_index(drop=True)
    )

    for horizon in config.forecast_horizons:
        target_col = config.target_col_for_horizon(horizon)
        horizon_pred = predictions_by_horizon[horizon].copy()
        horizon_pred = horizon_pred.rename(
            columns={
                config.target_col: f"target_real_t_plus_{horizon}",
                "prediction_catboost": f"target_pred_t_plus_{horizon}",
                "forecast_period_start": f"forecast_period_start_t_plus_{horizon}",
            }
        )

        merge_columns = config.group_cols + [config.period_col]
        keep_columns = merge_columns + [
            f"forecast_period_start_t_plus_{horizon}",
            f"target_real_t_plus_{horizon}",
            f"target_pred_t_plus_{horizon}",
        ]
        consolidated = consolidated.merge(
            horizon_pred[keep_columns],
            on=merge_columns,
            how="left",
        )

    return consolidated


def train_and_evaluate_pipeline(
    input_path: Path | None = None,
    output_dir: Path = RESULTS_DIR,
    config: ForecastConfig = ForecastConfig(),
    test_size_periods: int | None = None,
    valid_size_periods: int | None = None,
    catboost_params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    output_paths = build_output_paths(output_dir, config)

    max_horizon_config = config.with_horizon(max(config.forecast_horizons))
    max_horizon_input_path = input_path or build_horizon_input_path(config, max(config.forecast_horizons))
    max_horizon_modeling_table = load_processed_for_training_dataset(max_horizon_input_path, max_horizon_config)
    _, valid_periods, test_periods = resolve_temporal_split_periods(
        modeling_table=max_horizon_modeling_table,
        config=max_horizon_config,
        test_size_periods=test_size_periods,
        valid_size_periods=valid_size_periods,
    )

    all_metrics: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    all_train_frames: list[pd.DataFrame] = []
    all_valid_frames: list[pd.DataFrame] = []
    all_test_frames: list[pd.DataFrame] = []
    feature_importance_frames: list[pd.DataFrame] = []
    models_by_horizon: dict[int, Any] = {}
    predictions_by_horizon: dict[int, pd.DataFrame] = {}
    catboost_partition_predictions_by_horizon: dict[int, pd.DataFrame] = {}
    test_frames_by_horizon: dict[int, pd.DataFrame] = {}
    features_by_horizon: Dict[int, list[str]] = {}
    cat_features_by_horizon: Dict[int, list[str]] = {}

    for horizon in config.forecast_horizons:
        horizon_config = config.with_horizon(horizon)
        horizon_input_path = build_horizon_input_path(config, horizon)
        horizon_modeling_table = load_processed_for_training_dataset(horizon_input_path, horizon_config)
        feature_cols, cat_features = build_feature_columns(
            horizon_modeling_table,
            config=horizon_config,
            use_scaled_demand_features=True,
        )
        features_by_horizon[horizon] = feature_cols
        cat_features_by_horizon[horizon] = cat_features
        active_target_col = horizon_config.active_target_col
        active_target_scaled_col = horizon_config.active_target_scaled_col
        eligible_table = horizon_modeling_table.dropna(subset=[active_target_col, active_target_scaled_col]).copy()

        train_df = eligible_table[
            ~eligible_table[horizon_config.period_col].isin(valid_periods | test_periods)
        ].copy()
        valid_df = eligible_table[
            eligible_table[horizon_config.period_col].isin(valid_periods)
        ].copy()
        test_df = eligible_table[
            eligible_table[horizon_config.period_col].isin(test_periods)
        ].copy()

        train_df = add_time_decay_weights(
            train_df,
            horizon_config,
            reference_week=train_df[horizon_config.period_col].max(),
        )

        model = train_catboost_model(
            train_df=train_df,
            valid_df=valid_df,
            feature_cols=feature_cols,
            cat_features=cat_features,
            config=horizon_config,
            params=catboost_params,
        )

        baseline_metrics = evaluate_baselines(test_df, horizon_config)
        if not baseline_metrics.empty:
            baseline_metrics["horizon"] = horizon

        catboost_metrics, catboost_predictions = evaluate_catboost(
            model=model,
            test_df=test_df,
            feature_cols=feature_cols,
            cat_features=cat_features,
            config=horizon_config,
        )
        train_catboost_predictions = predict_with_catboost(
            model=model,
            data=train_df,
            feature_cols=feature_cols,
            cat_features=cat_features,
            config=horizon_config,
        )
        valid_catboost_predictions = predict_with_catboost(
            model=model,
            data=valid_df,
            feature_cols=feature_cols,
            cat_features=cat_features,
            config=horizon_config,
        )
        if config.use_hwes:
            hwes_metrics, hwes_predictions = evaluate_hwes(
                history_df=eligible_table,
                test_df=test_df,
                config=horizon_config,
            )
            formatted_hwes_predictions = format_hwes_predictions(hwes_predictions)
        else:
            hwes_metrics = pd.DataFrame()
            formatted_hwes_predictions = pd.DataFrame()
        if config.use_chronos:
            chronos_metrics, chronos_predictions = evaluate_chronos(
                history_df=eligible_table,
                test_df=test_df,
                config=horizon_config,
            )
            formatted_chronos_predictions = format_chronos_predictions(chronos_predictions)
        else:
            chronos_metrics = pd.DataFrame()
            formatted_chronos_predictions = pd.DataFrame()
        baseline_predictions = build_baseline_predictions(test_df, horizon_config)
        formatted_catboost_predictions = format_catboost_predictions(catboost_predictions)
        horizon_predictions = pd.concat(
            [
                formatted_catboost_predictions,
                formatted_hwes_predictions,
                formatted_chronos_predictions,
                baseline_predictions,
            ],
            ignore_index=True,
        )

        horizon_metrics = (
            pd.concat([baseline_metrics, catboost_metrics, hwes_metrics, chronos_metrics], ignore_index=True)
            .sort_values("wape")
            .reset_index(drop=True)
        )

        feature_importance = pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": model.get_feature_importance(),
                "horizon": horizon,
            }
        ).sort_values("importance", ascending=False).reset_index(drop=True)

        train_df["horizon"] = horizon
        valid_df["horizon"] = horizon
        test_df["horizon"] = horizon
        catboost_predictions["horizon"] = horizon
        horizon_predictions["horizon"] = horizon

        all_metrics.append(horizon_metrics)
        all_predictions.append(horizon_predictions)
        all_train_frames.append(train_df)
        all_valid_frames.append(valid_df)
        all_test_frames.append(test_df)
        feature_importance_frames.append(feature_importance)
        models_by_horizon[horizon] = model
        predictions_by_horizon[horizon] = catboost_predictions
        catboost_partition_predictions_by_horizon[horizon] = pd.concat(
            [
                build_catboost_prediction_export(
                    prediction_frame=enrich_prediction_frame_with_source_data(
                        prediction_frame=train_catboost_predictions,
                        source_frame=train_df,
                        config=horizon_config,
                    ),
                    feature_cols=feature_cols,
                    config=horizon_config,
                    split_name="train",
                ),
                build_catboost_prediction_export(
                    prediction_frame=enrich_prediction_frame_with_source_data(
                        prediction_frame=valid_catboost_predictions,
                        source_frame=valid_df,
                        config=horizon_config,
                    ),
                    feature_cols=feature_cols,
                    config=horizon_config,
                    split_name="val",
                ),
                build_catboost_prediction_export(
                    prediction_frame=enrich_prediction_frame_with_source_data(
                        prediction_frame=catboost_predictions,
                        source_frame=test_df,
                        config=horizon_config,
                    ),
                    feature_cols=feature_cols,
                    config=horizon_config,
                    split_name="test",
                ),
            ],
            ignore_index=True,
        ).sort_values(
            config.group_cols + [config.date_col, "train_val_test"]
        ).reset_index(drop=True)
        test_frames_by_horizon[horizon] = test_df.copy()

    metrics = pd.concat(all_metrics, ignore_index=True).sort_values(["horizon", "wape"]).reset_index(drop=True)
    catboost_predictions = pd.concat(all_predictions, ignore_index=True).sort_values(
        ["horizon", "predictor", *config.group_cols, config.period_col]
    ).reset_index(drop=True)
    train_df = pd.concat(all_train_frames, ignore_index=True)
    valid_df = pd.concat(all_valid_frames, ignore_index=True)
    test_df = pd.concat(all_test_frames, ignore_index=True)
    feature_importance = pd.concat(feature_importance_frames, ignore_index=True)
    all_feature_cols = list(
        dict.fromkeys([col for cols in features_by_horizon.values() for col in cols])
    )
    consolidated_predictions = build_consolidated_predictions_table(
        test_frames_by_horizon=test_frames_by_horizon,
        predictions_by_horizon=predictions_by_horizon,
        feature_cols=all_feature_cols,
        config=config,
    )
    catboost_partition_predictions = pd.concat(
        [catboost_partition_predictions_by_horizon[h] for h in config.forecast_horizons],
        ignore_index=True,
    ).sort_values(
        [*config.group_cols, config.date_col, "horizon", "train_val_test"]
    ).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths["models_dir"].mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_paths["train"], sep=";", index=False, encoding="utf-8-sig")
    valid_df.to_csv(output_paths["valid"], sep=";", index=False, encoding="utf-8-sig")
    test_df.to_csv(output_paths["test"], sep=";", index=False, encoding="utf-8-sig")
    metrics.to_excel(output_paths["metrics"], index=False)
    catboost_predictions.to_csv(output_paths["predictions"], sep=";", index=False, encoding="utf-8-sig")
    consolidated_predictions.to_csv(
        output_paths["predictions_consolidated"], sep=";", index=False, encoding="utf-8-sig"
    )
    catboost_partition_predictions.to_csv(
        build_catboost_prediction_output_path(output_dir, config),
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )
    for horizon, model in models_by_horizon.items():
        model.save_model(output_paths["models_dir"] / f"{config.dataset_slug}_catboost_model_{config.aggregation_level}_t_plus_{horizon}.cbm")
    save_feature_metadata(
        features_by_horizon=features_by_horizon,
        cat_features_by_horizon=cat_features_by_horizon,
        forecast_horizons=config.forecast_horizons,
        output_path=output_paths["feature_metadata"],
    )
    feature_importance.to_csv(output_paths["feature_importance"], sep=";", index=False, encoding="utf-8-sig")

    return {
        "models_by_horizon": models_by_horizon,
        "modeling_table": max_horizon_modeling_table,
        "train_df": train_df,
        "valid_df": valid_df,
        "test_df": test_df,
        "feature_cols": feature_cols,
        "cat_features": cat_features,
        "metrics": metrics,
        "catboost_predictions": catboost_predictions,
        "consolidated_predictions": consolidated_predictions,
        "feature_importance": feature_importance,
    }


def main() -> None:
    config = ForecastConfig()
    results = train_and_evaluate_pipeline(config=config)
    output_paths = build_output_paths(RESULTS_DIR, config)
    print(f"train saved to: {output_paths['train']}")
    print(f"valid saved to: {output_paths['valid']}")
    print(f"test saved to: {output_paths['test']}")
    print(f"metrics saved to: {output_paths['metrics']}")
    print(f"predictions saved to: {output_paths['predictions']}")
    print(f"consolidated predictions saved to: {output_paths['predictions_consolidated']}")
    print(f"models saved to: {output_paths['models_dir']}")
    print(f"feature metadata saved to: {output_paths['feature_metadata']}")
    print(f"feature importance saved to: {output_paths['feature_importance']}")
    print(results["metrics"])


if __name__ == "__main__":
    main()
