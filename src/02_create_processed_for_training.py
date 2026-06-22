from dataclasses import replace
from pathlib import Path

import pandas as pd

from utils.utils_feature_engineering import (
    ForecastConfig,
    build_feature_columns,
    build_modeling_table,
    identify_demand_feature_columns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_processed_input_path(config: ForecastConfig) -> Path:
    return config.processed_output_path


def build_output_paths(output_dir: Path, config: ForecastConfig) -> dict[str, Path]:
    suffix = f"{config.aggregation_level}_h{config.max_horizon_periods}"
    return {
        "parquet": output_dir / config.build_dataset_file_name("processed_for_training", suffix=suffix, extension=".parquet"),
        "csv": output_dir / config.build_dataset_file_name("processed_for_training", suffix=suffix, extension=".csv"),
    }


def build_horizon_output_paths(
    output_dir: Path,
    config: ForecastConfig,
    horizon: int,
) -> dict[str, Path]:
    suffix = f"{config.aggregation_level}_h{config.max_horizon_periods}_t_plus_{horizon}"
    return {
        "parquet": output_dir / config.build_dataset_file_name("processed_for_training", suffix=suffix, extension=".parquet"),
        "csv": output_dir / config.build_dataset_file_name("processed_for_training", suffix=suffix, extension=".csv"),
    }


def select_horizon_training_table(
    modeling_table: pd.DataFrame,
    config: ForecastConfig,
    horizon: int,
) -> pd.DataFrame:
    horizon_config = config.with_horizon(horizon)
    scaled_feature_cols, _ = build_feature_columns(
        modeling_table,
        config=horizon_config,
        use_scaled_demand_features=True,
    )
    raw_feature_cols, _ = build_feature_columns(
        modeling_table,
        config=horizon_config,
        use_scaled_demand_features=False,
    )
    demand_cols = identify_demand_feature_columns(modeling_table)
    raw_demand_feature_cols = [col for col in raw_feature_cols if col in demand_cols]
    scaled_demand_feature_cols = [
        f"{col}_scaled"
        for col in raw_demand_feature_cols
        if f"{col}_scaled" in modeling_table.columns
    ]
    non_demand_feature_cols = [
        col for col in raw_feature_cols if col not in demand_cols
    ]
    selected_columns = list(
        dict.fromkeys(
            config.group_cols
            + [config.period_col, config.value_col]
            + non_demand_feature_cols
            + raw_demand_feature_cols
            + scaled_demand_feature_cols
            + [
                "sample_weight",
                "scale_factor",
                "log_scale_factor",
                horizon_config.active_target_col,
                horizon_config.active_target_scaled_col,
            ]
        )
    )
    available_columns = [col for col in selected_columns if col in modeling_table.columns]
    return modeling_table.loc[:, available_columns].copy()


def load_processed_dataset(input_path: Path, config: ForecastConfig) -> pd.DataFrame:
    df = pd.read_parquet(input_path)
    df[config.period_col] = pd.to_datetime(df[config.period_col])
    return (
        df.loc[:, config.group_cols + [config.period_col, config.value_col]]
        .sort_values(config.group_cols + [config.period_col])
        .reset_index(drop=True)
    )


def create_processed_for_training_dataset(
    input_path: Path | None = None,
    output_dir: Path | None = None,
    config: ForecastConfig = ForecastConfig(),
) -> dict[int, pd.DataFrame]:
    resolved_input_path = input_path or build_processed_input_path(config)
    resolved_output_dir = output_dir or config.processed_for_training_dir
    df = load_processed_dataset(resolved_input_path, config)
    all_lags = config.all_lag_periods
    union_config = replace(
        config,
        lag_weeks=all_lags if config.aggregation_level == "week" else config.lag_weeks,
        lag_months=all_lags if config.aggregation_level == "month" else config.lag_months,
        lag_weeks_by_horizon={} if config.aggregation_level == "week" else config.lag_weeks_by_horizon,
        lag_months_by_horizon={} if config.aggregation_level == "month" else config.lag_months_by_horizon,
    )
    modeling_table = build_modeling_table(df, union_config)

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    horizon_tables: dict[int, pd.DataFrame] = {}

    for horizon in config.forecast_horizons:
        horizon_table = select_horizon_training_table(modeling_table, config, horizon)
        horizon_output_paths = build_horizon_output_paths(resolved_output_dir, config, horizon)
        horizon_table.to_parquet(horizon_output_paths["parquet"], index=False, engine="pyarrow")
        horizon_table.to_csv(horizon_output_paths["csv"], sep=";", index=False, encoding="utf-8-sig")
        horizon_tables[horizon] = horizon_table

    return horizon_tables


def main() -> None:
    config = ForecastConfig()
    horizon_tables = create_processed_for_training_dataset(config=config)
    for horizon in config.forecast_horizons:
        horizon_output_paths = build_horizon_output_paths(
            config.processed_for_training_dir,
            config,
            horizon,
        )
        print(f"processed_for_training t+{horizon} saved to: {horizon_output_paths['parquet']}")
        print(f"processed_for_training t+{horizon} csv saved to: {horizon_output_paths['csv']}")
    print(f"aggregation level: {config.aggregation_level}")
    print(f"forecast horizons: {config.forecast_horizons}")
    print(f"rows t+1: {len(horizon_tables[config.forecast_horizons[0]])}")
    print(f"columns t+1: {len(horizon_tables[config.forecast_horizons[0]].columns)}")


if __name__ == "__main__":
    main()
