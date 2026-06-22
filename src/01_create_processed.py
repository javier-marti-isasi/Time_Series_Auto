from pathlib import Path

import pandas as pd

from utils.utils_feature_engineering import ForecastConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_processed_output_path(config: ForecastConfig) -> Path:
    return config.processed_output_path


def build_processed_csv_output_path(config: ForecastConfig) -> Path:
    return config.processed_csv_output_path


def get_period_start(dates: pd.Series, aggregation_level: str) -> pd.Series:
    normalized_dates = pd.to_datetime(dates).dt.normalize()

    if aggregation_level == "week":
        return normalized_dates - pd.to_timedelta(normalized_dates.dt.weekday, unit="D")

    if aggregation_level == "month":
        return normalized_dates.dt.to_period("M").dt.to_timestamp()

    raise ValueError("aggregation_level must be one of: week, month")


def load_interim_dataset(input_path: Path, config: ForecastConfig) -> pd.DataFrame:
    data = pd.read_parquet(input_path)
    data[config.date_col] = pd.to_datetime(data[config.date_col])
    return data


def aggregate_processed_dataset(
    data: pd.DataFrame,
    config: ForecastConfig,
) -> pd.DataFrame:
    group_cols = list(config.aggregation_group_cols)
    data = data.copy()
    data[config.period_col] = get_period_start(data[config.date_col], config.aggregation_level)

    aggregated = (
        data.groupby(group_cols + [config.period_col], as_index=False)[config.value_col]
        .sum()
        .sort_values(group_cols + [config.period_col])
        .reset_index(drop=True)
    )

    return aggregated


def drop_last_period_per_group(
    data: pd.DataFrame,
    config: ForecastConfig,
) -> pd.DataFrame:
    if not config.drop_incomplete_last_period:
        return data.copy()

    group_cols = list(config.aggregation_group_cols)
    max_period_by_group = data.groupby(group_cols)[config.period_col].transform("max")
    return data.loc[data[config.period_col] != max_period_by_group].reset_index(drop=True)


def filter_groups_by_min_periods(
    data: pd.DataFrame,
    config: ForecastConfig,
) -> pd.DataFrame:
    group_cols = list(config.aggregation_group_cols)

    if config.min_periods_per_group <= 0:
        return data.copy()

    group_sizes = data.groupby(group_cols)[config.period_col].transform("size")
    return data.loc[group_sizes >= config.min_periods_per_group].reset_index(drop=True)


def create_processed_dataset(
    input_path: Path | None = None,
    output_path: Path | None = None,
    config: ForecastConfig = ForecastConfig(),
) -> pd.DataFrame:
    resolved_input_path = input_path or config.interim_input_path
    data = load_interim_dataset(resolved_input_path, config)
    processed = aggregate_processed_dataset(data, config)
    processed = drop_last_period_per_group(processed, config)
    processed = filter_groups_by_min_periods(processed, config)

    resolved_output_path = output_path or build_processed_output_path(config)
    resolved_csv_output_path = build_processed_csv_output_path(config)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_parquet(resolved_output_path, index=False, engine="pyarrow")
    processed.to_csv(resolved_csv_output_path, sep=";", index=False, encoding="utf-8-sig")

    return processed


def main() -> None:
    config = ForecastConfig()
    output_path = build_processed_output_path(config)
    csv_output_path = build_processed_csv_output_path(config)
    processed = create_processed_dataset(config=config)
    print(f"processed saved to: {output_path}")
    print(f"processed csv saved to: {csv_output_path}")
    print(f"aggregation level: {config.aggregation_level}")
    print(f"group columns: {config.aggregation_group_cols}")
    print(f"rows: {len(processed)}")
    print(f"columns: {len(processed.columns)}")


if __name__ == "__main__":
    main()
