from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd


EPS = 1e-9
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ForecastConfig:
    dataset_name: Literal["Lecta", "Competition"] = "Competition"
    aggregation_group_cols: List[str] = field(default_factory=lambda: ["product", "store"]) #field(default_factory=lambda: ["code", "warehouse"])
    date_col: str = "datetime"
    value_col: str = "value"

    aggregation_level: Literal["week", "month"] = "month"
    max_horizon_weeks: int = 6
    max_horizon_months: int = 6
    use_hwes: bool = True
    use_chronos: bool = False
    chronos_model_name: str = "amazon/chronos-t5-tiny" #"chronos-t5-tiny", "chronos-t5-mini", "chronos-t5-small", "chronos-t5-base", "chronos-t5-large"
    chronos_num_samples: int = 20
    use_gpu_chronos: bool = True

    drop_incomplete_last_period: bool = True
    min_periods_per_group_weeks: int = 104
    min_periods_per_group_months: int = 24
    test_size_periods_weeks: int = 8 #13, 12
    test_size_periods_months: int = 6
    valid_size_periods_weeks: int = 4
    valid_size_periods_months: int = 2
    per_test_size: Optional[float] = None #0.2 If defined, used percentage for splitting instead of absolute value
    per_valid_size: Optional[float] = None #0.1 If defined, used percentage for splitting instead of absolute value
    drop_incomplete_last_week: bool = True
    lag_weeks: Tuple[int, ...] = (0, 1, 2, 3, 4, 8, 13, 26, 51, 52, 53) # used for all horizons if no horizon-specific config
    lag_months: Tuple[int, ...] = (0, 1, 2, 3, 6, 12, 13) # used for all horizons if no horizon-specific config
    lag_weeks_by_horizon: Dict[int, Tuple[int, ...]] = field(
        default_factory=lambda: {
            1: (0, 1, 2, 3, 4, 12, 25, 51),
            2: (0, 1, 2, 3, 4, 11, 24, 50),
            3: (0, 1, 2, 3, 4, 10, 23, 49),
            4: (0, 1, 2, 3, 4,  9, 22, 48),
            5: (0, 1, 2, 3, 4,  8, 21, 47),
            6: (0, 1, 2, 3, 4,  7, 20, 46),
        }
    )
    lag_months_by_horizon: Dict[int, Tuple[int, ...]] = field(
        default_factory=lambda: {
            1: (0, 1, 2, 5, 11),
            2: (0, 1, 2, 4, 10),
            3: (0, 1, 2, 3, 9),
            4: (0, 1, 2, 3, 8),
            5: (0, 1, 2, 3, 7),
            6: (0, 1, 2, 3, 6),
        }
    )
    rolling_windows_weeks: Tuple[int, ...] = (3, 4, 5, 8, 13, 26, 52)
    rolling_windows_months: Tuple[int, ...] = (3, 4, 6, 12)
    ewm_spans_weeks: Tuple[int, ...] = (4, 8, 13, 26)
    ewm_spans_months: Tuple[int, ...] = (3, 6, 12)
    slope_windows_weeks: Tuple[int, ...] = (4, 8, 13)
    slope_windows_months: Tuple[int, ...] = (3, 6, 12)
    nonzero_windows_weeks: Tuple[int, ...] = (4, 8, 13, 26, 52)
    nonzero_windows_months: Tuple[int, ...] = (3, 6, 12)
    spike_rate_windows_weeks: Tuple[int, ...] = (8, 13, 26)
    spike_rate_windows_months: Tuple[int, ...] = (3, 6, 12)
    fourier_order: int = 3
    scale_window_weeks: int = 53
    scale_window_months: int = 12
    min_scale: float = 1.0
    spike_window_weeks: int = 13
    spike_window_months: int = 6
    spike_z_threshold: float = 3.0
    time_decay_half_life_weeks: int = 52
    time_decay_half_life_months: int = 12
    random_seed: int = 42
    baseline_moving_average_weeks: int = 13 #4
    baseline_moving_average_months: int = 3

    """
    training_feature_families: Tuple[str, ...] = (
        "lag",
        "rolling_mean",
    #    "rolling_median",
        "rolling_std",
    #    "rolling_min",
    #    "rolling_max",
    #    "rolling_iqr",
        "ewm_mean",
    #    "momentum_diff",
    #    "momentum_ratio",
        "rolling_slope",
        "last_year_mean_3",
    #    "last_year_median_3",
        "last_year_ratio_3",
    #    "week_of_year",
    #    "month_of_year",
    #    "quarter_of_year",
    #    "year",
        "fourier_cos",
        "fourier_sin",
    #    "spike_prior_mad",
    #    "spike_prior_median",
    #    "robust_z_spike",
    #    "is_spike",
        "spike_rate",
    #    "time_since_last_spike",
        "nonzero_rate",
    #    "is_nonzero_lag_0",
        "log_scale_factor",
    #    "scale_factor",
        "product", #"code"
        "store", #"warehouse"
    )
    """

    training_feature_families: Tuple[str, ...] = (
        "lag",
        "rolling_mean",
        "rolling_median",
        "rolling_std",
        "rolling_min",
        "rolling_max",
        "rolling_iqr",
        "ewm_mean",
        "momentum_diff",
        "momentum_ratio",
        "rolling_slope",
        "last_year_mean_3",
        "last_year_median_3",
        "last_year_ratio_3",
        "week_of_year",
        "month_of_year",
        "quarter_of_year",
        "year",
        "fourier_cos",
        "fourier_sin",
        "spike_prior_mad",
        "spike_prior_median",
        "robust_z_spike",
        "is_spike",
        "spike_rate",
        "time_since_last_spike",
        "nonzero_rate",
        "is_nonzero_lag_0",
        "log_scale_factor",
        "scale_factor",
        "product", #"code"
        "store", #"warehouse"
    )

    #TODO: REMOVE HARDCODING OF CATEGORTICAL VALUES IN training_feature_families

    _active_horizon_weeks: int = field(default=1, init=False, repr=False)
    _active_horizon_months: int = field(default=1, init=False, repr=False)
    _period_col_weeks: str = field(default="week", init=False, repr=False)
    _period_col_months: str = field(default="month", init=False, repr=False)
    _target_col: str = field(default="target", init=False, repr=False)

    @property
    def horizon_periods(self) -> int:
        if self.aggregation_level == "month":
            return self._active_horizon_months
        return self._active_horizon_weeks

    @property
    def max_horizon_periods(self) -> int:
        if self.aggregation_level == "month":
            return self.max_horizon_months
        return self.max_horizon_weeks

    @property
    def forecast_horizons(self) -> List[int]:
        return list(range(1, self.max_horizon_periods + 1))

    @property
    def min_periods_per_group(self) -> int:
        if self.aggregation_level == "month":
            return self.min_periods_per_group_months
        return self.min_periods_per_group_weeks

    @property
    def test_size_periods(self) -> int:
        if self.aggregation_level == "month":
            return self.test_size_periods_months
        return self.test_size_periods_weeks

    @property
    def valid_size_periods(self) -> int:
        if self.aggregation_level == "month":
            return self.valid_size_periods_months
        return self.valid_size_periods_weeks

    @property
    def period_col(self) -> str:
        if self.aggregation_level == "month":
            return self._period_col_months
        return self._period_col_weeks

    @property
    def target_col(self) -> str:
        return self._target_col

    @property
    def lag_periods(self) -> Tuple[int, ...]:
        by_horizon = (
            self.lag_months_by_horizon
            if self.aggregation_level == "month"
            else self.lag_weeks_by_horizon
        )
        if self.horizon_periods in by_horizon:
            return by_horizon[self.horizon_periods]
        return self.lag_months if self.aggregation_level == "month" else self.lag_weeks

    @property
    def all_lag_periods(self) -> Tuple[int, ...]:
        base = set(
            self.lag_months if self.aggregation_level == "month" else self.lag_weeks
        )
        by_horizon = (
            self.lag_months_by_horizon
            if self.aggregation_level == "month"
            else self.lag_weeks_by_horizon
        )
        for lags in by_horizon.values():
            base.update(lags)
        return tuple(sorted(base))

    @property
    def rolling_windows(self) -> Tuple[int, ...]:
        if self.aggregation_level == "month":
            return self.rolling_windows_months
        return self.rolling_windows_weeks

    @property
    def ewm_spans(self) -> Tuple[int, ...]:
        if self.aggregation_level == "month":
            return self.ewm_spans_months
        return self.ewm_spans_weeks

    @property
    def slope_windows(self) -> Tuple[int, ...]:
        if self.aggregation_level == "month":
            return self.slope_windows_months
        return self.slope_windows_weeks

    @property
    def nonzero_windows(self) -> Tuple[int, ...]:
        if self.aggregation_level == "month":
            return self.nonzero_windows_months
        return self.nonzero_windows_weeks

    @property
    def spike_rate_windows(self) -> Tuple[int, ...]:
        if self.aggregation_level == "month":
            return self.spike_rate_windows_months
        return self.spike_rate_windows_weeks

    @property
    def scale_window(self) -> int:
        if self.aggregation_level == "month":
            return self.scale_window_months
        return self.scale_window_weeks

    @property
    def spike_window(self) -> int:
        if self.aggregation_level == "month":
            return self.spike_window_months
        return self.spike_window_weeks

    @property
    def time_decay_half_life_periods(self) -> int:
        if self.aggregation_level == "month":
            return self.time_decay_half_life_months
        return self.time_decay_half_life_weeks

    @property
    def group_cols(self) -> List[str]:
        return list(self.aggregation_group_cols)

    @property
    def dataset_dir(self) -> Path:
        return PROJECT_ROOT / "data" / self.dataset_name

    @property
    def dataset_slug(self) -> str:
        return self.dataset_name.lower()

    def build_dataset_file_name(self, artifact_name: str, suffix: str = "", extension: str = "") -> str:
        normalized_suffix = f"_{suffix}" if suffix else ""
        normalized_extension = extension if not extension or extension.startswith(".") else f".{extension}"
        return f"{self.dataset_slug}_{artifact_name}{normalized_suffix}{normalized_extension}"

    def build_dataset_path(
        self,
        stage_dir: str,
        artifact_name: str,
        suffix: str = "",
        extension: str = "",
    ) -> Path:
        return self.dataset_dir / stage_dir / self.build_dataset_file_name(
            artifact_name=artifact_name,
            suffix=suffix,
            extension=extension,
        )

    @property
    def interim_input_path(self) -> Path:
        return self.build_dataset_path("interim", "interim", extension=".parquet")

    @property
    def processed_output_path(self) -> Path:
        return self.build_dataset_path(
            "processed",
            "processed",
            suffix=self.aggregation_level,
            extension=".parquet",
        )

    @property
    def processed_csv_output_path(self) -> Path:
        return self.build_dataset_path(
            "processed",
            "processed",
            suffix=self.aggregation_level,
            extension=".csv",
        )

    @property
    def processed_for_training_dir(self) -> Path:
        return self.dataset_dir / "processed_for_training"

    def processed_for_training_output_path(self, horizon: int) -> Path:
        suffix = f"{self.aggregation_level}_h{self.max_horizon_periods}_t_plus_{horizon}"
        return self.build_dataset_path(
            "processed_for_training",
            "processed_for_training",
            suffix=suffix,
            extension=".parquet",
        )

    def processed_for_training_csv_output_path(self, horizon: int) -> Path:
        suffix = f"{self.aggregation_level}_h{self.max_horizon_periods}_t_plus_{horizon}"
        return self.build_dataset_path(
            "processed_for_training",
            "processed_for_training",
            suffix=suffix,
            extension=".csv",
        )

    @property
    def results_slug(self) -> str:
        return f"{self.dataset_slug}_{self.aggregation_level}_h{self.max_horizon_periods}"

    def with_horizon(self, horizon: int) -> "ForecastConfig":
        config = replace(self)
        if self.aggregation_level == "month":
            config._active_horizon_months = horizon
        else:
            config._active_horizon_weeks = horizon
        return config

    def target_col_for_horizon(self, horizon: int) -> str:
        return f"{self.target_col}_t_plus_{horizon}"

    def target_scaled_col_for_horizon(self, horizon: int) -> str:
        return f"{self.target_col}_scaled_t_plus_{horizon}"

    @property
    def active_target_col(self) -> str:
        return self.target_col_for_horizon(self.horizon_periods)

    @property
    def active_target_scaled_col(self) -> str:
        return self.target_scaled_col_for_horizon(self.horizon_periods)


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.where(np.abs(denominator) > EPS, np.nan)
    return numerator / denominator


def add_lag_features(
    df: pd.DataFrame,
    config: ForecastConfig = ForecastConfig(),
) -> pd.DataFrame:
    data = df.copy()
    data = data.sort_values(config.group_cols + [config.period_col]).reset_index(drop=True)
    group = data.groupby(config.group_cols, sort=False)[config.value_col]

    for lag in config.lag_periods:
        if lag == 0:
            data[f"lag_{lag}"] = data[config.value_col]
        else:
            data[f"lag_{lag}"] = group.shift(lag)

    return data


def add_rolling_features(
    df: pd.DataFrame,
    config: ForecastConfig = ForecastConfig(),
) -> pd.DataFrame:
    data = df.copy()
    group = data.groupby(config.group_cols, sort=False)[config.value_col]

    for window in config.rolling_windows:
        data[f"rolling_mean_{window}"] = group.transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        data[f"rolling_median_{window}"] = group.transform(
            lambda s: s.rolling(window, min_periods=1).median()
        )
        data[f"rolling_std_{window}"] = group.transform(
            lambda s: s.rolling(window, min_periods=2).std()
        )
        data[f"rolling_min_{window}"] = group.transform(
            lambda s: s.rolling(window, min_periods=1).min()
        )
        data[f"rolling_max_{window}"] = group.transform(
            lambda s: s.rolling(window, min_periods=1).max()
        )

        q75 = group.transform(
            lambda s: s.rolling(window, min_periods=2).quantile(0.75)
        )
        q25 = group.transform(
            lambda s: s.rolling(window, min_periods=2).quantile(0.25)
        )
        data[f"rolling_iqr_{window}"] = q75 - q25

    return data


def add_ewm_features(
    df: pd.DataFrame,
    config: ForecastConfig = ForecastConfig(),
) -> pd.DataFrame:
    data = df.copy()
    group = data.groupby(config.group_cols, sort=False)[config.value_col]

    for span in config.ewm_spans:
        data[f"ewm_mean_{span}"] = group.transform(
            lambda s: s.ewm(span=span, adjust=False, min_periods=1).mean()
        )

    return data


def rolling_slope(values: np.ndarray) -> float:
    y = np.asarray(values, dtype=float)
    mask = np.isfinite(y)

    if mask.sum() < 2:
        return np.nan

    x = np.arange(len(y), dtype=float)[mask]
    y = y[mask]
    return float(np.polyfit(x, y, 1)[0])


def add_trend_features(
    df: pd.DataFrame,
    config: ForecastConfig = ForecastConfig(),
) -> pd.DataFrame:
    data = df.copy()

    if {"lag_0", "lag_1"}.issubset(data.columns):
        data["momentum_diff_1"] = data["lag_0"] - data["lag_1"]
        data["momentum_ratio_1"] = safe_divide(data["lag_0"], data["lag_1"])

    if {"lag_0", "lag_4"}.issubset(data.columns):
        data["momentum_diff_4"] = data["lag_0"] - data["lag_4"]
        data["momentum_ratio_4"] = safe_divide(data["lag_0"], data["lag_4"])

    if {"rolling_mean_4", "rolling_mean_13"}.issubset(data.columns):
        data["momentum_mean_4_vs_13"] = data["rolling_mean_4"] - data["rolling_mean_13"]
        data["momentum_ratio_4_vs_13"] = safe_divide(
            data["rolling_mean_4"],
            data["rolling_mean_13"],
        )

    if {"rolling_mean_8", "rolling_mean_26"}.issubset(data.columns):
        data["momentum_mean_8_vs_26"] = data["rolling_mean_8"] - data["rolling_mean_26"]
        data["momentum_ratio_8_vs_26"] = safe_divide(
            data["rolling_mean_8"],
            data["rolling_mean_26"],
        )

    group = data.groupby(config.group_cols, sort=False)[config.value_col]

    for window in config.slope_windows:
        data[f"rolling_slope_{window}"] = group.transform(
            lambda s: s.rolling(window, min_periods=2).apply(rolling_slope, raw=True)
        )

    return data


def add_seasonality_features(
    df: pd.DataFrame,
    config: ForecastConfig = ForecastConfig(),
) -> pd.DataFrame:
    data = df.copy()
    period_series = data[config.period_col]
    iso = period_series.dt.isocalendar()
    data["year"] = period_series.dt.year.astype(int)
    data["week_of_year"] = iso.week.astype(int)
    data["month_of_year"] = period_series.dt.month.astype(int)
    data["quarter_of_year"] = period_series.dt.quarter.astype(int)
    period_days = 30.4375 if config.aggregation_level == "month" else 7.0
    data["time_index"] = np.floor(
        (period_series - period_series.min()).dt.days / period_days
    ).astype(int)

    seasonal_position = data["month_of_year"] if config.aggregation_level == "month" else data["week_of_year"]
    seasonal_cycle = 12.0 if config.aggregation_level == "month" else 52.1775

    for k in range(1, config.fourier_order + 1):
        angle = 2 * np.pi * k * seasonal_position / seasonal_cycle
        data[f"fourier_sin_{k}"] = np.sin(angle)
        data[f"fourier_cos_{k}"] = np.cos(angle)

    last_year_lags = [11, 12, 13] if config.aggregation_level == "month" else [51, 52, 53]
    last_year_cols = [col for col in [f"lag_{lag}" for lag in last_year_lags] if col in data.columns]

    if last_year_cols:
        data["last_year_mean_3"] = data[last_year_cols].mean(axis=1)
        data["last_year_median_3"] = data[last_year_cols].median(axis=1)

        if "lag_0" in data.columns:
            data["last_year_ratio_3"] = safe_divide(
                data["lag_0"],
                data["last_year_mean_3"],
            )

    return data


def add_intermittency_features(
    df: pd.DataFrame,
    config: ForecastConfig = ForecastConfig(),
) -> pd.DataFrame:
    data = df.copy()
    data["is_nonzero_lag_0"] = (data[config.value_col] > 0).astype(int)
    group = data.groupby(config.group_cols, sort=False)[config.value_col]

    for window in config.nonzero_windows:
        data[f"nonzero_rate_{window}"] = group.transform(
            lambda s: (s > 0).astype(float).rolling(window, min_periods=1).mean()
        )

    return data


def rolling_mad(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    median = np.median(x)
    return float(np.median(np.abs(x - median)))


def weeks_since_event(flags: pd.Series) -> pd.Series:
    output = []
    last_event_position = None

    for i, flag in enumerate(flags.fillna(0).astype(int).to_numpy()):
        if flag == 1:
            last_event_position = i
            output.append(0.0)
        elif last_event_position is None:
            output.append(np.nan)
        else:
            output.append(float(i - last_event_position))

    return pd.Series(output, index=flags.index)


def add_spike_features(
    df: pd.DataFrame,
    config: ForecastConfig = ForecastConfig(),
) -> pd.DataFrame:
    data = df.copy()
    group = data.groupby(config.group_cols, sort=False)[config.value_col]
    window = config.spike_window

    data[f"spike_prior_median_{window}"] = group.transform(
        lambda s: s.shift(1).rolling(window, min_periods=4).median()
    )
    data[f"spike_prior_mad_{window}"] = group.transform(
        lambda s: s.shift(1).rolling(window, min_periods=4).apply(rolling_mad, raw=True)
    )

    denominator = 1.4826 * data[f"spike_prior_mad_{window}"] + EPS
    data[f"robust_z_spike_{window}"] = (
        data[config.value_col] - data[f"spike_prior_median_{window}"]
    ) / denominator

    valid_spike_context = (
        data[f"spike_prior_median_{window}"].notna()
        & data[f"spike_prior_mad_{window}"].notna()
    )

    data["is_spike"] = (
        valid_spike_context
        & (data[f"robust_z_spike_{window}"] > config.spike_z_threshold)
        & (data[config.value_col] > data[f"spike_prior_median_{window}"])
    ).astype(int)

    data["time_since_last_spike"] = (
        data.groupby(config.group_cols, group_keys=False)["is_spike"].apply(weeks_since_event)
    )

    spike_group = data.groupby(config.group_cols, sort=False)["is_spike"]

    for window in config.spike_rate_windows:
        data[f"spike_rate_{window}"] = spike_group.transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )

    return data


def add_target(
    df: pd.DataFrame,
    config: ForecastConfig = ForecastConfig(),
) -> pd.DataFrame:
    data = df.copy()

    grouped_values = data.groupby(config.group_cols, sort=False)[config.value_col]
    for horizon in config.forecast_horizons:
        data[config.target_col_for_horizon(horizon)] = grouped_values.shift(-horizon)

    data[config.target_col] = data[config.active_target_col]
    return data


def add_scale_features(
    df: pd.DataFrame,
    config: ForecastConfig = ForecastConfig(),
) -> pd.DataFrame:
    data = df.copy()
    group = data.groupby(config.group_cols, sort=False)[config.value_col]
    data["scale_factor"] = group.transform(
        lambda s: s.rolling(config.scale_window, min_periods=1).mean()
    )
    data["scale_factor"] = data["scale_factor"].fillna(config.min_scale).clip(lower=config.min_scale)
    data["log_scale_factor"] = np.log1p(data["scale_factor"])

    for horizon in config.forecast_horizons:
        target_col = config.target_col_for_horizon(horizon)
        target_scaled_col = config.target_scaled_col_for_horizon(horizon)
        if target_col in data.columns:
            data[target_scaled_col] = data[target_col] / data["scale_factor"]

    if config.active_target_col in data.columns:
        data["target_scaled"] = data[config.active_target_scaled_col]

    return data


def identify_demand_feature_columns(df: pd.DataFrame) -> List[str]:
    demand_prefixes = (
        "lag_",
        "rolling_mean_",
        "rolling_median_",
        "rolling_std_",
        "rolling_min_",
        "rolling_max_",
        "rolling_iqr_",
        "ewm_mean_",
        "momentum_diff_",
        "momentum_mean_",
        "rolling_slope_",
        "last_year_mean_",
        "last_year_median_",
        "spike_prior_median_",
        "spike_prior_mad_",
    )
    demand_cols = []

    for col in df.columns:
        if col.endswith("_scaled"):
            continue
        if "ratio" in col:
            continue
        if col.startswith(demand_prefixes):
            demand_cols.append(col)

    return demand_cols


def add_scaled_demand_features(
    df: pd.DataFrame,
    config: ForecastConfig = ForecastConfig(),
) -> pd.DataFrame:
    data = df.copy()
    demand_cols = identify_demand_feature_columns(data)

    for col in demand_cols:
        data[f"{col}_scaled"] = data[col] / data["scale_factor"]

    return data


def add_time_decay_weights(
    df: pd.DataFrame,
    config: ForecastConfig = ForecastConfig(),
    reference_week: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    data = df.copy()

    if reference_week is None:
        reference_week = data[config.period_col].max()

    period_days = 30.4375 if config.aggregation_level == "month" else 7.0
    age_weeks = ((reference_week - data[config.period_col]).dt.days / period_days).clip(lower=0)
    data["sample_weight"] = 0.5 ** (
        age_weeks / max(config.time_decay_half_life_periods, 1)
    )

    return data


def build_modeling_table(
    df: pd.DataFrame,
    config: ForecastConfig = ForecastConfig(),
) -> pd.DataFrame:
    data = add_lag_features(df, config)
    data = add_rolling_features(data, config)
    data = add_ewm_features(data, config)
    data = add_trend_features(data, config)
    data = add_seasonality_features(data, config)
    data = add_intermittency_features(data, config)
    data = add_spike_features(data, config)
    data = add_target(data, config)
    data = add_scale_features(data, config)
    data = add_scaled_demand_features(data, config)
    data = add_time_decay_weights(data, config)
    return data.sort_values(config.group_cols + [config.period_col]).reset_index(drop=True)


def is_feature_selected_for_training(
    col: str,
    config: ForecastConfig,
    use_scaled_demand_features: bool,
) -> bool:
    selected_families = set(config.training_feature_families)

    demand_prefix_by_family = {
        "lag": "lag_",
        "rolling_mean": "rolling_mean_",
        "rolling_median": "rolling_median_",
        "rolling_std": "rolling_std_",
        "rolling_min": "rolling_min_",
        "rolling_max": "rolling_max_",
        "rolling_iqr": "rolling_iqr_",
        "ewm_mean": "ewm_mean_",
        "momentum_diff": "momentum_diff_",
        "momentum_ratio": "momentum_ratio_",
        "rolling_slope": "rolling_slope_",
        "last_year_mean_3": "last_year_mean_3",
        "last_year_median_3": "last_year_median_3",
        "last_year_ratio_3": "last_year_ratio_3",
        "spike_prior_mad": "spike_prior_mad_",
        "spike_prior_median": "spike_prior_median_",
        "robust_z_spike": "robust_z_spike_",
    }
    seasonal_prefix_by_family = {
        "fourier_cos": "fourier_cos_",
        "fourier_sin": "fourier_sin_",
        "spike_rate": "spike_rate_",
        "nonzero_rate": "nonzero_rate_",
    }
    exact_match_by_family = {
        "code": "code",
        "warehouse": "warehouse",
        "week_of_year": "week_of_year",
        "month_of_year": "month_of_year",
        "quarter_of_year": "quarter_of_year",
        "year": "year",
        "is_spike": "is_spike",
        "time_since_last_spike": "time_since_last_spike",
        "is_nonzero_lag_0": "is_nonzero_lag_0",
        "log_scale_factor": "log_scale_factor",
        "scale_factor": "scale_factor",
    }

    for family, prefix in demand_prefix_by_family.items():
        if family not in selected_families:
            continue
        if use_scaled_demand_features:
            if col.startswith(prefix) and col.endswith("_scaled"):
                return True
        elif col.startswith(prefix) and not col.endswith("_scaled"):
            return True

    for family, prefix in seasonal_prefix_by_family.items():
        if family in selected_families and col.startswith(prefix):
            return True

    for family, exact_col in exact_match_by_family.items():
        if family in selected_families and col == exact_col:
            return True

    return False


def build_feature_columns(
    modeling_table: pd.DataFrame,
    config: ForecastConfig = ForecastConfig(),
    use_scaled_demand_features: bool = True,
) -> Tuple[List[str], List[str]]:
    excluded_cols = {
        config.value_col,
        config.target_col,
        config.period_col,
        "sample_weight",
    }
    excluded_cols.update(
        config.target_col_for_horizon(horizon)
        for horizon in config.forecast_horizons
    )
    excluded_cols.update(
        config.target_scaled_col_for_horizon(horizon)
        for horizon in config.forecast_horizons
    )
    excluded_cols.add("target_scaled")

    demand_cols = identify_demand_feature_columns(modeling_table)

    if use_scaled_demand_features:
        excluded_cols.update(demand_cols)
    else:
        excluded_cols.update([f"{col}_scaled" for col in demand_cols])

    allowed_lags = {f"lag_{lag}" for lag in config.lag_periods}
    allowed_lag_features = (
        {f"lag_{lag}_scaled" for lag in config.lag_periods}
        if use_scaled_demand_features
        else allowed_lags
    )
    feature_cols = [
        col
        for col in modeling_table.columns
        if col not in excluded_cols
        and not col.startswith("_")
        and is_feature_selected_for_training(
            col,
            config=config,
            use_scaled_demand_features=use_scaled_demand_features,
        )
        and (
            not col.startswith("lag_")
            or col in allowed_lag_features
        )
    ]

    cat_features = [
        col
        for col in config.group_cols + ["week_of_year", "month_of_year", "quarter_of_year"]
        if col in feature_cols
    ]

    return feature_cols, cat_features
