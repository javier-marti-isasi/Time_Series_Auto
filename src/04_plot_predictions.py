from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from utils.utils_feature_engineering import ForecastConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_BY_MODEL_DIR = PLOTS_DIR / "plots_by_model"
PLOTS_BY_FORECAST_ORIGIN_DIR = PLOTS_DIR / "plots_by_forecast_origin"

DATASET_NAME: str = "Competition"
GROUP_FILTER_VALUES: list[str] = ["146", "62"] #["40002456", "4201"]
 # ["code", "warehouse"] for Lecta
 # ["product", "store"] for Competition

AGGREGATION_LEVEL = "month"
MAX_HORIZON = 6


def build_results_suffix(config: ForecastConfig) -> str:
    return f"{config.aggregation_level}_h{config.max_horizon_periods}"


def build_predictions_path(config: ForecastConfig) -> Path:
    suffix = build_results_suffix(config)
    return RESULTS_DIR / config.build_dataset_file_name("test_predictions", suffix=suffix, extension=".csv")


def build_consolidated_predictions_path(config: ForecastConfig) -> Path:
    suffix = build_results_suffix(config)
    return RESULTS_DIR / config.build_dataset_file_name("test_predictions_multi_horizon", suffix=suffix, extension=".csv")


def build_processed_input_path(config: ForecastConfig) -> Path:
    return config.processed_output_path


def resolve_plot_dpi(config: ForecastConfig) -> int:
    return 450 if config.aggregation_level == "week" else 150


def resolve_line_widths(config: ForecastConfig) -> dict[str, float]:
    if config.aggregation_level == "week":
        return {
            "history": 0.6,
            "prediction": 0.6,
            "target": 0.7,
            "origin": 0.6,
        }

    return {
        "history": 2.0,
        "prediction": 1.8,
        "target": 2.0,
        "origin": 1.5,
    }


def resolve_marker_sizes(config: ForecastConfig) -> dict[str, float]:
    if config.aggregation_level == "week":
        return {
            "history": 1,
            "prediction": 1,
            "target": 1,
        }

    return {
        "history": 4.0,
        "prediction": 5.0,
        "target": 5.0,
    }


def resolve_period_column(df: pd.DataFrame, config: ForecastConfig) -> str:
    if config.period_col in df.columns:
        return config.period_col

    for candidate in ["month", "week"]:
        if candidate in df.columns:
            return candidate

    raise KeyError(f"No supported period column found. Expected one of: {config.period_col}, month, week")


def load_historical_series(config: ForecastConfig) -> pd.DataFrame:
    history = pd.read_parquet(build_processed_input_path(config))
    period_col = resolve_period_column(history, config)
    history[period_col] = pd.to_datetime(history[period_col])
    if period_col != config.period_col:
        history[config.period_col] = history[period_col]
    return history.sort_values(config.group_cols + [config.period_col]).reset_index(drop=True)


def parse_plot_datetime(values: pd.Series) -> pd.Series:
    values_as_text = values.astype(str)
    has_slash_dates = values_as_text.str.contains("/", regex=False).any()
    return pd.to_datetime(values, dayfirst=has_slash_dates, errors="coerce")


def parse_plot_numeric(values: pd.Series) -> pd.Series:
    parsed = pd.to_numeric(values, errors="coerce")
    missing_mask = parsed.isna() & values.notna()
    if not missing_mask.any():
        return parsed

    normalized = values.astype(str).str.replace(".", "", regex=False)
    divisors = normalized.str.len().sub(1).rpow(10)
    parsed.loc[missing_mask] = pd.to_numeric(normalized.loc[missing_mask], errors="coerce") / divisors.loc[missing_mask]
    return parsed


def load_predictions(config: ForecastConfig) -> pd.DataFrame:
    predictions = pd.read_csv(build_predictions_path(config), sep=";")
    period_col = resolve_period_column(predictions, config)
    predictions[period_col] = parse_plot_datetime(predictions[period_col])
    if period_col != config.period_col:
        predictions[config.period_col] = predictions[period_col]
    predictions["forecast_period_start"] = parse_plot_datetime(predictions["forecast_period_start"])
    predictions["target"] = parse_plot_numeric(predictions["target"])
    predictions["prediction"] = parse_plot_numeric(predictions["prediction"])
    return predictions.sort_values(
        ["horizon", "predictor", *config.group_cols, "forecast_period_start"]
    ).reset_index(drop=True)


def load_consolidated_predictions(config: ForecastConfig) -> pd.DataFrame:
    predictions = pd.read_csv(build_consolidated_predictions_path(config), sep=";")
    period_col = resolve_period_column(predictions, config)
    predictions[period_col] = parse_plot_datetime(predictions[period_col])
    if period_col != config.period_col:
        predictions[config.period_col] = predictions[period_col]
    for horizon in config.forecast_horizons:
        forecast_col = f"forecast_period_start_t_plus_{horizon}"
        if forecast_col in predictions.columns:
            predictions[forecast_col] = parse_plot_datetime(predictions[forecast_col])
    return predictions.sort_values(config.group_cols + [config.period_col]).reset_index(drop=True)


def filter_group(df: pd.DataFrame, config: ForecastConfig, group_values: Sequence[str]) -> pd.DataFrame:
    filtered = df.copy()
    mask = pd.Series(True, index=filtered.index)
    for col, val in zip(config.group_cols, group_values):
        filtered[col] = filtered[col].astype(str)
        mask &= filtered[col] == str(val)
    return filtered[mask].copy()


def plot_group_predictions(
    history_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    config: ForecastConfig,
    group_values: Sequence[str],
) -> list[Path]:
    _period_label = "month" if config.aggregation_level == "month" else "week"
    _rolling_window = config.baseline_moving_average_months if config.aggregation_level == "month" else config.baseline_moving_average_weeks
    predictor_styles = {
        "catboost": {"color": "#1f77b4", "marker": "o"},
        "catboost_global_t_plus_1": {"color": "#1f77b4", "marker": "o"},
        "catboost_global_t_plus_2": {"color": "#1f77b4", "marker": "o"},
        "catboost_global_t_plus_3": {"color": "#1f77b4", "marker": "o"},
        "catboost_global_t_plus_4": {"color": "#1f77b4", "marker": "o"},
        "catboost_global_t_plus_5": {"color": "#1f77b4", "marker": "o"},
        "catboost_global_t_plus_6": {"color": "#1f77b4", "marker": "o"},
        "hwes": {"color": "#9467bd", "marker": "P"},
        f"baseline_previous_{_period_label}": {"color": "#ff7f0e", "marker": "s"},
        f"baseline_mean_last_{_rolling_window}_{_period_label}s": {"color": "#2ca02c", "marker": "^"},
    }
    line_widths = resolve_line_widths(config)
    marker_sizes = resolve_marker_sizes(config)

    saved_paths: list[Path] = []
    _group_suffix = "_".join(str(v) for v in group_values)
    plot_prefix = f"{config.aggregation_level}_h{config.max_horizon_periods}_{_group_suffix}"

    for horizon in config.forecast_horizons:
        horizon_predictions = predictions_df[predictions_df["horizon"] == horizon].copy()
        if horizon_predictions.empty:
            continue

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(
            history_df[config.period_col],
            history_df[config.value_col],
            color="#111111",
            linewidth=line_widths["history"],
            marker="o",
            markersize=marker_sizes["history"],
            label="real_series",
        )

        for predictor, predictor_df in horizon_predictions.groupby("predictor", sort=False):
            predictor_df = predictor_df.dropna(subset=["prediction"]).copy()
            if predictor_df.empty:
                continue
            style = predictor_styles.get(
                predictor,
                {"color": "#7f7f7f", "marker": "o"},
            )
            ax.plot(
                predictor_df["forecast_period_start"],
                predictor_df["prediction"],
                linestyle="--",
                linewidth=line_widths["prediction"],
                marker=style["marker"],
                markersize=marker_sizes["prediction"],
                color=style["color"],
                label=predictor,
            )

        actual_future = (
            horizon_predictions[["forecast_period_start", "target"]]
            .drop_duplicates()
            .sort_values("forecast_period_start")
        )
        ax.plot(
            actual_future["forecast_period_start"],
            actual_future["target"],
            color="#d62728",
            linewidth=line_widths["target"],
            marker="D",
            markersize=marker_sizes["target"],
            label=f"target_t_plus_{horizon}",
        )

        _group_title = ", ".join(f"{c}={v}" for c, v in zip(config.group_cols, group_values))
        ax.set_title(f"Predictions for {_group_title}, horizon=t+{horizon}")
        ax.set_xlabel("period")
        ax.set_ylabel(config.value_col)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()

        output_path = PLOTS_BY_MODEL_DIR / f"{plot_prefix}_t_plus_{horizon}.png"
        fig.savefig(output_path, dpi=resolve_plot_dpi(config), bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(output_path)

    return saved_paths


def build_forecast_origin_plot_table(
    predictions_df: pd.DataFrame,
    config: ForecastConfig,
) -> pd.DataFrame:
    required_cols = config.group_cols + [
        config.period_col,
        "forecast_period_start",
        "target",
        "predictor",
        "prediction",
        "horizon",
    ]
    available_cols = [col for col in required_cols if col in predictions_df.columns]
    if len(available_cols) != len(required_cols):
        return pd.DataFrame()

    forecast_origin_df = predictions_df.loc[:, required_cols].copy().rename(
        columns={config.period_col: "forecast_origin"}
    )

    return forecast_origin_df.sort_values(
        [*config.group_cols, "forecast_origin", "horizon"]
    ).reset_index(drop=True)


def plot_group_predictions_by_forecast_origin(
    history_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    config: ForecastConfig,
    group_values: Sequence[str],
) -> list[Path]:
    _period_label = "month" if config.aggregation_level == "month" else "week"
    _rolling_window = config.baseline_moving_average_months if config.aggregation_level == "month" else config.baseline_moving_average_weeks
    predictor_styles = {
        "catboost": {"color": "#1f77b4", "marker": "o"},
        "catboost_global_t_plus_1": {"color": "#1f77b4", "marker": "o"},
        "catboost_global_t_plus_2": {"color": "#1f77b4", "marker": "o"},
        "catboost_global_t_plus_3": {"color": "#1f77b4", "marker": "o"},
        "catboost_global_t_plus_4": {"color": "#1f77b4", "marker": "o"},
        "catboost_global_t_plus_5": {"color": "#1f77b4", "marker": "o"},
        "catboost_global_t_plus_6": {"color": "#1f77b4", "marker": "o"},
        "hwes": {"color": "#9467bd", "marker": "P"},
        f"baseline_previous_{_period_label}": {"color": "#ff7f0e", "marker": "s"},
        f"baseline_mean_last_{_rolling_window}_{_period_label}s": {"color": "#2ca02c", "marker": "^"},
    }
    line_widths = resolve_line_widths(config)
    marker_sizes = resolve_marker_sizes(config)
    saved_paths: list[Path] = []
    _group_suffix = "_".join(str(v) for v in group_values)
    plot_prefix = f"{config.aggregation_level}_h{config.max_horizon_periods}_{_group_suffix}"
    history_series = history_df[[config.period_col, config.value_col]].drop_duplicates().sort_values(config.period_col)

    for forecast_origin, origin_df in predictions_df.groupby("forecast_origin", sort=True):
        origin_df = origin_df.sort_values(["predictor", "horizon"]).copy()
        if origin_df.empty:
            continue

        actual_future = (
            origin_df[["forecast_period_start", "target"]]
            .drop_duplicates()
            .sort_values("forecast_period_start")
        )

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(
            history_series[config.period_col],
            history_series[config.value_col],
            color="#111111",
            linewidth=line_widths["history"],
            marker="o",
            markersize=marker_sizes["history"],
            label="real_series",
        )
        ax.axvline(forecast_origin, color="#7f7f7f", linestyle=":", linewidth=line_widths["origin"], label="forecast_origin")
        ax.plot(
            actual_future["forecast_period_start"],
            actual_future["target"],
            color="#d62728",
            linewidth=line_widths["target"],
            marker="D",
            markersize=marker_sizes["target"],
            label="target_real",
        )

        for predictor, predictor_df in origin_df.groupby("predictor", sort=False):
            predictor_df = predictor_df.dropna(subset=["prediction"]).sort_values("horizon")
            if predictor_df.empty:
                continue
            style = predictor_styles.get(
                predictor,
                {"color": "#7f7f7f", "marker": "o"},
            )
            ax.plot(
                predictor_df["forecast_period_start"],
                predictor_df["prediction"],
                color=style["color"],
                linewidth=line_widths["prediction"],
                linestyle="--",
                marker=style["marker"],
                markersize=marker_sizes["prediction"],
                label=predictor,
            )

        _group_title = ", ".join(f"{c}={v}" for c, v in zip(config.group_cols, group_values))
        ax.set_title(
            f"Forecast origin={forecast_origin.strftime('%Y-%m-%d')} for {_group_title}"
        )
        ax.set_xlabel("period")
        ax.set_ylabel(config.value_col)
        ax.grid(True, alpha=0.25)
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()
        fig.tight_layout()

        output_path = (
            PLOTS_BY_FORECAST_ORIGIN_DIR
            / f"{plot_prefix}_origin_{forecast_origin.strftime('%Y_%m_%d')}.png"
        )
        fig.savefig(output_path, dpi=resolve_plot_dpi(config), bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(output_path)

    return saved_paths


def main() -> None:
    config = ForecastConfig(
        dataset_name=DATASET_NAME,
        aggregation_level=AGGREGATION_LEVEL,
        max_horizon_months=MAX_HORIZON if AGGREGATION_LEVEL == "month" else 1,
        max_horizon_weeks=MAX_HORIZON if AGGREGATION_LEVEL == "week" else 1,
    )

    history_df = filter_group(load_historical_series(config), config, GROUP_FILTER_VALUES)
    predictions_df = filter_group(load_predictions(config), config, GROUP_FILTER_VALUES)
    _group_label = ", ".join(f"{c}={v}" for c, v in zip(config.group_cols, GROUP_FILTER_VALUES))
    if history_df.empty:
        raise ValueError(f"No historical data found for {_group_label}.")

    if predictions_df.empty:
        raise ValueError(f"No predictions found for {_group_label}.")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_BY_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_BY_FORECAST_ORIGIN_DIR.mkdir(parents=True, exist_ok=True)

    saved_paths = plot_group_predictions(history_df, predictions_df, config, GROUP_FILTER_VALUES)
    forecast_origin_df = build_forecast_origin_plot_table(predictions_df, config)
    saved_paths.extend(
        plot_group_predictions_by_forecast_origin(
            history_df,
            forecast_origin_df,
            config,
            GROUP_FILTER_VALUES,
        )
    )

    if not saved_paths:
        raise ValueError(f"No plot was generated for {_group_label}.")

    print(f"plots by model directory: {PLOTS_BY_MODEL_DIR}")
    print(f"plots by forecast origin directory: {PLOTS_BY_FORECAST_ORIGIN_DIR}")
    for path in saved_paths:
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
