import random
from statistics import mean
from typing import Any, Dict, List

from clearml import PipelineDecorator, Task


PROJECT_NAME = "Time Series Auto"
PIPELINE_NAME = "ClearML Simple Pipeline Demo Remote"


@PipelineDecorator.component(return_values=["numbers"])
def load_data(sample_size: int, random_seed: int) -> List[float]:
    random.seed(random_seed)
    numbers = [round(random.uniform(0, 1), 4) for _ in range(sample_size)]

    task = Task.current_task()
    task.get_logger().report_scalar(
        title="data",
        series="sample_size",
        value=float(sample_size),
        iteration=0,
    )
    task.get_logger().report_text(f"Generated {sample_size} random numbers")

    return numbers


@PipelineDecorator.component(return_values=["scaled_numbers"])
def preprocess_data(numbers: List[float], multiplier: float) -> List[float]:
    scaled_numbers = [round(value * multiplier, 4) for value in numbers]

    task = Task.current_task()
    task.get_logger().report_scalar(
        title="preprocess",
        series="multiplier",
        value=float(multiplier),
        iteration=0,
    )

    return scaled_numbers


@PipelineDecorator.component(return_values=["model_summary"])
def train_model(scaled_numbers: List[float], threshold: float) -> Dict[str, Any]:
    average_value = mean(scaled_numbers)
    values_above_threshold = sum(value > threshold for value in scaled_numbers)

    model_summary = {
        "average_value": round(average_value, 4),
        "threshold": threshold,
        "values_above_threshold": values_above_threshold,
        "total_values": len(scaled_numbers),
    }

    task = Task.current_task()
    task.get_logger().report_scalar(
        title="train",
        series="average_value",
        value=float(model_summary["average_value"]),
        iteration=0,
    )
    task.upload_artifact(name="model_summary", artifact_object=model_summary)

    return model_summary


@PipelineDecorator.component(return_values=["evaluation"])
def evaluate_model(model_summary: Dict[str, Any]) -> Dict[str, Any]:
    ratio_above_threshold = 0.0
    if model_summary["total_values"]:
        ratio_above_threshold = model_summary["values_above_threshold"] / model_summary["total_values"]

    evaluation = {
        "ratio_above_threshold": round(ratio_above_threshold, 4),
        "average_value": model_summary["average_value"],
        "threshold": model_summary["threshold"],
    }

    task = Task.current_task()
    task.get_logger().report_scalar(
        title="evaluation",
        series="ratio_above_threshold",
        value=float(evaluation["ratio_above_threshold"]),
        iteration=0,
    )
    task.upload_artifact(name="evaluation", artifact_object=evaluation)

    return evaluation


@PipelineDecorator.pipeline(
    name=PIPELINE_NAME,
    project=PROJECT_NAME,
    version="1.0",
)
def simple_learning_pipeline(
    sample_size: int = 20,
    random_seed: int = 42,
    multiplier: float = 10.0,
    threshold: float = 5.0,
) -> None:
    numbers = load_data(sample_size=sample_size, random_seed=random_seed)
    scaled_numbers = preprocess_data(numbers=numbers, multiplier=multiplier)
    model_summary = train_model(scaled_numbers=scaled_numbers, threshold=threshold)
    evaluation = evaluate_model(model_summary=model_summary)

    print("Pipeline finished successfully")
    print(f"Final evaluation: {evaluation}")


def main() -> None:
    simple_learning_pipeline(
        sample_size=20,
        random_seed=42,
        multiplier=10.0,
        threshold=5.0,
    )


if __name__ == "__main__":
    main()
