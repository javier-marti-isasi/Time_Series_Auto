from __future__ import annotations
import re
from pathlib import Path
from clearml import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "Competition" / "processed_for_training"
CLEARML_FILES_SERVER = "http://192.168.1.173:8081"


def parse_file_info(file_name: str):
    match = re.search(r"_(month|week)_h\d+_t_plus_(\d+)\.parquet$", file_name)
    if not match:
        return None, None
    aggregation = match.group(1)
    horizon = match.group(2)
    return aggregation, horizon


def main() -> None:
    parquet_files = sorted(DATA_DIR.glob("*.parquet"))

    if not parquet_files:
        print(f"No .parquet files found in {DATA_DIR}")
        return

    for parquet_file in parquet_files:
        aggregation, horizon = parse_file_info(parquet_file.name)
        if aggregation is None or horizon is None:
            print(f"Skipping unrecognized file: {parquet_file.name}")
            continue

        dataset_name = f"competition_processed_for_training_{aggregation}_h{horizon}"

        print(f"Updating {dataset_name}: adding {parquet_file.name} as v1.0.3")

        new_dataset = Dataset.create(
            dataset_name=dataset_name,
            dataset_project="Time Series Auto",
            dataset_version="1.0.3",
            dataset_tags=[aggregation, f"horizon_{horizon}", "parquet"],
        )

        new_dataset.add_files(path=str(parquet_file))
        new_dataset.upload(output_url=CLEARML_FILES_SERVER)
        new_dataset.finalize()

        print(f"  -> New Dataset ID: {new_dataset.id}")


if __name__ == "__main__":
    main()
