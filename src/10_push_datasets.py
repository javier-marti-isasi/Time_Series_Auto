from __future__ import annotations
import re
from pathlib import Path
from clearml import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "Competition" / "processed_for_training"


def parse_file_info(file_name: str):
    match = re.search(r"_(month|week)_h\d+_t_plus_(\d+)\.csv$", file_name)
    if not match:
        return None, None
    aggregation = match.group(1)
    horizon = match.group(2)
    return aggregation, horizon


def main() -> None:
    csv_files = sorted(DATA_DIR.glob("*.csv"))

    if not csv_files:
        print(f"No .csv files found in {DATA_DIR}")
        return

    for csv_file in csv_files:
        aggregation, horizon = parse_file_info(csv_file.name)
        if aggregation is None or horizon is None:
            print(f"Skipping unrecognized file: {csv_file.name}")
            continue

        dataset_name = f"competition_processed_for_training_{aggregation}_h{horizon}"
        tags = [aggregation, f"horizon_{horizon}"]

        print(f"Uploading: {csv_file.name} as dataset '{dataset_name}' with tags {tags}")

        dataset = Dataset.create(
            dataset_name=dataset_name,
            dataset_project="Time Series Auto",
            dataset_tags=tags,
        )

        dataset.add_files(path=str(csv_file))
        dataset.upload()
        dataset.finalize()

        print(f"  -> Dataset ID: {dataset.id}")


if __name__ == "__main__":
    main()
