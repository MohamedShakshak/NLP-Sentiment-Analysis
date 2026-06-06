from pathlib import Path
import pandas as pd
from omegaconf import DictConfig


def load_data(cfg: DictConfig) -> pd.DataFrame:
    """
    Load dataset according to Hydra dataset configuration.
    """
    dataset_cfg = cfg.dataset

    path = Path(dataset_cfg.raw_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(
        path,
        **dataset_cfg.read_csv_kwargs
    )

    # Rename columns
    df = df.rename(
        columns=dataset_cfg.column_mapping
    )

    # Drop unwanted classes
    if dataset_cfg.drop_classes:
        df = df[
            ~df["target"].isin(
                dataset_cfg.drop_classes
            )
        ]

    # Remap target labels
    df["target"] = (
        df["target"]
        .map(dataset_cfg.target_mapping)
    )

    # Remove rows with unmapped targets
    df = df.dropna(
        subset=["target"]
    )

    df["target"] = df["target"].astype(int)

    return df
