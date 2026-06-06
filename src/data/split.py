from sklearn.model_selection import train_test_split
from omegaconf import DictConfig
import pandas as pd


def split_data(df: pd.DataFrame, split_cfg: DictConfig):
    """
    Split the dataset into train and validation sets.
    """
    stratify_col = df["target"] if split_cfg.stratify else None
    
    df_train, df_val = train_test_split(
        df,
        test_size=split_cfg.test_size,
        random_state=split_cfg.random_state,
        stratify=stratify_col
    )
    return df_train, df_val
