"""
HuggingFace upload utilities for SCAR data pipeline.
Handles dataset creation and push operations.
"""

from pathlib import Path
from datasets import Dataset, DatasetDict
import pandas as pd


def parquet_to_hf_dataset(parquet_path: str) -> Dataset:
    """Load a parquet file into an HF Dataset."""
    df = pd.read_parquet(parquet_path)
    return Dataset.from_pandas(df)


def push_dataset_to_hub(
    dataset: Dataset,
    repo_id: str,
    config_name: str = "default",
    private: bool = True,
    token: str | None = None
):
    """Push a dataset to HuggingFace Hub.

    Args:
        dataset: HF Dataset object
        repo_id: e.g. "username/scar-corpus"
        config_name: Config/subset name (default "default")
        private: Whether to keep dataset private (default True)
        token: HF API token (if not using huggingface-cli login)
    """
    dataset.push_to_hub(
        repo_id,
        config_name=config_name,
        private=private,
        token=token,
    )
    print(f"Pushed {len(dataset)} rows to {repo_id} (config: {config_name})")


def save_intermediate(df: pd.DataFrame, drive_path: str, name: str) -> str:
    """Save intermediate DataFrame as parquet to Google Drive.

    Args:
        df: Processed DataFrame
        drive_path: Base path in Google Drive (e.g., '/content/drive/MyDrive/SCAR/intermediates')
        name: Filename without extension (e.g., 'disl_processed')

    Returns:
        Full path to saved file.
    """
    Path(drive_path).mkdir(parents=True, exist_ok=True)
    filepath = f"{drive_path}/{name}.parquet"
    df.to_parquet(filepath, index=False)
    print(f"Saved {len(df)} rows to {filepath}")
    return filepath


def load_intermediate(drive_path: str, name: str) -> pd.DataFrame | None:
    """Load intermediate parquet from Google Drive if it exists.

    Returns None if file doesn't exist (signals need to reprocess).
    """
    filepath = f"{drive_path}/{name}.parquet"
    if Path(filepath).exists():
        df = pd.read_parquet(filepath)
        print(f"Loaded existing intermediate: {filepath} ({len(df)} rows)")
        return df
    return None


def check_intermediate_exists(drive_path: str, name: str) -> bool:
    """Check if an intermediate file already exists (for disconnect resilience)."""
    filepath = f"{drive_path}/{name}.parquet"
    exists = Path(filepath).exists()
    if exists:
        print(f"Intermediate already exists: {filepath} — skipping processing")
    return exists
