"""
Data loading module for NLP sentiment analysis.
Handles downloading and loading datasets from various sources.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import boto3
from datasets import load_dataset

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and manage NLP datasets."""

    def __init__(self, data_dir: str = "data", aws_profile: Optional[str] = None):
        """
        Initialize DataLoader.

        Args:
            data_dir: Directory to store downloaded datasets
            aws_profile: AWS profile for S3 access
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.aws_profile = aws_profile

    def load_huggingface_dataset(
        self, dataset_name: str, split: str = "train", sample_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load dataset from Hugging Face.

        Args:
            dataset_name: HF dataset identifier (e.g., 'imdb', 'glue')
            split: Dataset split (train, validation, test)
            sample_size: Number of samples to load (None for all)

        Returns:
            DataFrame with dataset samples
        """
        logger.info(f"Loading {dataset_name} from Hugging Face...")

        try:
            # Try to load with specific split, disabling problematic verifications
            dataset = load_dataset(
                dataset_name, split=split, verification_mode="no_checks"
            )
            df = pd.DataFrame(dataset)

        except Exception as e:
            logger.error(
                f"Failed to load dataset {dataset_name} with verification_mode: {str(e)}"
            )
            # Fallback: try loading all splits
            try:
                logger.warning(f"Attempting fallback method: loading all splits")
                dataset_dict = load_dataset(dataset_name, verification_mode="no_checks")
                if split in dataset_dict:
                    dataset = dataset_dict[split]
                else:
                    # Use first available split
                    available_splits = list(dataset_dict.keys())
                    logger.warning(
                        f"Split '{split}' not found. Available splits: {available_splits}. Using '{available_splits[0]}'"
                    )
                    dataset = dataset_dict[available_splits[0]]
                df = pd.DataFrame(dataset)
            except Exception as inner_e:
                logger.error(f"Failed to load dataset {dataset_name}: {str(inner_e)}")
                raise

        if sample_size:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)

        logger.info(f"Loaded {len(df)} samples from {dataset_name}")
        return df

    def load_from_s3(
        self, bucket: str, key: str, local_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load CSV data from S3.

        Args:
            bucket: S3 bucket name
            key: S3 object key
            local_path: Local save path (optional)

        Returns:
            DataFrame with S3 data
        """
        logger.info(f"Loading data from s3://{bucket}/{key}...")

        try:
            session = boto3.Session(profile_name=self.aws_profile)
            s3_client = session.client("s3")

            # Download file
            obj = s3_client.get_object(Bucket=bucket, Key=key)
            df = pd.read_csv(obj["Body"])

            if local_path:
                df.to_csv(local_path, index=False)
                logger.info(f"Saved to {local_path}")

            logger.info(f"Loaded {len(df)} rows from S3")
            return df

        except Exception as e:
            logger.error(f"Failed to load from S3: {str(e)}")
            raise

    def load_local_csv(self, filepath: str) -> pd.DataFrame:
        """Load CSV from local filesystem."""
        logger.info(f"Loading local CSV: {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} rows")
        return df

    def save_to_s3(self, df: pd.DataFrame, bucket: str, key: str) -> None:
        """Save DataFrame to S3 as CSV."""
        try:
            csv_buffer = df.to_csv(index=False)
            session = boto3.Session(profile_name=self.aws_profile)
            s3_client = session.client("s3")
            s3_client.put_object(
                Bucket=bucket, Key=key, Body=csv_buffer.encode("utf-8")
            )
            logger.info(f"Saved {len(df)} rows to s3://{bucket}/{key}")
        except Exception as e:
            logger.error(f"Failed to save to S3: {str(e)}")
            raise

    def create_train_test_split(
        self,
        df: pd.DataFrame,
        text_column: str,
        label_column: str,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create train/test split for NLP data.

        Args:
            df: Input DataFrame
            text_column: Name of text column
            label_column: Name of label column
            test_size: Test set fraction
            random_state: Random seed

        Returns:
            Tuple of (train_df, test_df)
        """
        from sklearn.model_selection import train_test_split

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df[label_column],
        )

        logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")
        return train_df, test_df


def create_train_test_split(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create train/test split for NLP data.

    Args:
        df: Input DataFrame
        text_column: Name of text column
        label_column: Name of label column
        test_size: Test set fraction
        random_state: Random seed

    Returns:
        Tuple of (train_df, test_df)
    """
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[label_column]
    )

    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")
    return train_df, test_df
