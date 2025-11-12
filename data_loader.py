"""
Robust Data Loader for FinanceBench Dataset
Handles loading and preprocessing of FinanceBench data from HuggingFace

This implementation is defensive about dataset -> DataFrame conversions and
avoids boolean checks on arrays/Series that trigger the "truth value of an
array is ambiguous" error.
"""

import logging
from datasets import load_dataset
from typing import Dict, List, Any, Optional
import pandas as pd
import traceback

logger = logging.getLogger(__name__)


class FinanceBenchLoader:
    """Loads and manages the FinanceBench dataset"""

    def __init__(self, dataset_name: str = "PatronusAI/financebench"):
        """
        Initialize the data loader

        Args:
            dataset_name: HuggingFace dataset identifier
        """
        self.dataset_name = dataset_name
        self.data = None
        self.df: Optional[pd.DataFrame] = None

    def load_data(self, split: str = "train") -> pd.DataFrame:
        """
        Load FinanceBench dataset from HuggingFace

        Args:
            split: Dataset split to load (default: "train")

        Returns:
            pandas DataFrame with questions, answers, evidence, and other columns
        """
        logger.info(f"Loading FinanceBench dataset: {self.dataset_name}, split: {split}")

        try:
            # Load dataset from HuggingFace
            self.data = load_dataset(self.dataset_name, split=split)

            # Prefer Dataset.to_pandas() (keeps native column types sensible).
            # Fall back to constructing a DataFrame directly if not available.
            if hasattr(self.data, "to_pandas"):
                try:
                    self.df = self.data.to_pandas()
                except Exception:
                    # Some datasets may expose to_pandas but it can still fail; fallback
                    self.df = pd.DataFrame(self.data)
            else:
                self.df = pd.DataFrame(self.data)

            # Ensure we have a proper DataFrame object
            if not isinstance(self.df, pd.DataFrame):
                self.df = pd.DataFrame(self.df)

            logger.info(f"Successfully loaded {len(self.df)} examples")
            logger.info(f"Columns available: {list(self.df.columns)}")

            # Log basic statistics (defensive: catch errors in statistics computation)
            try:
                self._log_dataset_statistics()
            except Exception as e:
                logger.exception("Failed while computing dataset statistics")

            return self.df

        except Exception:
            # Log full traceback to help diagnose ambiguous-truth errors
            logger.error("Error loading dataset: %s", traceback.format_exc())
            raise

    def _log_dataset_statistics(self):
        """Log comprehensive dataset statistics (defensive)."""
        if self.df is None:
            logger.warning("No data loaded yet")
            return

        logger.info("=" * 80)
        logger.info("DATASET STATISTICS")
        logger.info("=" * 80)

        # Basic info
        logger.info(f"Total examples: {len(self.df)}")
        logger.info(f"Columns: {list(self.df.columns)}")

        # Check for required columns
        required_columns = ["question", "answer", "evidence", "doc_name"]
        for col in required_columns:
            if col in self.df.columns:
                logger.info(f"✓ Column '{col}' present")
            else:
                logger.warning(f"✗ Column '{col}' missing")

        # Text length statistics — compute each inside try/except to avoid issues
        try:
            if "question" in self.df.columns:
                q_lengths = self.df["question"].astype(str).str.len()
                logger.info("\nQuestion lengths:")
                logger.info(f"  Mean: {q_lengths.mean():.2f} chars")
                logger.info(f"  Min: {q_lengths.min()} chars")
                logger.info(f"  Max: {q_lengths.max()} chars")
                logger.info(f"  Median: {q_lengths.median():.2f} chars")
        except Exception:
            logger.exception("Failed to compute question length statistics")

        try:
            if "answer" in self.df.columns:
                a_lengths = self.df["answer"].astype(str).str.len()
                logger.info("\nAnswer lengths:")
                logger.info(f"  Mean: {a_lengths.mean():.2f} chars")
                logger.info(f"  Min: {a_lengths.min()} chars")
                logger.info(f"  Max: {a_lengths.max()} chars")
                logger.info(f"  Median: {a_lengths.median():.2f} chars")
        except Exception:
            logger.exception("Failed to compute answer length statistics")

        try:
            if "evidence" in self.df.columns:
                # Convert entries to string safely then compute lengths.
                # Some entries can be array-like; avoid using pd.notna(x) in a boolean
                # context which raises "truth value of an array is ambiguous".
                import numpy as _np

                def _safe_len(x):
                    try:
                        if x is None:
                            return 0
                        # Treat common sequence/array-like types as non-missing
                        if isinstance(x, (list, tuple, _np.ndarray)):
                            return len(str(x))
                        # pd.isna handles pandas NA / float('nan') scalars
                        if pd.isna(x):
                            return 0
                        return len(str(x))
                    except Exception:
                        # Fallback: coerce to string and measure
                        try:
                            return len(str(x))
                        except Exception:
                            return 0

                e_lengths = self.df["evidence"].apply(_safe_len)
                logger.info("\nEvidence lengths:")
                logger.info(f"  Mean: {e_lengths.mean():.2f} chars")
                logger.info(f"  Min: {e_lengths.min()} chars")
                logger.info(f"  Max: {e_lengths.max()} chars")
                logger.info(f"  Median: {e_lengths.median():.2f} chars")
        except Exception:
            logger.exception("Failed to compute evidence length statistics")

        # Check for missing values
        try:
            logger.info("\nMissing values:")
            for col in self.df.columns:
                missing = int(self.df[col].isna().sum())
                if missing > 0:
                    logger.info(f"  {col}: {missing} ({missing/len(self.df)*100:.2f}%)")
        except Exception:
            logger.exception("Failed while computing missing-value statistics")

        logger.info("=" * 80)

    def get_sample(self, index: int = 0) -> Dict[str, Any]:
        """
        Get a single sample from the dataset

        Args:
            index: Index of the sample to retrieve

        Returns:
            Dictionary containing the sample data
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if index >= len(self.df):
            raise IndexError(f"Index {index} out of range. Dataset has {len(self.df)} examples.")

        sample = self.df.iloc[index].to_dict()

        logger.info(f"\nSample {index}:")
        logger.info(f"  Question: {str(sample.get('question',''))[:100]}...")
        logger.info(f"  Answer: {str(sample.get('answer',''))[:100]}...")
        logger.info(f"  Evidence length: {len(str(sample.get('evidence','')))} chars")

        return sample

    def get_batch(self, indices: List[int] = None, start: int = 0, end: int = None) -> List[Dict[str, Any]]:
        """
        Get a batch of samples

        Args:
            indices: Specific indices to retrieve (optional)
            start: Start index for range-based retrieval
            end: End index for range-based retrieval

        Returns:
            List of sample dictionaries
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if indices is not None:
            batch_df = self.df.iloc[indices]
        else:
            end = end or len(self.df)
            batch_df = self.df.iloc[start:end]

        logger.info(f"Retrieved batch of {len(batch_df)} examples")

        return batch_df.to_dict('records')

    def get_all_data(self) -> pd.DataFrame:
        """Get the entire dataset as DataFrame"""
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        return self.df

    def filter_by_doc(self, doc_name: str) -> pd.DataFrame:
        """
        Filter dataset by document name

        Args:
            doc_name: Name of the document to filter by

        Returns:
            Filtered DataFrame
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if 'doc_name' not in self.df.columns:
            logger.warning("'doc_name' column not found in dataset")
            return pd.DataFrame()

        filtered = self.df[self.df['doc_name'] == doc_name]
        logger.info(f"Filtered to {len(filtered)} examples for document: {doc_name}")

        return filtered


if __name__ == "__main__":
    # Basic local test (only run when executed directly)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    loader = FinanceBenchLoader()
    df = loader.load_data()
    sample = loader.get_sample(0)
    print("\nDATA LOADER TEST COMPLETE")