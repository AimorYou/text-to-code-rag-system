from pathlib import Path

from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class CodeSearchDataset(Dataset):
    """Dataset for code search task with text and code pairs"""

    def __init__(
        self,
        split: str,
        text_tokenizer: AutoTokenizer,
        code_tokenizer: AutoTokenizer,
        max_text_length: int = 128,
        max_code_length: int = 256,
        subset_size: int | None = None,
        data_path: Path | str | None = None,
    ):
        """Initialize dataset

        Args:
            split: Dataset split ('train', 'validation', 'test')
            text_tokenizer: Tokenizer for text descriptions
            code_tokenizer: Tokenizer for code snippets
            max_text_length: Maximum length for text tokens
            max_code_length: Maximum length for code tokens
            subset_size: If provided, limit dataset to this size
            data_path: Optional path to local data directory (DVC-managed)
        """
        self.split = split
        self.text_tokenizer = text_tokenizer
        self.code_tokenizer = code_tokenizer
        self.max_text_length = max_text_length
        self.max_code_length = max_code_length
        self.data_path = Path(data_path) if data_path else None

        self.data = self._load_data(subset_size)

    def _load_data(self, subset_size: int | None):
        """Load CodeSearchNet dataset from local path or HuggingFace.

        Args:
            subset_size: If provided, limit dataset to this number of samples

        Returns:
            Loaded dataset
        """
        if self.data_path and self.data_path.exists():
            split_path = self.data_path / self.split
            parquet_file = split_path / "data.parquet"

            if parquet_file.exists():
                print(f"Loading data from local path: {parquet_file}")
                dataset = load_dataset("parquet", data_files=str(parquet_file), split="train")
            else:
                print("Parquet file not found, loading from HuggingFace")
                dataset = load_dataset(
                    "espejelomar/code_search_net_python_10000_examples", split=self.split
                )
        else:
            print("Loading data from HuggingFace")
            dataset = load_dataset(
                "espejelomar/code_search_net_python_10000_examples", split=self.split
            )

        if subset_size is not None:
            dataset = dataset.select(range(min(subset_size, len(dataset))))

        return dataset

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            Dataset size
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Dictionary containing:
                - text_input_ids: Tokenized text input IDs
                - text_attention_mask: Attention mask for text
                - code_input_ids: Tokenized code input IDs
                - code_attention_mask: Attention mask for code
                - text: Original text description
                - code: Original code snippet
        """
        item = self.data[idx]

        text = item.get("func_documentation_string", "")
        code = item.get("func_code_string", "")

        text_encoded = self.text_tokenizer(
            text,
            max_length=self.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        code_encoded = self.code_tokenizer(
            code,
            max_length=self.max_code_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "text_input_ids": text_encoded["input_ids"].squeeze(0),
            "text_attention_mask": text_encoded["attention_mask"].squeeze(0),
            "code_input_ids": code_encoded["input_ids"].squeeze(0),
            "code_attention_mask": code_encoded["attention_mask"].squeeze(0),
            "text": text,
            "code": code,
        }
