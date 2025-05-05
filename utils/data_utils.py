"""
Data utilities for transformer models.
"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Optional, Dict, Any, Union, List, Tuple, Callable, Iterator


class TransformerDataset(Dataset):
    """
    Dataset wrapper for transformer models.
    """
    
    def __init__(
        self,
        input_ids: List[List[int]],
        attention_mask: Optional[List[List[int]]] = None,
        labels: Optional[List[int]] = None,
        return_tensors: bool = True,
    ):
        """
        Initialize dataset.
        
        Args:
            input_ids: List of tokenized inputs
            attention_mask: List of attention masks
            labels: List of labels
            return_tensors: Whether to convert to tensors
        """
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.return_tensors = return_tensors
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.input_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, List[int], int]]:
        """
        Get dataset item.
        
        Args:
            idx: Index
            
        Returns:
            Dictionary of items
        """
        item = {"input_ids": self.input_ids[idx]}
        
        if self.attention_mask is not None:
            item["attention_mask"] = self.attention_mask[idx]
        
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        
        # Convert to tensors if requested
        if self.return_tensors:
            item = {k: torch.tensor(v) for k, v in item.items()}
        
        return item


class SequenceDataset(Dataset):
    """
    Dataset for sequence-to-sequence transformer models.
    """
    
    def __init__(
        self,
        input_ids: List[List[int]],
        decoder_input_ids: List[List[int]],
        attention_mask: Optional[List[List[int]]] = None,
        decoder_attention_mask: Optional[List[List[int]]] = None,
        labels: Optional[List[List[int]]] = None,
        return_tensors: bool = True,
    ):
        """
        Initialize dataset.
        
        Args:
            input_ids: List of tokenized encoder inputs
            decoder_input_ids: List of tokenized decoder inputs
            attention_mask: List of encoder attention masks
            decoder_attention_mask: List of decoder attention masks
            labels: List of labels (often target sequences)
            return_tensors: Whether to convert to tensors
        """
        self.input_ids = input_ids
        self.decoder_input_ids = decoder_input_ids
        self.attention_mask = attention_mask
        self.decoder_attention_mask = decoder_attention_mask
        self.labels = labels
        self.return_tensors = return_tensors
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.input_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, List[int]]]:
        """
        Get dataset item.
        
        Args:
            idx: Index
            
        Returns:
            Dictionary of items
        """
        item = {
            "input_ids": self.input_ids[idx],
            "decoder_input_ids": self.decoder_input_ids[idx],
        }
        
        if self.attention_mask is not None:
            item["attention_mask"] = self.attention_mask[idx]
        
        if self.decoder_attention_mask is not None:
            item["decoder_attention_mask"] = self.decoder_attention_mask[idx]
        
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        
        # Convert to tensors if requested
        if self.return_tensors:
            item = {k: torch.tensor(v) for k, v in item.items()}
        
        return item


def create_tokenized_dataset(
    texts: List[str],
    tokenizer: Any,
    max_length: int = 512,
    padding: str = "max_length",
    truncation: bool = True,
    labels: Optional[List[Union[int, List[int]]]] = None,
    is_decoder_input: bool = False,
    return_tensors: bool = True,
) -> Dataset:
    """
    Create a tokenized dataset from texts.
    
    Args:
        texts: List of text inputs
        tokenizer: Tokenizer
        max_length: Maximum sequence length
        padding: Padding strategy
        truncation: Whether to truncate sequences
        labels: List of labels
        is_decoder_input: Whether this is for decoder inputs
        return_tensors: Whether to convert to tensors
        
    Returns:
        Dataset
    """
    # Tokenize inputs
    tokenized = tokenizer(
        texts,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors=None,  # Return lists, not tensors
    )
    
    # For decoder inputs in seq2seq, we might need to prepare differently
    if is_decoder_input:
        # Create decoder_input_ids by shifting labels right
        # This is a common practice for transformer decoder inputs
        if labels is not None:
            if isinstance(labels[0], list):
                # Labels are already tokenized
                decoder_input_ids = [[tokenizer.pad_token_id] + l[:-1] for l in labels]
            else:
                # Tokenize labels for decoder inputs
                tokenized_labels = tokenizer(
                    labels,
                    max_length=max_length,
                    padding=padding,
                    truncation=truncation,
                    return_tensors=None,
                )
                decoder_input_ids = [[tokenizer.pad_token_id] + ids[:-1] 
                                      for ids in tokenized_labels["input_ids"]]
            
            return SequenceDataset(
                input_ids=tokenized["input_ids"],
                decoder_input_ids=decoder_input_ids,
                attention_mask=tokenized["attention_mask"],
                decoder_attention_mask=None,  # Create this based on decoder_input_ids if needed
                labels=labels,
                return_tensors=return_tensors,
            )
    
    # For standard transformer models
    return TransformerDataset(
        input_ids=tokenized["input_ids"],
        attention_mask=tokenized["attention_mask"],
        labels=labels,
        return_tensors=return_tensors,
    )


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    collate_fn: Optional[Callable] = None,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader from a Dataset.
    
    Args:
        dataset: Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle the dataset
        collate_fn: Function to collate data
        num_workers: Number of workers for data loading
        
    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def get_tokenizer(
    model_name_or_path: str,
    cache_dir: Optional[str] = None,
) -> Any:
    """
    Get a tokenizer from a model name or path.
    
    Args:
        model_name_or_path: Model name or path
        cache_dir: Cache directory
        
    Returns:
        Tokenizer
    """
    try:
        # Try to import tokenizers from transformers
        from transformers import AutoTokenizer
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
        )
        
        return tokenizer
    
    except ImportError:
        # If transformers is not available, return a message
        raise ImportError(
            "Could not import AutoTokenizer from transformers. "
            "Please install with: pip install transformers"
        )


def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split a dataset into train, validation, and test sets.
    
    Args:
        dataset: Dataset to split
        train_ratio: Ratio of data to use for training
        val_ratio: Ratio of data to use for validation
        test_ratio: Ratio of data to use for testing
        seed: Random seed
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Check that ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    # Set the seed for reproducibility
    torch.manual_seed(seed)
    
    # Calculate sizes
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )
    
    return train_dataset, val_dataset, test_dataset
