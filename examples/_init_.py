"""
Advanced Transformer Implementation in PyTorch.
"""

__version__ = "0.1.0"

from .config import (
    TransformerConfig,
    EncoderConfig,
    DecoderConfig,
    BERT_BASE_CONFIG,
    GPT2_BASE_CONFIG,
    T5_BASE_CONFIG,
)

from .models.transformer import Transformer
from .models.encoder import TransformerEncoder
from .models.decoder import TransformerDecoder

__all__ = [
    "Transformer",
    "TransformerEncoder",
    "TransformerDecoder",
    "TransformerConfig",
    "EncoderConfig",
    "DecoderConfig",
    "BERT_BASE_CONFIG",
    "GPT2_BASE_CONFIG",
    "T5_BASE_CONFIG",
]
