"""
Configuration classes for the transformer models.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Union, Dict, Any


@dataclass
class TransformerConfig:
    """
    Configuration class for the Transformer model.
    
    This configuration class contains all the parameters needed to initialize a Transformer model.
    """
    # Model architecture
    vocab_size: int = 30000
    d_model: int = 512  # Hidden size
    n_heads: int = 8  # Number of attention heads
    n_layers: int = 6  # Number of layers
    d_ff: int = 2048  # Feed-forward hidden size
    max_seq_length: int = 512  # Maximum sequence length
    
    # Dropout probabilities
    dropout: float = 0.1  # General dropout
    attention_dropout: float = 0.1  # Dropout in attention
    hidden_dropout: float = 0.1  # Dropout in feed-forward networks
    
    # Positional encoding
    positional_encoding_type: str = "sinusoidal"  # Options: "sinusoidal", "learned", "alibi", "rotary"
    
    # Activation function
    activation_function: str = "gelu"  # Options: "gelu", "relu", "swiglu"
    
    # Layer normalization
    layer_norm_epsilon: float = 1e-5
    use_pre_norm: bool = True  # Whether to use pre-layer normalization
    
    # Initialization
    initializer_range: float = 0.02
    
    # Other options
    tie_word_embeddings: bool = True  # Whether to tie input and output embeddings
    padding_idx: int = 0  # Token ID for padding
    
    # Task-specific parameters
    is_encoder_decoder: bool = True  # Whether to use encoder-decoder architecture
    is_decoder: bool = False  # Whether this is a decoder-only model
    
    def __post_init__(self):
        """Post-initialization validation and derived parameters."""
        assert self.d_model % self.n_heads == 0, f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        
        # Derived parameter: head dimension
        self.d_head = self.d_model // self.n_heads
        
        # Validate activation function
        valid_activations = ["gelu", "relu", "swiglu"]
        assert self.activation_function in valid_activations, f"activation_function must be one of {valid_activations}"
        
        # Validate positional encoding type
        valid_pos_encodings = ["sinusoidal", "learned", "alibi", "rotary"]
        assert self.positional_encoding_type in valid_pos_encodings, f"positional_encoding_type must be one of {valid_pos_encodings}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TransformerConfig":
        """Create a config from a dictionary."""
        return cls(**config_dict)


@dataclass
class EncoderConfig(TransformerConfig):
    """Configuration for transformer encoder."""
    is_encoder_decoder: bool = False  # Override parent
    is_decoder: bool = False  # Explicitly mark as encoder
    

@dataclass
class DecoderConfig(TransformerConfig):
    """Configuration for transformer decoder."""
    is_encoder_decoder: bool = False  # Override parent
    is_decoder: bool = True  # Explicitly mark as decoder
    use_cache: bool = True  # Whether to use key-value caching during inference
    
    
# Preset configurations for common architectures
BERT_BASE_CONFIG = TransformerConfig(
    vocab_size=30522,
    d_model=768,
    n_heads=12,
    n_layers=12,
    d_ff=3072,
    max_seq_length=512,
    positional_encoding_type="learned",
    is_encoder_decoder=False,
    is_decoder=False,
)

GPT2_BASE_CONFIG = TransformerConfig(
    vocab_size=50257,
    d_model=768,
    n_heads=12,
    n_layers=12,
    d_ff=3072,
    max_seq_length=1024,
    positional_encoding_type="learned",
    is_encoder_decoder=False,
    is_decoder=True,
)

T5_BASE_CONFIG = TransformerConfig(
    vocab_size=32128,
    d_model=768,
    n_heads=12,
    n_layers=12,
    d_ff=3072,
    max_seq_length=512,
    positional_encoding_type="alibi",  # T5 uses relative position biases
    is_encoder_decoder=True,
)
