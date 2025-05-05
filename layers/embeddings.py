"""
Embedding layers for transformer models.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, Union, List


class TokenEmbedding(nn.Module):
    """
    Token embedding layer for transformers.
    
    This embeds token IDs into a continuous vector space.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.d_model = d_model
    
    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        Forward pass for token embedding.
        
        Args:
            x: Input tensor of token IDs [batch_size, seq_len]
            
        Returns:
            embeddings: Embedded tokens [batch_size, seq_len, d_model]
        """
        # Scale embeddings by sqrt(d_model) following the transformer paper
        return self.embedding(x) * math.sqrt(self.d_model)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformers.
    
    This implements the positional encoding described in "Attention Is All You Need":
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(
        self,
        d_model: int,
        max_seq_length: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        # [max_seq_length, d_model]
        pe = torch.zeros(max_seq_length, d_model)
        
        # Create position vector
        # [max_seq_length, 1]
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        
        # Create division term
        # [1, d_model/2]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension [1, max_seq_length, d_model]
        pe = pe.unsqueeze(0)
        
        # Register as a buffer (not a parameter)
        self.register_buffer("pe", pe, persistent=False)
    
    def forward(self, x: torch.Tensor, position_start: int = 0) -> torch.Tensor:
        """
        Forward pass for positional encoding.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            position_start: Starting position for positional encoding (for incremental decoding)
            
        Returns:
            x: Input with positional encoding added [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        
        # Get the appropriate slice of the positional encoding
        positional_encoding = self.pe[:, position_start:position_start + seq_len, :]
        
        # Add positional encoding to input
        x = x + positional_encoding.to(x.device)
        
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional encoding for transformers.
    
    This implements a trainable positional encoding, where the position embeddings
    are learned parameters rather than fixed values.
    """
    
    def __init__(
        self,
        d_model: int,
        max_seq_length: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create learned position embedding
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        
        # Initialize with sinusoidal encoding for better convergence
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_seq_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Initialize embedding weights
        self.position_embedding.weight.data.copy_(pe)
    
    def forward(self, x: torch.Tensor, position_start: int = 0) -> torch.Tensor:
        """
        Forward pass for learned positional encoding.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            position_start: Starting position for positional encoding (for incremental decoding)
            
        Returns:
            x: Input with positional encoding added [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        
        # Create position indices
        positions = torch.arange(
            position_start, 
            position_start + seq_len,
            device=x.device
        )
        
        # Get position embeddings
        positional_encoding = self.position_embedding(positions).unsqueeze(0)
        
        # Add positional encoding to input
        x = x + positional_encoding
        
        return self.dropout(x)


class ALiBiPositionalEncoding(nn.Module):
    """
    ALiBi (Attention with Linear Biases) positional encoding.
    
    This implements the ALiBi approach described in 
    "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
    
    Note: ALiBi doesn't actually add position embeddings to tokens, but rather modifies
    the attention matrix directly. This dummy module exists for compatibility with other
    positional encoding methods in the architecture.
    """
    
    def __init__(
        self,
        d_model: int,
        max_seq_length: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor, position_start: int = 0) -> torch.Tensor:
        """
        Forward pass for ALiBi positional encoding.
        
        ALiBi doesn't add positional encodings to the embeddings, but applies
        position-dependent biases to the attention scores. This method just applies
        dropout to maintain interface compatibility.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            position_start: Starting position (unused, for API compatibility)
            
        Returns:
            x: Input with dropout applied [batch_size, seq_len, d_model]
        """
        return self.dropout(x)


class RotaryPositionalEncoding(nn.Module):
    """
    Rotary positional encoding for transformers.
    
    This implements the rotary position embeddings as described in 
    "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    
    Note: RoPE doesn't actually add position embeddings to tokens, but rather modifies
    the query and key vectors directly during attention. This dummy module exists for 
    compatibility with other positional encoding methods in the architecture.
    """
    
    def __init__(
        self,
        d_model: int,
        max_seq_length: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor, position_start: int = 0) -> torch.Tensor:
        """
        Forward pass for rotary positional encoding.
        
        RoPE doesn't add positional encodings to the embeddings, but applies
        rotations to the query and key vectors in the attention layer. This method 
        just applies dropout to maintain interface compatibility.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            position_start: Starting position (unused, for API compatibility)
            
        Returns:
            x: Input with dropout applied [batch_size, seq_len, d_model]
        """
        return self.dropout(x)


class PositionalEncoding(nn.Module):
    """
    Factory class for various positional encoding strategies.
    """
    
    def __init__(
        self,
        d_model: int,
        max_seq_length: int = 5000,
        dropout: float = 0.1,
        encoding_type: str = "sinusoidal",
    ):
        super().__init__()
        
        # Create the appropriate positional encoding based on type
        if encoding_type == "sinusoidal":
            self.encoding = SinusoidalPositionalEncoding(
                d_model, max_seq_length, dropout
            )
        elif encoding_type == "learned":
            self.encoding = LearnedPositionalEncoding(
                d_model, max_seq_length, dropout
            )
        elif encoding_type == "alibi":
            self.encoding = ALiBiPositionalEncoding(
                d_model, max_seq_length, dropout
            )
        elif encoding_type == "rotary":
            self.encoding = RotaryPositionalEncoding(
                d_model, max_seq_length, dropout
            )
        else:
            raise ValueError(f"Unsupported positional encoding type: {encoding_type}")
    
    def forward(self, x: torch.Tensor, position_start: int = 0) -> torch.Tensor:
        """
        Forward pass for positional encoding.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            position_start: Starting position (for incremental decoding)
            
        Returns:
            x: Input with positional encoding applied [batch_size, seq_len, d_model]
        """
        return self.encoding(x, position_start)
