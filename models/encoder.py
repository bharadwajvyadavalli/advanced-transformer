"""
Transformer Encoder implementation.
"""
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Dict, Any, Union, List

from ..config import TransformerConfig
from ..layers.attention import MultiHeadAttention
from ..layers.feed_forward import PositionWiseFFN
from ..layers.embeddings import PositionalEncoding
from ..layers.normalization import TransformerLayerNorm


class EncoderLayer(nn.Module):
    """
    Single layer of the transformer encoder.
    
    Each encoder layer consists of:
    1. Multi-head self-attention
    2. Add & Norm
    3. Position-wise feed-forward network
    4. Add & Norm
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Self-attention layer
        self.self_attention = MultiHeadAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.attention_dropout,
        )
        
        # Layer normalization
        self.norm1 = TransformerLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.norm2 = TransformerLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        
        # Feed-forward network
        self.feed_forward = PositionWiseFFN(
            d_model=config.d_model,
            d_ff=config.d_ff,
            dropout=config.hidden_dropout,
            activation=config.activation_function,
        )
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(config.dropout)
        
        # Whether to use pre-normalization or post-normalization
        self.use_pre_norm = config.use_pre_norm
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for the encoder layer.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, d_model]
            attention_mask: Mask to avoid attention on padding tokens [batch_size, seq_len, seq_len]
            output_attentions: Whether to return attention weights
            
        Returns:
            hidden_states: Output tensor [batch_size, seq_len, d_model]
            attention_weights: Attention weights [batch_size, n_heads, seq_len, seq_len] (if output_attentions=True)
        """
        # Store residual for skip connection
        residual = hidden_states
        
        # Apply pre-norm if configured
        if self.use_pre_norm:
            hidden_states = self.norm1(hidden_states)
        
        # Self-attention
        attention_output = self.self_attention(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        
        # Unpack attention output
        if output_attentions:
            hidden_states, attention_weights = attention_output
        else:
            hidden_states = attention_output
        
        # Apply dropout and residual connection
        hidden_states = residual + self.dropout(hidden_states)
        
        # Apply post-norm if configured
        if not self.use_pre_norm:
            hidden_states = self.norm1(hidden_states)
        
        # Feed-forward network
        residual = hidden_states
        
        # Apply pre-norm if configured
        if self.use_pre_norm:
            hidden_states = self.norm2(hidden_states)
        
        # Apply feed-forward network
        hidden_states = self.feed_forward(hidden_states)
        
        # Apply dropout and residual connection
        hidden_states = residual + self.dropout(hidden_states)
        
        # Apply post-norm if configured
        if not self.use_pre_norm:
            hidden_states = self.norm2(hidden_states)
        
        if output_attentions:
            return hidden_states, attention_weights
        else:
            return hidden_states


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder module.
    
    The encoder consists of:
    1. Input embeddings
    2. Positional encoding
    3. N identical encoder layers
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings are defined in the main Transformer model
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model=config.d_model,
            max_seq_length=config.max_seq_length,
            dropout=config.dropout,
            encoding_type=config.positional_encoding_type,
        )
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Create encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(config) for _ in range(config.n_layers)
        ])
        
        # Final layer norm if using pre-norm
        self.final_layer_norm = TransformerLayerNorm(config.d_model, eps=config.layer_norm_epsilon) if config.use_pre_norm else None
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass for the encoder.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Mask to avoid attention on padding tokens [batch_size, seq_len]
            inputs_embeds: Pre-computed input embeddings [batch_size, seq_len, d_model]
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states from all layers
            return_dict: Whether to return a dictionary or a tuple
            
        Returns:
            A tuple or dictionary containing:
            - last_hidden_state: Output of the last encoder layer [batch_size, seq_len, d_model]
            - hidden_states: Outputs of all encoder layers (if output_hidden_states=True)
            - attentions: Attention weights from all layers (if output_attentions=True)
        """
        # Get input embeddings - these come from the main model
        hidden_states = inputs_embeds
        
        # Apply positional encoding
        hidden_states = self.positional_encoding(hidden_states)
        
        # Apply dropout
        hidden_states = self.dropout(hidden_states)
        
        # Prepare lists for outputs if needed
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        
        # Create attention mask for multi-head attention
        # Convert from [batch_size, seq_len] to [batch_size, 1, 1, seq_len]
        if attention_mask is not None:
            # Make sure the attention mask has the right shape
            # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            
            # Convert from 0/1 to -inf/0 for softmax
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None
        
        # Process through encoder layers
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            
            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=extended_attention_mask,
                output_attentions=output_attentions,
            )
            
            if output_attentions:
                hidden_states, attention_weights = layer_outputs
                all_attentions.append(attention_weights)
            else:
                hidden_states = layer_outputs
        
        # Apply final layer norm if using pre-norm
        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)
        
        # Add the last hidden state to the list of hidden states
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        
        # Prepare outputs based on return_dict
        if not return_dict:
            outputs = (hidden_states,)
            if output_hidden_states:
                outputs += (all_hidden_states,)
            if output_attentions:
                outputs += (all_attentions,)
            return outputs
        
        return {
            "last_hidden_state": hidden_states,
            "hidden_states": all_hidden_states if output_hidden_states else None,
            "attentions": all_attentions if output_attentions else None,
        }
