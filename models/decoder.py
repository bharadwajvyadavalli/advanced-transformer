"""
Transformer Decoder implementation.
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


class DecoderLayer(nn.Module):
    """
    Single layer of the transformer decoder.
    
    Each decoder layer consists of:
    1. Masked multi-head self-attention
    2. Add & Norm
    3. Multi-head cross-attention (to encoder outputs)
    4. Add & Norm
    5. Position-wise feed-forward network
    6. Add & Norm
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Self-attention layer (masked)
        self.self_attention = MultiHeadAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.attention_dropout,
        )
        
        # Cross-attention layer (for encoder-decoder attention)
        self.cross_attention = MultiHeadAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.attention_dropout,
        ) if config.is_encoder_decoder else None
        
        # Layer normalization
        self.norm1 = TransformerLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.norm2 = TransformerLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.norm3 = TransformerLayerNorm(config.d_model, eps=config.layer_norm_epsilon) if config.is_encoder_decoder else None
        
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
        
        # Flag to indicate if this is a decoder-only model or part of encoder-decoder
        self.is_decoder_only = not config.is_encoder_decoder
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor]]]:
        """
        Forward pass for the decoder layer.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, d_model]
            attention_mask: Mask for self-attention to prevent looking ahead [batch_size, seq_len, seq_len]
            encoder_hidden_states: Output of the encoder [batch_size, enc_seq_len, d_model]
            encoder_attention_mask: Mask for cross-attention [batch_size, seq_len, enc_seq_len]
            past_key_value: Cached key values for faster inference
            use_cache: Whether to use cached key values
            output_attentions: Whether to return attention weights
            
        Returns:
            hidden_states: Output tensor [batch_size, seq_len, d_model]
            self_attention_weights: Self-attention weights (if output_attentions=True)
            cross_attention_weights: Cross-attention weights (if output_attentions=True)
            past_key_value: Updated cache (if use_cache=True)
        """
        # Unpack the past key value if provided
        self_attn_past_key_value = past_key_value[0:2] if past_key_value is not None else None
        cross_attn_past_key_value = past_key_value[2:4] if past_key_value is not None and not self.is_decoder_only else None
        
        # Initialize output attention weights if needed
        self_attention_weights = None
        cross_attention_weights = None
        
        # Store residual for skip connection
        residual = hidden_states
        
        # Apply pre-norm if configured
        if self.use_pre_norm:
            hidden_states = self.norm1(hidden_states)
        
        # Self-attention (masked)
        attention_output = self.self_attention(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            attention_mask=attention_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            is_causal=True,  # Always use causal masking in decoder
        )
        
        # Unpack attention output
        if output_attentions:
            if use_cache:
                hidden_states, self_attention_weights, present_key_value = attention_output
            else:
                hidden_states, self_attention_weights = attention_output
        else:
            if use_cache:
                hidden_states, present_key_value = attention_output
            else:
                hidden_states = attention_output
        
        # Apply dropout and residual connection
        hidden_states = residual + self.dropout(hidden_states)
        
        # Apply post-norm if configured
        if not self.use_pre_norm:
            hidden_states = self.norm1(hidden_states)
        
        # Cross-attention to encoder outputs (only in encoder-decoder models)
        if self.cross_attention is not None and encoder_hidden_states is not None:
            residual = hidden_states
            
            # Apply pre-norm if configured
            if self.use_pre_norm:
                hidden_states = self.norm2(hidden_states)
            
            # Cross-attention
            attention_output = self.cross_attention(
                query=hidden_states,
                key=encoder_hidden_states,
                value=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                past_key_value=cross_attn_past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            
            # Unpack attention output
            if output_attentions:
                if use_cache:
                    hidden_states, cross_attention_weights, cross_present_key_value = attention_output
                else:
                    hidden_states, cross_attention_weights = attention_output
            else:
                if use_cache:
                    hidden_states, cross_present_key_value = attention_output
                else:
                    hidden_states = attention_output
            
            # Apply dropout and residual connection
            hidden_states = residual + self.dropout(hidden_states)
            
            # Apply post-norm if configured
            if not self.use_pre_norm:
                hidden_states = self.norm2(hidden_states)
        
        # Feed-forward network
        residual = hidden_states
        
        # Apply pre-norm if configured
        if self.use_pre_norm:
            norm_layer = self.norm3 if self.norm3 is not None else self.norm2
            hidden_states = norm_layer(hidden_states)
        
        # Apply feed-forward network
        hidden_states = self.feed_forward(hidden_states)
        
        # Apply dropout and residual connection
        hidden_states = residual + self.dropout(hidden_states)
        
        # Apply post-norm if configured
        if not self.use_pre_norm:
            norm_layer = self.norm3 if self.norm3 is not None else self.norm2
            hidden_states = norm_layer(hidden_states)
        
        # Prepare outputs
        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (self_attention_weights,)
            if cross_attention_weights is not None:
                outputs += (cross_attention_weights,)
        
        if use_cache:
            if self.is_decoder_only:
                # For decoder-only models, only cache self-attention
                past_key_value = (present_key_value,)
            else:
                # For encoder-decoder models, cache both self-attention and cross-attention
                past_key_value = (present_key_value, cross_present_key_value)
            
            outputs += (past_key_value,)
        
        return outputs


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder module.
    
    The decoder consists of:
    1. Input embeddings
    2. Positional encoding
    3. N identical decoder layers
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
        
        # Create decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(config) for _ in range(config.n_layers)
        ])
        
        # Final layer norm if using pre-norm
        self.final_layer_norm = TransformerLayerNorm(config.d_model, eps=config.layer_norm_epsilon) if config.use_pre_norm else None
        
        # Flag to indicate if this is a decoder-only model or part of encoder-decoder
        self.is_decoder_only = not config.is_encoder_decoder
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass for the decoder.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Mask to avoid attention on padding tokens [batch_size, seq_len]
            encoder_hidden_states: Output of the encoder [batch_size, enc_seq_len, d_model]
            encoder_attention_mask: Mask for encoder outputs [batch_size, enc_seq_len]
            inputs_embeds: Pre-computed input embeddings [batch_size, seq_len, d_model]
            past_key_values: Cached key values for faster inference
            use_cache: Whether to use cached key values
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states from all layers
            return_dict: Whether to return a dictionary or a tuple
            
        Returns:
            A tuple or dictionary containing:
            - last_hidden_state: Output of the last decoder layer [batch_size, seq_len, d_model]
            - past_key_values: Cache for key values (if use_cache=True)
            - hidden_states: Outputs of all decoder layers (if output_hidden_states=True)
            - attentions: Self-attention weights from all layers (if output_attentions=True)
            - cross_attentions: Cross-attention weights from all layers (if output_attentions=True)
        """
        # Get input embeddings - these come from the main model
        hidden_states = inputs_embeds
        
        # Check if using cached values
        use_cache = use_cache if use_cache is not None else self.config.use_cache if hasattr(self.config, "use_cache") else False
        
        # Get output shape
        batch_size, seq_length = hidden_states.size()[:2]
        
        # Apply positional encoding
        # For cached generation, only apply positional encoding to the new tokens
        if past_key_values is not None:
            past_length = past_key_values[0][0].size(2)  # key's sequence length
            position_start = past_length
            hidden_states = self.positional_encoding(hidden_states, position_start=position_start)
        else:
            hidden_states = self.positional_encoding(hidden_states)
        
        # Apply dropout
        hidden_states = self.dropout(hidden_states)
        
        # Prepare lists for outputs if needed
        all_hidden_states = [] if output_hidden_states else None
        all_self_attentions = [] if output_attentions else None
        all_cross_attentions = [] if output_attentions and not self.is_decoder_only else None
        
        # Initialize next cache
        next_decoder_cache = [] if use_cache else None
        
        # Create self-attention mask (causal masking)
        if attention_mask is not None:
            # Make sure the attention mask has the right shape
            # [batch_size, seq_len] -> [batch_size, 1, seq_len, seq_len]
            if past_key_values is not None:
                # For generation, we need to extend the attention mask
                past_length = past_key_values[0][0].size(2)  # key's sequence length
                
                # Create attention mask that covers the full sequence including past tokens
                attention_mask_full = torch.ones(
                    batch_size, seq_length + past_length, device=attention_mask.device
                )
                
                # Copy the provided mask for the new tokens
                attention_mask_full[:, -seq_length:] = attention_mask
                
                # Convert to causal mask
                causal_mask = torch.tril(
                    torch.ones(
                        (seq_length + past_length, seq_length + past_length),
                        device=attention_mask.device
                    )
                )
                
                # Combine causal and padding masks
                extended_attention_mask = attention_mask_full.unsqueeze(1).unsqueeze(2) * causal_mask.unsqueeze(0)
            else:
                # Standard case - just use causal masking
                # Convert to 4D: [batch_size, 1, seq_len, seq_len]
                extended_attention_mask = torch.tril(
                    torch.ones(
                        (batch_size, 1, seq_length, seq_length),
                        device=attention_mask.device
                    )
                )
                
                # Apply padding mask
                extended_attention_mask = extended_attention_mask * attention_mask.unsqueeze(1).unsqueeze(2)
            
            # Convert from 0/1 to -inf/0 for softmax
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            # If no attention mask is provided, just use causal masking
            extended_attention_mask = torch.tril(
                torch.ones(
                    (batch_size, 1, seq_length, seq_length),
                    device=hidden_states.device
                )
            )
            
            # Convert from 0/1 to -inf/0 for softmax
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Create encoder attention mask (for cross-attention)
        if encoder_attention_mask is not None and encoder_hidden_states is not None:
            # Make sure the attention mask has the right shape
            # [batch_size, enc_seq_len] -> [batch_size, 1, seq_len, enc_seq_len]
            extended_encoder_attention_mask = encoder_attention_mask.unsqueeze(1).unsqueeze(2)
            
            # Convert from 0/1 to -inf/0 for softmax
            extended_encoder_attention_mask = (1.0 - extended_encoder_attention_mask) * -10000.0
        else:
            extended_encoder_attention_mask = None
        
        # Process through decoder layers
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            
            # Get cached values for this layer
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=extended_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=extended_encoder_attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            
            # Unpack layer outputs
            if use_cache:
                if output_attentions:
                    if not self.is_decoder_only:
                        hidden_states, self_attn, cross_attn, present_key_value = layer_outputs
                        if all_cross_attentions is not None:
                            all_cross_attentions.append(cross_attn)
                    else:
                        hidden_states, self_attn, present_key_value = layer_outputs
                    
                    all_self_attentions.append(self_attn)
                    next_decoder_cache.append(present_key_value)
                else:
                    hidden_states, present_key_value = layer_outputs[:2]
                    next_decoder_cache.append(present_key_value)
            else:
                if output_attentions:
                    if not self.is_decoder_only:
                        hidden_states, self_attn, cross_attn = layer_outputs
                        if all_cross_attentions is not None:
                            all_cross_attentions.append(cross_attn)
                    else:
                        hidden_states, self_attn = layer_outputs
                    
                    all_self_attentions.append(self_attn)
                else:
                    hidden_states = layer_outputs[0]
        
        # Apply final layer norm if using pre-norm
        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)
        
        # Add the last hidden state to the list of hidden states
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        
        # Prepare outputs based on return_dict
        if not return_dict:
            outputs = (hidden_states,)
            
            if use_cache:
                outputs += (next_decoder_cache,)
            
            if output_hidden_states:
                outputs += (all_hidden_states,)
            
            if output_attentions:
                outputs += (all_self_attentions,)
                if all_cross_attentions is not None:
                    outputs += (all_cross_attentions,)
            
            return outputs
        
        return {
            "last_hidden_state": hidden_states,
            "past_key_values": next_decoder_cache if use_cache else None,
            "hidden_states": all_hidden_states if output_hidden_states else None,
            "attentions": all_self_attentions if output_attentions else None,
            "cross_attentions": all_cross_attentions if output_attentions and not self.is_decoder_only else None,
        }
