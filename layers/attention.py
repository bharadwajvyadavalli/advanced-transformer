"""
Multi-head attention implementations for transformers.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, Union, List


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.
    
    This implements the core attention function:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    """
    
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        is_causal: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for scaled dot-product attention.
        
        Args:
            query: Query tensor [batch_size, n_heads, seq_len_q, d_k]
            key: Key tensor [batch_size, n_heads, seq_len_k, d_k]
            value: Value tensor [batch_size, n_heads, seq_len_k, d_v]
            attention_mask: Mask tensor [batch_size, n_heads, seq_len_q, seq_len_k]
            output_attentions: Whether to return attention weights
            is_causal: Whether to use causal masking (for decoder self-attention)
            
        Returns:
            attention_output: Attention output [batch_size, n_heads, seq_len_q, d_v]
            attention_weights: Attention weights [batch_size, n_heads, seq_len_q, seq_len_k]
                (only if output_attentions=True)
        """
        # Get dimensions
        d_k = query.size(-1)
        
        # Compute scaled dot-product attention
        # [batch_size, n_heads, seq_len_q, d_k] x [batch_size, n_heads, d_k, seq_len_k]
        # -> [batch_size, n_heads, seq_len_q, seq_len_k]
        # Note: transposing key for the matrix multiplication
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Apply causal masking if requested (for decoder self-attention)
        if is_causal:
            seq_len_q, seq_len_k = query.size(-2), key.size(-2)
            causal_mask = torch.triu(
                torch.ones(seq_len_q, seq_len_k, device=query.device),
                diagonal=1
            ).bool()
            attention_scores.masked_fill_(causal_mask[None, None, :, :], float("-inf"))
        
        # Compute attention weights using softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply dropout
        attention_weights = self.dropout(attention_weights)
        
        # Compute attention output by weighted sum of values
        # [batch_size, n_heads, seq_len_q, seq_len_k] x [batch_size, n_heads, seq_len_k, d_v]
        # -> [batch_size, n_heads, seq_len_q, d_v]
        attention_output = torch.matmul(attention_weights, value)
        
        if output_attentions:
            return attention_output, attention_weights
        else:
            return attention_output


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module for transformers.
    
    This implements the multi-head attention mechanism:
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_o
    where head_i = Attention(QW_q_i, KW_k_i, VW_v_i)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        
        # Check that model dimensions are divisible by number of heads
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        # Save parameters
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension of each head
        self.d_v = d_model // n_heads  # Dimension of value in each head
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        
        # Use a single matrix for QKV projection when query, key, and value are the same
        # This is an optimization for self-attention
        self.use_packed_qkv_projection = True
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        
        # Final output projection
        self.output_proj = nn.Linear(d_model, d_model, bias=bias)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout=dropout)
        
        # Dropout for output
        self.dropout = nn.Dropout(p=dropout)
        
        # Flag to optimize with torch.nn.functional.scaled_dot_product_attention when available
        self.use_sdpa = hasattr(F, 'scaled_dot_product_attention')
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        is_causal: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor]]]:
        """
        Forward pass for multi-head attention.
        
        Args:
            query: Query tensor [batch_size, seq_len_q, d_model]
            key: Key tensor [batch_size, seq_len_k, d_model]
            value: Value tensor [batch_size, seq_len_k, d_model]
            attention_mask: Mask tensor [batch_size, seq_len_q, seq_len_k] or [batch_size, 1, seq_len_q, seq_len_k]
            past_key_value: Cached key and value tensors for incremental decoding
            use_cache: Whether to return key and value tensors for caching
            output_attentions: Whether to return attention weights
            is_causal: Whether to use causal masking (for decoder self-attention)
            
        Returns:
            attention_output: Attention output [batch_size, seq_len_q, d_model]
            attention_weights: Attention weights (only if output_attentions=True)
            past_key_value: Cached key and value tensors (only if use_cache=True)
        """
        batch_size, seq_len_q = query.size(0), query.size(1)
        
        # Handle cached past key value
        if past_key_value is not None:
            # Retrieve cached keys and values
            key_states, value_states = past_key_value
            seq_len_k = key_states.size(2) + key.size(1)  # Past + current seq length
        else:
            # No cached values
            seq_len_k = key.size(1)
            key_states = None
            value_states = None
        
        # Optimize: use packed QKV projection for self-attention
        if self.use_packed_qkv_projection and query is key and key is value and past_key_value is None:
            # [batch_size, seq_len, d_model] -> [batch_size, seq_len, 3 * d_model]
            qkv = self.qkv_proj(query)
            
            # Split QKV
            # [batch_size, seq_len, 3 * d_model] -> 3 x [batch_size, seq_len, d_model]
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            # Separate projections for Q, K, V
            # [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
            q = self.q_proj(query)
            
            if key_states is None:
                k = self.k_proj(key)
                v = self.v_proj(value)
            else:
                # During incremental decoding, only project new keys and values
                k_new = self.k_proj(key)
                v_new = self.v_proj(value)
                
                # Concatenate with past keys and values
                k = torch.cat([key_states, k_new], dim=2)
                v = torch.cat([value_states, v_new], dim=2)
        
        # Reshape for multi-head attention
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, n_heads, d_k] -> [batch_size, n_heads, seq_len, d_k]
        q = q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        
        # Prepare attention mask if provided
        if attention_mask is not None:
            # Broadcast to the multi-head shape
            # If mask is [batch_size, seq_len_q, seq_len_k], reshape to [batch_size, 1, seq_len_q, seq_len_k]
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
                
            # Make sure the mask has the correct shape
            assert attention_mask.size(0) == batch_size, "Attention mask batch size must match input batch size"
            assert attention_mask.size(2) == seq_len_q, "Attention mask seq_len_q must match input seq_len_q"
            assert attention_mask.size(3) == seq_len_k, "Attention mask seq_len_k must match input seq_len_k"
        
        # Compute attention
        if self.use_sdpa:
            # Use PyTorch's built-in scaled_dot_product_attention
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=is_causal and past_key_value is None
            )
            
            # Get attention weights if needed
            if output_attentions:
                # We need to recompute attention weights since SDPA doesn't return them
                with torch.no_grad():
                    attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
                    if attention_mask is not None:
                        attn_weights = attn_weights + attention_mask
                    if is_causal and past_key_value is None:
                        causal_mask = torch.triu(
                            torch.ones(seq_len_q, seq_len_k, device=q.device),
                            diagonal=1
                        ).bool()
                        attn_weights.masked_fill_(causal_mask[None, None, :, :], float("-inf"))
                    attn_weights = F.softmax(attn_weights, dim=-1)
            else:
                attn_weights = None
        else:
            # Use our implementation of scaled dot-product attention
            attention_output = self.attention(
                q, k, v,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                is_causal=is_causal,
            )
            
            # Unpack attention output
            if output_attentions:
                attn_output, attn_weights = attention_output
            else:
                attn_output = attention_output
                attn_weights = None
        
        # Reshape attention output
        # [batch_size, n_heads, seq_len_q, d_v] -> [batch_size, seq_len_q, n_heads, d_v] -> [batch_size, seq_len_q, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final projection
        attention_output = self.output_proj(attn_output)
        
        # Apply dropout
        attention_output = self.dropout(attention_output)
        
        # Prepare outputs
        outputs = (attention_output,)
        
        if output_attentions:
            outputs += (attn_weights,)
        
        if use_cache:
            # For incremental decoding, cache the key and value states
            past_key_value = (k, v)
            outputs += (past_key_value,)
        
        # Return appropriate outputs based on flags
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary positional embeddings for transformers.
    
    This implements the rotary position embeddings as described in "RoFormer: Enhanced Transformer
    with Rotary Position Embedding" https://arxiv.org/abs/2104.09864
    """
    
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Create rotation matrix cache
        self.register_buffer(
            "cos_cached", 
            self._get_cos_cached(max_seq_len, dim),
            persistent=False
        )
        self.register_buffer(
            "sin_cached", 
            self._get_sin_cached(max_seq_len, dim),
            persistent=False
        )
    
    def _get_cos_cached(self, seq_len: int, dim: int) -> torch.Tensor:
        """Precompute cos values for rotary embeddings."""
        # Each dimension of the query and key gets a different frequency
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        
        # Create position sequence
        t = torch.arange(seq_len, dtype=inv_freq.dtype)
        
        # Compute all required cos values
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = torch.cos(freqs)
        
        return cos
    
    def _get_sin_cached(self, seq_len: int, dim: int) -> torch.Tensor:
        """Precompute sin values for rotary embeddings."""
        # Each dimension of the query and key gets a different frequency
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        
        # Create position sequence
        t = torch.arange(seq_len, dtype=inv_freq.dtype)
        
        # Compute all required sin values
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        sin = torch.sin(freqs)
        
        return sin
    
    def forward(self, x: torch.Tensor, seq_dim: int = 2, offset: int = 0) -> torch.Tensor:
        """
        Apply rotary position embeddings to input tensor.
        
        Args:
            x: Input tensor [batch_size, n_heads, seq_len, d_head]
            seq_dim: Dimension corresponding to sequence length
            offset: Position offset (for incremental decoding)
            
        Returns:
            x_rotated: Tensor with rotary position embeddings applied
        """
        seq_len = x.size(seq_dim)
        
        # Ensure the sequence length is not too long
        assert seq_len <= self.max_seq_len, f"Sequence length ({seq_len}) exceeds maximum ({self.max_seq_len})"
        
        # Take the required parts of the precomputed sin and cos
        cos = self.cos_cached[offset:offset + seq_len]
        sin = self.sin_cached[offset:offset + seq_len]
        
        # Adapt the shape of sin and cos to the shape of x
        if seq_dim == 1:
            # For example [batch_size, seq_len, n_heads, d_head]
            cos = cos.unsqueeze(0).unsqueeze(2)
            sin = sin.unsqueeze(0).unsqueeze(2)
        else:
            # For example [batch_size, n_heads, seq_len, d_head]
            cos = cos.unsqueeze(0).unsqueeze(1)
            sin = sin.unsqueeze(0).unsqueeze(1)
        
        # Split embedding dimension into pairs
        x_pairs = x.view(*x.shape[:-1], -1, 2)
        
        # Apply rotation by complex multiplication
        # (a + ib) * (c + id) = (ac - bd) + i(ad + bc)
        # where a, b are the original query/key vectors and c, d are cos, sin
        x1, x2 = x_pairs[..., 0], x_pairs[..., 1]
        
        # Rotate each pair (dim=-1 are pairs)
        x_rotated = torch.stack(
            [
                x1 * cos - x2 * sin,  # Real part
                x2 * cos + x1 * sin,  # Imaginary part
            ],
            dim=-1
        )
        
        # Return to original shape
        return x_rotated.flatten(-2)


class ALiBiAttention(nn.Module):
    """
    Attention with Linear Biases (ALiBi) for transformers.
    
    This implements the ALiBi attention mechanism as described in
    "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
    https://arxiv.org/abs/2108.12409
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        
        # Check that model dimensions are divisible by number of heads
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        # Save parameters
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension of each head
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        
        # Final output projection
        self.output_proj = nn.Linear(d_model, d_model, bias=bias)
        
        # Dropout for output
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute ALiBi slopes
        self.register_buffer("slopes", self._get_slopes(n_heads))
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout=dropout)
    
    def _get_slopes(self, n_heads: int) -> torch.Tensor:
        """
        Compute ALiBi slopes for each attention head.
        
        Args:
            n_heads: Number of attention heads
            
        Returns:
            slopes: Tensor of shape [n_heads]
        """
        # Compute the closest power of 2 greater than or equal to n_heads
        power = 2 ** math.ceil(math.log2(n_heads))
        
        # For ALiBi, the slopes follow a geometric sequence
        # For power-of-2 n_heads, this is simple
        if n_heads == power:
            return torch.tensor([2 ** (-8 + i) for i in range(n_heads)])
        
        # For non-power-of-2 n_heads, we need to interpolate
        slopes = torch.tensor([2 ** (-8 + i) for i in range(power)])
        
        # Keep only the slopes corresponding to the n_heads we need
        ratio = power // n_heads
        slopes = slopes[torch.arange(0, power, ratio)][:n_heads]
        
        return slopes
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        is_causal: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for ALiBi attention.
        
        Args:
            query: Query tensor [batch_size, seq_len_q, d_model]
            key: Key tensor [batch_size, seq_len_k, d_model]
            value: Value tensor [batch_size, seq_len_k, d_model]
            attention_mask: Mask tensor [batch_size, seq_len_q, seq_len_k] or [batch_size, 1, seq_len_q, seq_len_k]
            output_attentions: Whether to return attention weights
            is_causal: Whether to use causal masking (for decoder self-attention)
            
        Returns:
            attention_output: Attention output [batch_size, seq_len_q, d_model]
            attention_weights: Attention weights (only if output_attentions=True)
        """
        batch_size, seq_len_q, _ = query.size()
        _, seq_len_k, _ = key.size()
        
        # Project Q, K, V
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, n_heads, d_k] -> [batch_size, n_heads, seq_len, d_k]
        q = q.view(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute scaled dot-product attention
        # [batch_size, n_heads, seq_len_q, d_k] x [batch_size, n_heads, d_k, seq_len_k]
        # -> [batch_size, n_heads, seq_len_q, seq_len_k]
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply ALiBi bias
        # Create position bias matrix
        # We use a simple outer product to compute all pairwise distances
        positions_q = torch.arange(seq_len_q, device=q.device).unsqueeze(1)
        positions_k = torch.arange(seq_len_k, device=k.device).unsqueeze(0)
        distance = positions_q - positions_k  # [seq_len_q, seq_len_k]
        
        # For causal attention, we only want negative distances
        if is_causal:
            distance = distance.masked_fill(distance > 0, 0)
        
        # Apply slopes for each head
        # [n_heads, 1, 1] * [1, seq_len_q, seq_len_k] -> [n_heads, seq_len_q, seq_len_k]
        alibi_bias = self.slopes.view(-1, 1, 1) * distance.unsqueeze(0)
        
        # Add ALiBi bias to attention scores
        # [batch_size, n_heads, seq_len_q, seq_len_k] + [n_heads, seq_len_q, seq_len_k]
        attention_scores = attention_scores + alibi_bias.unsqueeze(0)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # If mask is [batch_size, seq_len_q, seq_len_k], reshape to [batch_size, 1, seq_len_q, seq_len_k]
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            attention_scores = attention_scores + attention_mask
        
        # Apply causal masking if requested (for decoder self-attention)
        if is_causal:
            causal_mask = torch.triu(
                torch.ones(seq_len_q, seq_len_k, device=q.device),
                diagonal=1
            ).bool()
            attention_scores.masked_fill_(causal_mask[None, None, :, :], float("-inf"))
        
        # Compute attention weights using softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply dropout
        attention_weights = self.dropout(attention_weights)
        
        # Compute attention output by weighted sum of values
        # [batch_size, n_heads, seq_len_q, seq_len_k] x [batch_size, n_heads, seq_len_k, d_v]
        # -> [batch_size, n_heads, seq_len_q, d_v]
        attention_output = torch.matmul(attention_weights, v)
        
        # Reshape attention output
        # [batch_size, n_heads, seq_len_q, d_v] -> [batch_size, seq_len_q, n_heads, d_v] -> [batch_size, seq_len_q, d_model]
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        
        # Final projection
        attention_output = self.output_proj(attention_output)
        
        # Apply dropout
        attention_output = self.dropout(attention_output)
        
        if output_attentions:
            return attention_output, attention_weights
        else:
            return attention_output