"""
Feed-forward networks for transformer models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable


class PositionWiseFFN(nn.Module):
    """
    Position-wise Feed-Forward Network for Transformers.
    
    This implements the FFN in Transformer architecture:
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    
    With support for different activation functions, including GELU, ReLU, and SwiGLU.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation
        
        # First linear transformation
        self.linear1 = nn.Linear(d_model, d_ff)
        
        # Second linear transformation
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # Additional linear for SwiGLU
        self.linear3 = nn.Linear(d_model, d_ff) if activation == "swiglu" else None
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Select activation function
        if activation == "relu":
            self.activation_fn = F.relu
        elif activation == "gelu":
            self.activation_fn = F.gelu
        elif activation == "swiglu":
            # SwiGLU doesn't use activation_fn directly
            pass
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the position-wise feed-forward network.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        if self.activation == "swiglu":
            # SwiGLU variant
            x1 = self.linear1(x)
            x2 = self.linear3(x)
            x = F.silu(x1) * x2
        else:
            # Standard FFN with activation function
            x = self.linear1(x)
            x = self.activation_fn(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Second linear transformation
        x = self.linear2(x)
        
        return x


class GatedFeedForward(nn.Module):
    """
    Gated Feed-Forward Network with various gating mechanisms.
    
    This is a more general implementation that supports different gating mechanisms:
    - GLU (Gated Linear Unit)
    - SwiGLU (Swish Gated Linear Unit)
    - GeGLU (GELU Gated Linear Unit)
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        gate_type: str = "swiglu",
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Value projection
        self.w_v = nn.Linear(d_model, d_ff)
        
        # Gate projection
        self.w_g = nn.Linear(d_model, d_ff)
        
        # Output projection
        self.w_o = nn.Linear(d_ff, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Gate type
        self.gate_type = gate_type
        
        # Gate activation function
        if gate_type == "glu":
            self.gate_activation = lambda x: torch.sigmoid(x)
        elif gate_type == "swiglu":
            self.gate_activation = F.silu
        elif gate_type == "geglu":
            self.gate_activation = F.gelu
        else:
            raise ValueError(f"Unsupported gate_type: {gate_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the gated feed-forward network.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        # Value projection
        v = self.w_v(x)
        
        # Gate projection and activation
        g = self.gate_activation(self.w_g(x))
        
        # Apply gating
        x = v * g
        
        # Apply dropout
        x = self.dropout(x)
        
        # Output projection
        x = self.w_o(x)
        
        return x


class MoeFFN(nn.Module):
    """
    Mixture of Experts Feed-Forward Network.
    
    This implements a simple Mixture of Experts (MoE) layer with a gating network
    that selects which expert to use for each token.
    
    NOTE: This is a simplified version of MoE. A real-world implementation would
    include load balancing, capacity factors, and distributed training support.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        num_experts_per_token: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        
        # Create experts (each expert is a feed-forward network)
        self.experts = nn.ModuleList([
            PositionWiseFFN(d_model, d_ff, dropout, activation)
            for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Linear(d_model, num_experts)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the mixture of experts feed-forward network.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Reshape for per-token routing
        x_flat = x.view(-1, d_model)  # [batch_size * seq_len, d_model]
        
        # Compute routing probabilities
        router_logits = self.gate(x_flat)  # [batch_size * seq_len, num_experts]
        
        # Get top-k experts for each token
        routing_weights, selected_experts = torch.topk(
            router_logits, 
            self.num_experts_per_token, 
            dim=-1
        )  # Both: [batch_size * seq_len, num_experts_per_token]
        
        # Normalize weights with softmax
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        # Initialize output
        final_output = torch.zeros_like(x_flat)
        
        # Process each token through its selected experts
        for i in range(self.num_experts_per_token):
            # Get expert indices for this position
            expert_idx = selected_experts[:, i]  # [batch_size * seq_len]
            
            # Get weights for this position
            weight = routing_weights[:, i].unsqueeze(-1)  # [batch_size * seq_len, 1]
            
            # Process each token with its selected expert
            for j, expert in enumerate(self.experts):
                # Find which tokens use this expert
                mask = (expert_idx == j)
                
                if mask.any():
                    # Process only the tokens that use this expert
                    expert_input = x_flat[mask]
                    expert_output = expert(expert_input)
                    
                    # Apply weights and add to final output
                    final_output[mask] += weight[mask] * expert_output
        
        # Reshape back to original dimensions
        output = final_output.view(batch_size, seq_len, d_model)
        
        return output
