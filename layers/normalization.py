"""
Normalization layers for transformer models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerLayerNorm(nn.Module):
    """
    Layer normalization for transformers.
    
    This implements layer normalization with optional bias terms.
    """
    
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-12,
        bias: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size)) if bias else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for layer normalization.
        
        Args:
            x: Input tensor of shape [..., hidden_size]
            
        Returns:
            Layer normalized tensor of shape [..., hidden_size]
        """
        # Compute mean and variance along last dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize
        x = (x - mean) / torch.sqrt(var + self.eps)
        
        # Scale and shift
        if self.bias is not None:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight
        
        return x


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization for transformers.
    
    This implements RMSNorm as described in "Root Mean Square Layer Normalization"
    https://arxiv.org/abs/1910.07467
    """
    
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-12,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size)) if bias else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RMS normalization.
        
        Args:
            x: Input tensor of shape [..., hidden_size]
            
        Returns:
            RMS normalized tensor of shape [..., hidden_size]
        """
        # Compute RMS along last dimension
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # Normalize
        x = x / rms
        
        # Scale and shift
        if self.bias is not None:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight
        
        return x


class GatedRMSNorm(nn.Module):
    """
    Gated RMS Normalization for transformers.
    
    This is a variant of RMSNorm with a learnable gate that controls
    how much normalization is applied.
    """
    
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-12,
        bias: bool = False,
        init_gate: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size)) if bias else None
        
        # Gate parameter
        self.gate = nn.Parameter(torch.ones(1) * init_gate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for gated RMS normalization.
        
        Args:
            x: Input tensor of shape [..., hidden_size]
            
        Returns:
            Gated RMS normalized tensor of shape [..., hidden_size]
        """
        # Compute RMS along last dimension
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # Normalize
        x_norm = x / rms
        
        # Scale and shift
        if self.bias is not None:
            x_norm = x_norm * self.weight + self.bias
        else:
            x_norm = x_norm * self.weight
        
        # Apply gating
        gate = torch.sigmoid(self.gate)
        
        return gate * x_norm + (1 - gate) * x


class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization for transformers.
    
    This implements a layer normalization with adaptive weights and biases
    that are computed from the input features.
    """
    
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        
        # Projections to compute adaptive parameters
        self.weight_proj = nn.Linear(hidden_size, hidden_size)
        self.bias_proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for adaptive layer normalization.
        
        Args:
            x: Input tensor of shape [..., hidden_size]
            
        Returns:
            Adaptively normalized tensor of shape [..., hidden_size]
        """
        # Compute mean and variance along last dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Compute adaptive parameters
        # We use the mean across the feature dimension to predict adaptive parameters
        feature_mean = x.mean(dim=-1, keepdim=True)
        
        # Compute adaptive weights and biases
        weight = self.weight_proj(feature_mean)
        bias = self.bias_proj(feature_mean)
        
        # Apply adaptive scaling and shifting
        return x_norm * weight + bias
