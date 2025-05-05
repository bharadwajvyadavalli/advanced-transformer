"""
Optimization utilities for transformer models.
"""
import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from typing import Callable, Iterable, Optional, Tuple, Union


class AdamW(Optimizer):
    """
    Implements AdamW optimizer with weight decay fix as introduced in
    "Decoupled Weight Decay Regularization" paper.
    """
    
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        """
        Initialize AdamW optimizer.
        
        Args:
            params: Iterable of parameters to optimize
            lr: Learning rate
            betas: Coefficients for computing running averages of gradient and its square
            eps: Term added to the denominator to improve numerical stability
            weight_decay: Weight decay (L2 penalty)
            correct_bias: Whether to correct bias in Adam
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias,
        )
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """
        Perform a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
            
        Returns:
            Loss value from closure if provided
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # Get parameter gradients
                grad = p.grad.data
                
                # Skip if there are no gradients
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                
                # Get optimizer state
                state = self.state[p]
                
                # Initialize state if needed
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                
                # Get optimizer parameters
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                
                # Update step count
                state["step"] += 1
                
                # Decay the first and second moment running averages
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                
                # Bias correction
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1
                else:
                    step_size = group["lr"]
                
                # Update parameters
                denom = exp_avg_sq.sqrt().add_(group["eps"])
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
                # Add weight decay after the main gradient-based updates
                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])
        
        return loss


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr
    after a warmup period to 0.
    
    Args:
        optimizer: Optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Number of training steps
        last_epoch: The index of the last epoch
        
    Returns:
        Learning rate scheduler
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Warmup phase
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Linear decay phase
        return max(
            0.0, 
            float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases following the values of the
    cosine function between the initial lr and 0, after a warmup period.
    
    Args:
        optimizer: Optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Number of training steps
        num_cycles: Number of cycles in the cosine schedule
        last_epoch: The index of the last epoch
        
    Returns:
        Learning rate scheduler
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Warmup phase
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_constant_schedule_with_warmup(
    optimizer: Optimizer, 
    num_warmup_steps: int, 
    last_epoch: int = -1
) -> LambdaLR:
    """
    Create a schedule with a constant learning rate preceded by a warmup period.
    
    Args:
        optimizer: Optimizer
        num_warmup_steps: Number of warmup steps
        last_epoch: The index of the last epoch
        
    Returns:
        Learning rate scheduler
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Warmup phase
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Constant phase
        return 1.0
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)
