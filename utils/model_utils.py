"""
Model utilities for transformer models.
"""
import os
import json
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union, List, Tuple

from ..config import TransformerConfig
from ..models.transformer import Transformer
from ..training.optimization import AdamW, get_linear_schedule_with_warmup


def save_model(
    model: nn.Module,
    config: TransformerConfig,
    output_dir: str,
    tokenizer: Optional[Any] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> None:
    """
    Save model, configuration, tokenizer, optimizer, and scheduler.
    
    Args:
        model: Model to save
        config: Model configuration
        output_dir: Output directory
        tokenizer: Tokenizer to save
        optimizer: Optimizer to save
        scheduler: Scheduler to save
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model state dictionary
    model_to_save = model.module if hasattr(model, "module") else model
    model_path = os.path.join(output_dir, "model.pt")
    torch.save(model_to_save.state_dict(), model_path)
    
    # Save configuration
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        if hasattr(config, "to_dict"):
            json.dump(config.to_dict(), f, indent=2)
        else:
            json.dump(config.__dict__, f, indent=2)
    
    # Save tokenizer if available
    if tokenizer is not None:
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(output_dir)
        else:
            # Try to save tokenizer attributes
            tokenizer_path = os.path.join(output_dir, "tokenizer.json")
            try:
                with open(tokenizer_path, "w") as f:
                    json.dump(tokenizer.__dict__, f, indent=2)
            except TypeError:
                print("Warning: Tokenizer could not be saved as JSON. Skipping.")
    
    # Save optimizer and scheduler if available
    if optimizer is not None:
        optimizer_path = os.path.join(output_dir, "optimizer.pt")
        torch.save(optimizer.state_dict(), optimizer_path)
    
    if scheduler is not None:
        scheduler_path = os.path.join(output_dir, "scheduler.pt")
        torch.save(scheduler.state_dict(), scheduler_path)


def load_model(
    model_path: str,
    device: Optional[torch.device] = None,
    load_optimizer: bool = False,
    load_scheduler: bool = False,
) -> Tuple[nn.Module, TransformerConfig, Optional[Any], Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler._LRScheduler]]:
    """
    Load model, configuration, tokenizer, optimizer, and scheduler.
    
    Args:
        model_path: Path to model directory
        device: Device to load model on
        load_optimizer: Whether to load optimizer
        load_scheduler: Whether to load scheduler
        
    Returns:
        Tuple of (model, config, tokenizer, optimizer, scheduler)
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load configuration
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    # Create configuration object
    config = TransformerConfig(**config_dict)
    
    # Initialize model
    model = Transformer(config)
    
    # Load model state
    model_file = os.path.join(model_path, "model.pt")
    model.load_state_dict(torch.load(model_file, map_location=device))
    
    # Move model to device
    model.to(device)
    
    # Load tokenizer if available
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        if os.path.exists(os.path.join(model_path, "tokenizer_config.json")):
            tokenizer = AutoTokenizer.from_pretrained(model_path)
    except (ImportError, FileNotFoundError):
        tokenizer_path = os.path.join(model_path, "tokenizer.json")
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, "r") as f:
                tokenizer_dict = json.load(f)
            
            # This is a basic placeholder - creating a functional tokenizer would need more logic
            class SimpleTokenizer:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)
            
            tokenizer = SimpleTokenizer(**tokenizer_dict)
    
    # Load optimizer if requested
    optimizer = None
    if load_optimizer:
        optimizer_path = os.path.join(model_path, "optimizer.pt")
        if os.path.exists(optimizer_path):
            # Create optimizer
            optimizer = get_optimizer(model, config.learning_rate if hasattr(config, "learning_rate") else 5e-5)
            
            # Load optimizer state
            optimizer_state = torch.load(optimizer_path, map_location=device)
            optimizer.load_state_dict(optimizer_state)
    
    # Load scheduler if requested
    scheduler = None
    if load_scheduler:
        scheduler_path = os.path.join(model_path, "scheduler.pt")
        if os.path.exists(scheduler_path):
            # Create scheduler (simplified, assumes a schedule with warmup)
            scheduler = get_linear_schedule_with_warmup(
                optimizer or get_optimizer(model, config.learning_rate if hasattr(config, "learning_rate") else 5e-5),
                num_warmup_steps=0,
                num_training_steps=1000,  # Placeholder value, will be overwritten
            )
            
            # Load scheduler state
            scheduler_state = torch.load(scheduler_path, map_location=device)
            scheduler.load_state_dict(scheduler_state)
    
    return model, config, tokenizer, optimizer, scheduler


def get_optimizer(
    model: nn.Module,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.0,
    warmup_steps: int = 0,
    total_steps: int = 0,
) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
    """
    Get optimizer and scheduler for a model.
    
    Args:
        model: Model
        learning_rate: Learning rate
        weight_decay: Weight decay
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    # Get parameters that require gradients
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    # Create AdamW optimizer
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    # Create scheduler if warmup steps and total steps are provided
    scheduler = None
    if warmup_steps > 0 and total_steps > 0:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    
    return optimizer, scheduler
