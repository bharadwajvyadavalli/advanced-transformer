"""
Training utilities for transformer models.
"""
import os
import math
import time
import json
import logging
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict, Any, Union, List, Tuple, Callable

from ..models.transformer import Transformer
from .optimization import get_linear_schedule_with_warmup, AdamW


logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer class for transformer models.
    
    This handles the training loop, evaluation, and saving/loading of models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[Any] = None,
        data_collator: Optional[Callable] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 8,
        per_device_eval_batch_size: int = 8,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        output_dir: str = "./outputs",
        logging_dir: Optional[str] = None,
        logging_steps: int = 500,
        save_steps: int = 500,
        eval_steps: int = 500,
        save_total_limit: Optional[int] = None,
        fp16: bool = False,
        fp16_opt_level: str = "O1",
        local_rank: int = -1,
        seed: int = 42,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            tokenizer: Tokenizer
            data_collator: Function to collate data
            optimizer: Optimizer
            lr_scheduler: Learning rate scheduler
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Batch size per device for training
            per_device_eval_batch_size: Batch size per device for evaluation
            gradient_accumulation_steps: Number of steps to accumulate gradients
            max_grad_norm: Maximum gradient norm
            learning_rate: Learning rate
            weight_decay: Weight decay
            warmup_steps: Number of warmup steps
            output_dir: Directory to save model and results
            logging_dir: Directory to save logs
            logging_steps: Number of steps between logging
            save_steps: Number of steps between saving
            eval_steps: Number of steps between evaluation
            save_total_limit: Maximum number of checkpoints to keep
            fp16: Whether to use mixed precision training
            fp16_opt_level: Mixed precision optimization level
            local_rank: Local rank for distributed training
            seed: Random seed
        """
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.data_collator = data_collator or self._default_data_collator
        
        # Training parameters
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        
        # Output and logging
        self.output_dir = output_dir
        self.logging_dir = logging_dir or os.path.join(output_dir, "logs")
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.save_total_limit = save_total_limit
        
        # Mixed precision training
        self.fp16 = fp16
        self.fp16_opt_level = fp16_opt_level
        
        # Distributed training
        self.local_rank = local_rank
        self.seed = seed
        
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Prepare output directory
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logging_dir, exist_ok=True)
        
        # Initialize tensorboard
        self.tb_writer = SummaryWriter(self.logging_dir)
        
        # Check for CUDA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = optimizer or self._create_optimizer()
        self.lr_scheduler = lr_scheduler or self._create_scheduler()
        
        # Initialize mixed precision training if requested
        if self.fp16:
            try:
                from apex import amp
                self.model, self.optimizer = amp.initialize(
                    self.model, self.optimizer, opt_level=self.fp16_opt_level
                )
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        
        # Wrap model in DataParallel if multiple GPUs are available
        if self.n_gpu > 1:
            self.model = nn.DataParallel(self.model)
        
        # Prepare data loaders
        self.train_dataloader = self._get_train_dataloader()
        self.eval_dataloader = self._get_eval_dataloader() if self.eval_dataset is not None else None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
    
    def _default_data_collator(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Default data collator that simply stacks the examples and converts to tensors.
        
        Args:
            examples: List of examples
            
        Returns:
            Batch dictionary
        """
        # Check if all examples are dictionaries
        if isinstance(examples[0], dict):
            batch = {}
            for key in examples[0].keys():
                if key in ["input_ids", "attention_mask", "labels"]:
                    # Convert to tensors for these keys
                    batch[key] = torch.tensor([example[key] for example in examples])
                else:
                    # Keep other data as-is
                    batch[key] = [example[key] for example in examples]
            return batch
        else:
            # For non-dictionary examples (e.g., simple list of token IDs)
            return torch.tensor(examples)
    
    def _get_train_dataloader(self) -> DataLoader:
        """
        Create a training dataloader.
        
        Returns:
            DataLoader for training
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=4,
            pin_memory=True,
        )
    
    def _get_eval_dataloader(self) -> DataLoader:
        """
        Create an evaluation dataloader.
        
        Returns:
            DataLoader for evaluation
        """
        return DataLoader(
            self.eval_dataset,
            batch_size=self.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=4,
            pin_memory=True,
        )
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create an optimizer.
        
        Returns:
            Optimizer
        """
        # Get parameters that require gradients
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        # Create AdamW optimizer
        return AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    
    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Create a learning rate scheduler.
        
        Returns:
            Learning rate scheduler
        """
        # Calculate total training steps
        num_examples = len(self.train_dataset)
        num_update_steps_per_epoch = math.ceil(
            num_examples / (self.per_device_train_batch_size * self.gradient_accumulation_steps)
        )
        num_training_steps = num_update_steps_per_epoch * self.num_train_epochs
        
        # Create linear scheduler with warmup
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=num_training_steps,
        )
    
    def train(self) -> Dict[str, float]:
        """
        Train the model.
        
        Returns:
            Dictionary of training metrics
        """
        logger.info("***** Running training *****")
        logger.info(f"  Number of examples = {len(self.train_dataset)}")
        logger.info(f"  Number of epochs = {self.num_train_epochs}")
        logger.info(f"  Batch size per device = {self.per_device_train_batch_size}")
        logger.info(f"  Gradient accumulation steps = {self.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {len(self.train_dataloader) // self.gradient_accumulation_steps * self.num_train_epochs}")
        
        # Training loop
        train_losses = []
        for epoch in range(self.num_train_epochs):
            self.epoch = epoch
            
            # Set model to training mode
            self.model.train()
            
            # Initialize epoch progress bar
            epoch_iterator = tqdm.tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch+1}/{self.num_train_epochs}",
                disable=self.local_rank not in [-1, 0],
            )
            
            # Track batch loss
            batch_losses = []
            
            # Process batches
            for step, batch in enumerate(epoch_iterator):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                
                # Get loss
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                
                # Account for gradient accumulation
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                if self.fp16:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                
                # Track batch loss
                batch_losses.append(loss.item() * self.gradient_accumulation_steps)
                
                # Update epoch progress description with loss
                epoch_iterator.set_description(
                    f"Epoch {epoch+1}/{self.num_train_epochs} (Loss: {batch_losses[-1]:.4f})"
                )
                
                # Update parameters if we've accumulated enough gradients
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Clip gradients
                    if self.max_grad_norm > 0:
                        if self.fp16:
                            torch.nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer), self.max_grad_norm
                            )
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.max_grad_norm
                            )
                    
                    # Optimizer step
                    self.optimizer.step()
                    
                    # Scheduler step
                    self.lr_scheduler.step()
                    
                    # Zero gradients
                    self.optimizer.zero_grad()
                    
                    # Update global step
                    self.global_step += 1
                    
                    # Log metrics
                    if self.logging_steps > 0 and self.global_step % self.logging_steps == 0:
                        # Log to tensorboard
                        if len(batch_losses) > 0:
                            self.tb_writer.add_scalar(
                                "train/loss", np.mean(batch_losses), self.global_step
                            )
                        self.tb_writer.add_scalar(
                            "train/learning_rate", self.lr_scheduler.get_last_lr()[0], self.global_step
                        )
                    
                    # Save model
                    if self.save_steps > 0 and self.global_step % self.save_steps == 0:
                        self.save_model()
                    
                    # Evaluate model
                    if self.eval_steps > 0 and self.global_step % self.eval_steps == 0:
                        metrics = self.evaluate()
                        
                        # Log evaluation metrics
                        for key, value in metrics.items():
                            self.tb_writer.add_scalar(f"eval/{key}", value, self.global_step)
                        
                        # Save best model
                        if metrics.get("loss", float("inf")) < self.best_eval_loss:
                            self.best_eval_loss = metrics.get("loss", float("inf"))
                            self.save_model(os.path.join(self.output_dir, "best_model"))
            
            # Average epoch loss
            avg_epoch_loss = np.mean(batch_losses)
            train_losses.append(avg_epoch_loss)
            
            # Log epoch metrics
            logger.info(f"Epoch {epoch+1}/{self.num_train_epochs} - Average Loss: {avg_epoch_loss:.4f}")
            self.tb_writer.add_scalar("train/epoch_loss", avg_epoch_loss, epoch + 1)
            
            # Save model after each epoch
            self.save_model(os.path.join(self.output_dir, f"epoch_{epoch+1}"))
            
            # Evaluate after each epoch
            if self.eval_dataset is not None:
                metrics = self.evaluate()
                
                # Log evaluation metrics
                for key, value in metrics.items():
                    self.tb_writer.add_scalar(f"eval/epoch_{key}", value, epoch + 1)
                
                # Save best model
                if metrics.get("loss", float("inf")) < self.best_eval_loss:
                    self.best_eval_loss = metrics.get("loss", float("inf"))
                    self.save_model(os.path.join(self.output_dir, "best_model"))
        
        # Close tensorboard writer
        self.tb_writer.close()
        
        # Calculate and return overall metrics
        return {
            "train_loss": np.mean(train_losses),
            "best_eval_loss": self.best_eval_loss,
        }
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("***** Running evaluation *****")
        logger.info(f"  Number of examples = {len(self.eval_dataset)}")
        logger.info(f"  Batch size per device = {self.per_device_eval_batch_size}")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize metrics
        eval_losses = []
        eval_preds = []
        eval_labels = []
        
        # Evaluation loop
        for batch in tqdm.tqdm(self.eval_dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass without gradient computation
            with torch.no_grad():
                outputs = self.model(**batch)
            
            # Get loss
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            
            # Track loss
            eval_losses.append(loss.item())
            
            # Get predictions and labels if available
            if isinstance(outputs, dict) and "logits" in outputs:
                preds = torch.argmax(outputs["logits"], dim=-1)
                eval_preds.append(preds.cpu().numpy())
                if "labels" in batch:
                    eval_labels.append(batch["labels"].cpu().numpy())
            elif len(outputs) > 1:
                preds = torch.argmax(outputs[1], dim=-1)
                eval_preds.append(preds.cpu().numpy())
                if "labels" in batch:
                    eval_labels.append(batch["labels"].cpu().numpy())
        
        # Calculate metrics
        metrics = {"loss": np.mean(eval_losses)}
        
        # Add accuracy if we have predictions and labels
        if eval_preds and eval_labels:
            eval_preds = np.concatenate(eval_preds, axis=0)
            eval_labels = np.concatenate(eval_labels, axis=0)
            metrics["accuracy"] = np.mean(eval_preds == eval_labels)
        
        # Log metrics
        logger.info(f"***** Evaluation Results *****")
        for key, value in metrics.items():
            logger.info(f"  {key} = {value:.4f}")
        
        return metrics
    
    def save_model(self, path: Optional[str] = None) -> None:
        """
        Save the model, tokenizer and configuration.
        
        Args:
            path: Path to save the model
        """
        # Use the provided path or the default output directory
        path = path or self.output_dir
        os.makedirs(path, exist_ok=True)
        
        # Save model state dictionary
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(model_to_save.state_dict(), os.path.join(path, "model.pt"))
        
        # Save model configuration
        if hasattr(model_to_save, "config"):
            if hasattr(model_to_save.config, "to_dict"):
                config_dict = model_to_save.config.to_dict()
                with open(os.path.join(path, "config.json"), "w") as f:
                    json.dump(config_dict, f)
        
        # Save tokenizer if available
        if self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(path)
        
        # Save training arguments
        training_args = {
            "learning_rate": self.learning_rate,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "num_train_epochs": self.num_train_epochs,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_grad_norm": self.max_grad_norm,
            "global_step": self.global_step,
            "best_eval_loss": self.best_eval_loss,
        }
        
        with open(os.path.join(path, "training_args.json"), "w") as f:
            json.dump(training_args, f)
        
        logger.info(f"Model saved to {path}")
        
        # Manage save_total_limit
        if self.save_total_limit is not None and path == self.output_dir:
            self._rotate_checkpoints()
    
    def _rotate_checkpoints(self) -> None:
        """
        Remove old checkpoints if save_total_limit is reached.
        """
        # Get all checkpoint directories
        if self.save_total_limit <= 0:
            return
        
        # Find checkpoint directories (epoch_X)
        checkpoints = [
            path for path in os.listdir(self.output_dir)
            if path.startswith("epoch_") and os.path.isdir(os.path.join(self.output_dir, path))
        ]
        
        # Sort checkpoints by epoch number
        checkpoints = sorted(
            checkpoints,
            key=lambda x: int(x.split("_")[1])
        )
        
        # If we don't have too many checkpoints, return
        if len(checkpoints) <= self.save_total_limit:
            return
        
        # Remove the oldest checkpoints
        checkpoints_to_remove = checkpoints[:-self.save_total_limit]
        for checkpoint in checkpoints_to_remove:
            checkpoint_path = os.path.join(self.output_dir, checkpoint)
            logger.info(f"Removing old checkpoint {checkpoint_path}")
            import shutil
            shutil.rmtree(checkpoint_path)
    
    def load_model(self, path: str) -> None:
        """
        Load the model, tokenizer and configuration.
        
        Args:
            path: Path to load the model from
        """
        # Load model state dictionary
        state_dict = torch.load(os.path.join(path, "model.pt"), map_location=self.device)
        
        # Load model
        model_to_load = self.model.module if hasattr(self.model, "module") else self.model
        model_to_load.load_state_dict(state_dict)
        
        # Load tokenizer if available
        if self.tokenizer is not None and hasattr(self.tokenizer, "from_pretrained"):
            if os.path.exists(os.path.join(path, "tokenizer_config.json")):
                self.tokenizer = type(self.tokenizer).from_pretrained(path)
        
        # Load training arguments
        if os.path.exists(os.path.join(path, "training_args.json")):
            with open(os.path.join(path, "training_args.json"), "r") as f:
                training_args = json.load(f)
                self.global_step = training_args.get("global_step", 0)
                self.best_eval_loss = training_args.get("best_eval_loss", float("inf"))
        
        logger.info(f"Model loaded from {path}")
