"""
Advanced usage example for the transformer library.

This example demonstrates:
1. Creating a custom transformer architecture
2. Training for a specific NLP task (sequence classification)
3. Using advanced features like ALiBi attention and SwiGLU activations
4. Custom learning rate scheduling and optimization techniques
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import logging
import json
from typing import List, Dict, Optional, Tuple, Any, Union

from transformer import Transformer
from transformer.config import TransformerConfig
from transformer.models.encoder import TransformerEncoder
from transformer.layers.attention import ALiBiAttention
from transformer.layers.feed_forward import GatedFeedForward
from transformer.training.train import Trainer
from transformer.training.optimization import get_cosine_schedule_with_warmup, AdamW
from transformer.utils.data_utils import create_tokenized_dataset, split_dataset


# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class CustomTransformerForClassification(nn.Module):
    """
    Custom transformer model for sequence classification.
    
    This implements an encoder-only transformer with:
    - ALiBi attention mechanism
    - SwiGLU activation in feed-forward networks
    - Pre-normalization architecture
    - Classifier head for sequence classification
    """
    
    def __init__(self, config: TransformerConfig, num_labels: int = 2):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.padding_idx)
        
        # Create encoder
        self.encoder = TransformerEncoder(config)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.Tanh(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, num_labels),
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for classification.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs (for segment embeddings) [batch_size, seq_len]
            labels: Optional labels for loss computation [batch_size]
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            
        Returns:
            Dictionary with model outputs
        """
        # Get embeddings
        embedding_output = self.embeddings(input_ids)
        
        # Pass through encoder
        encoder_outputs = self.encoder(
            inputs_embeds=embedding_output,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        # Get sequence output
        sequence_output = encoder_outputs["last_hidden_state"]
        
        # Use CLS token or mean pooling for classification
        if self.config.pooling_type == "cls":
            # Use the first token (CLS) for classification
            pooled_output = sequence_output[:, 0, :]
        else:
            # Use mean pooling over sequence (excluding padding)
            if attention_mask is not None:
                # Expand attention mask to match hidden size
                expanded_mask = attention_mask.unsqueeze(-1).expand(sequence_output.size())
                # Sum the hidden states where attention mask is 1
                sum_hidden = torch.sum(sequence_output * expanded_mask, dim=1)
                # Count non-padding tokens
                token_count = torch.sum(attention_mask, dim=1, keepdim=True)
                # Calculate mean
                pooled_output = sum_hidden / token_count
            else:
                # If no mask, simply mean over sequence
                pooled_output = torch.mean(sequence_output, dim=1)
        
        # Compute logits
        logits = self.classifier(pooled_output)
        
        # Prepare outputs
        outputs = {"logits": logits}
        
        # Add loss if labels are provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            outputs["loss"] = loss
        
        # Add encoder outputs if requested
        if output_hidden_states:
            outputs["hidden_states"] = encoder_outputs["hidden_states"]
        
        if output_attentions:
            outputs["attentions"] = encoder_outputs["attentions"]
        
        return outputs


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Advanced transformer usage example")
    
    # Model parameters
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--d_ff", type=int, default=3072, help="Feed-forward dimension")
    parser.add_argument("--dropout", type=int, default=0.1, help="Dropout rate")
    parser.add_argument("--attention_dropout", type=float, default=0.1, help="Attention dropout rate")
    parser.add_argument("--hidden_dropout", type=float, default=0.1, help="Hidden dropout rate")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    
    # Architecture choices
    parser.add_argument("--attention_type", type=str, default="alibi", choices=["scaled_dot", "alibi", "rotary"],
                        help="Type of attention mechanism")
    parser.add_argument("--activation", type=str, default="swiglu", choices=["gelu", "relu", "swiglu"],
                        help="Activation function")
    parser.add_argument("--pooling_type", type=str, default="mean", choices=["cls", "mean"],
                        help="Type of pooling for classification")
    parser.add_argument("--use_pre_norm", action="store_true", help="Use pre-norm architecture")
    
    # Training parameters
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Ratio of warmup steps")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32, help="Evaluation batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    parser.add_argument("--num_labels", type=int, default=2, help="Number of labels for classification")
    parser.add_argument("--train_file", type=str, required=True, help="Training data file")
    parser.add_argument("--validation_file", type=str, help="Validation data file")
    parser.add_argument("--test_file", type=str, help="Test data file")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./advanced_outputs", help="Output directory")
    parser.add_argument("--logging_steps", type=int, default=100, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save steps")
    parser.add_argument("--eval_steps", type=int, default=1000, help="Evaluation steps")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


def load_dataset_from_json(file_path: str, max_seq_length: int = 512, tokenizer=None) -> Dict[str, List]:
    """
    Load dataset from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        max_seq_length: Maximum sequence length
        tokenizer: Tokenizer to use
        
    Returns:
        Dictionary with texts and labels
    """
    logger.info(f"Loading dataset from {file_path}")
    
    # Load JSON file
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Extract texts and labels
    texts = [item["text"] for item in data]
    labels = [item["label"] for item in data]
    
    # Create attention masks based on sequence lengths
    attention_masks = []
    
    # If tokenizer is available, tokenize texts
    if tokenizer is not None:
        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_masks = encoded["attention_mask"]
    else:
        # Otherwise, use dummy input IDs
        input_ids = torch.randint(0, 1000, (len(texts), max_seq_length))
        attention_masks = torch.ones_like(input_ids)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": torch.tensor(labels),
    }


def create_model_config(args, vocab_size: int = 30000, num_labels: int = 2) -> TransformerConfig:
    """
    Create a configuration for the model based on command-line arguments.
    
    Args:
        args: Command-line arguments
        vocab_size: Vocabulary size
        num_labels: Number of labels for classification
        
    Returns:
        Model configuration
    """
    config = TransformerConfig(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        hidden_dropout=args.hidden_dropout,
        max_seq_length=args.max_seq_length,
        activation_function=args.activation,
        positional_encoding_type=args.attention_type,
        use_pre_norm=args.use_pre_norm,
        is_encoder_decoder=False,
        is_decoder=False,
    )
    
    # Add custom fields
    config.num_labels = num_labels
    config.pooling_type = args.pooling_type
    
    return config


def get_optimizer_and_scheduler(
    model: nn.Module,
    args,
    num_training_steps: int,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """
    Create an optimizer and scheduler.
    
    Args:
        model: Model to optimize
        args: Command-line arguments
        num_training_steps: Number of training steps
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    # Get parameters without weight decay for bias and LayerNorm
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    # Create optimizer
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    # Calculate warmup steps
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    
    # Create scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    return optimizer, scheduler


def main():
    """Main function to run the advanced transformer example."""
    # Parse arguments
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Try to load a tokenizer from transformers
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        vocab_size = tokenizer.vocab_size
    except ImportError:
        logger.warning("Could not import transformers. Using dummy tokenizer.")
        vocab_size = 30000
    
    # Load datasets
    train_dataset = load_dataset_from_json(
        os.path.join(args.data_dir, args.train_file),
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
    )
    
    # Load validation dataset if available
    eval_dataset = None
    if args.validation_file:
        eval_dataset = load_dataset_from_json(
            os.path.join(args.data_dir, args.validation_file),
            max_seq_length=args.max_seq_length,
            tokenizer=tokenizer,
        )
    
    # Create model configuration
    config = create_model_config(args, vocab_size=vocab_size, num_labels=args.num_labels)
    
    # Create model
    model = CustomTransformerForClassification(config, num_labels=args.num_labels)
    
    # Print model architecture
    logger.info(f"Model architecture:\n{model}")
    
    # Calculate training steps
    num_update_steps_per_epoch = len(train_dataset["input_ids"]) // (
        args.per_device_train_batch_size * args.gradient_accumulation_steps
    )
    num_training_steps = num_update_steps_per_epoch * args.num_train_epochs
    
    # Create optimizer and scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(model, args, num_training_steps)
    
    # Create dummy train dataset object that matches the expected interface of Trainer
    class DummyDataset:
        def __init__(self, data):
            self.data = data
            self.length = len(data["input_ids"])
        
        def __len__(self):
            return self.length
        
        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.data.items()}
    
    train_dataset_obj = DummyDataset(train_dataset)
    eval_dataset_obj = DummyDataset(eval_dataset) if eval_dataset is not None else None
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset_obj,
        eval_dataset=eval_dataset_obj,
        tokenizer=tokenizer,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=1.0,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=int(args.warmup_ratio * num_training_steps),
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=2,
    )
    
    # Train the model
    logger.info("Starting training...")
    training_results = trainer.train()
    
    # Save the final model
    trainer.save_model()
    
    # Save training arguments and results
    with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    with open(os.path.join(args.output_dir, "training_results.json"), "w") as f:
        json.dump(training_results, f, indent=2)
    
    logger.info(f"Training completed! Model and results saved to {args.output_dir}")
    
    # Evaluate on test set if available
    if args.test_file:
        logger.info("Evaluating on test set...")
        test_dataset = load_dataset_from_json(
            os.path.join(args.data_dir, args.test_file),
            max_seq_length=args.max_seq_length,
            tokenizer=tokenizer,
        )
        test_dataset_obj = DummyDataset(test_dataset)
        
        # Replace eval_dataset with test_dataset
        trainer.eval_dataset = test_dataset_obj
        
        # Run evaluation
        test_results = trainer.evaluate()
        
        # Save test results
        with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
            json.dump(test_results, f, indent=2)
        
        logger.info(f"Test results: {test_results}")


if __name__ == "__main__":
    main()
