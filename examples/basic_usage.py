"""
Basic usage example for the transformer library.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import logging

from transformer import Transformer
from transformer.config import TransformerConfig
from transformer.models.encoder import TransformerEncoder
from transformer.models.decoder import TransformerDecoder
from transformer.utils.data_utils import get_tokenizer, create_tokenized_dataset, create_dataloader, split_dataset
from transformer.utils.model_utils import save_model, load_model, get_optimizer
from transformer.training.train import Trainer


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a transformer model")
    
    # Model parameters
    parser.add_argument("--model_type", type=str, default="encoder-decoder", 
                        choices=["encoder-decoder", "encoder", "decoder"],
                        help="Type of model to train")
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--d_ff", type=int, default=1024, help="Feed-forward dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per device for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size per device for evaluation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--logging_steps", type=int, default=100, help="Number of steps between logging")
    parser.add_argument("--save_steps", type=int, default=500, help="Number of steps between saving")
    parser.add_argument("--eval_steps", type=int, default=500, help="Number of steps between evaluation")
    
    # Dataset parameters
    parser.add_argument("--data_file", type=str, required=True, help="Path to data file")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--load_model_path", type=str, default=None, help="Path to load model from")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer")
    
    return parser.parse_args()


def create_dummy_dataset(size=1000, seq_len=50, vocab_size=10000):
    """Create a dummy dataset for testing."""
    # Create random token sequences for inputs
    input_ids = [np.random.randint(1, vocab_size, size=seq_len).tolist() for _ in range(size)]
    
    # Create random labels (0 or 1 for classification, or token sequences for seq2seq)
    labels = [np.random.randint(0, 2) for _ in range(size)]
    
    return input_ids, labels


def demo_training():
    """Demonstrate training a transformer model."""
    # Parse arguments
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create configuration
    config = TransformerConfig(
        vocab_size=30000,  # Placeholder - will be updated with tokenizer vocab size
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_seq_length=args.max_seq_length,
        is_encoder_decoder=args.model_type == "encoder-decoder",
        is_decoder=args.model_type == "decoder",
    )
    
    # Get tokenizer or create one
    if args.tokenizer_path:
        tokenizer = get_tokenizer(args.tokenizer_path)
    else:
        # For demonstration, we'll use a simple tokenizer from transformers
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        except ImportError:
            logger.error("Could not import transformers. Please install with: pip install transformers")
            # Create a simple tokenizer
            logger.info("Using a simple placeholder tokenizer")
            tokenizer = None
    
    # Update vocabulary size if tokenizer is available
    if tokenizer is not None and hasattr(tokenizer, "vocab_size"):
        config.vocab_size = tokenizer.vocab_size
    
    # Load or create model
    if args.load_model_path:
        model, config, tokenizer, _, _ = load_model(args.load_model_path)
    else:
        model = Transformer(config)
    
    # For this example, let's create a dummy dataset
    logger.info("Creating dummy dataset")
    input_ids, labels = create_dummy_dataset(size=1000, seq_len=args.max_seq_length, vocab_size=config.vocab_size)
    
    # Create attention masks (1 for tokens, 0 for padding)
    attention_mask = [[1] * len(ids) for ids in input_ids]
    
    # Create datasets
    dataset = create_tokenized_dataset(
        texts=[],  # Not using actual texts
        tokenizer=None,  # Not using tokenizer for dummy data
        labels=labels,
        return_tensors=True,
    )
    
    # Manually set input_ids and attention_mask
    dataset.input_ids = torch.tensor(input_ids)
    dataset.attention_mask = torch.tensor(attention_mask)
    
    # Split into train, validation, and test sets
    train_dataset, eval_dataset, _ = split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    
    # Create optimizer and scheduler
    optimizer, scheduler = get_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
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
        warmup_steps=args.warmup_steps,
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=2,
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    
    logger.info(f"Training completed! Model saved to {args.output_dir}")


def demo_inference():
    """Demonstrate inference with a trained transformer model."""
    # Parse arguments
    args = parse_args()
    
    # Load model, configuration, and tokenizer
    model, config, tokenizer, _, _ = load_model(args.load_model_path or args.output_dir)
    
    # Set model to evaluation mode
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Example input (replace with real input)
    example_input = torch.randint(0, config.vocab_size, (1, 10)).to(device)
    example_attn_mask = torch.ones_like(example_input).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(
            input_ids=example_input,
            attention_mask=example_attn_mask,
        )
    
    # Process outputs
    logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
    predictions = torch.argmax(logits, dim=-1)
    
    logger.info(f"Input shape: {example_input.shape}")
    logger.info(f"Output logits shape: {logits.shape}")
    logger.info(f"Predictions shape: {predictions.shape}")
    
    # If tokenizer is available, decode the predictions
    if tokenizer is not None and hasattr(tokenizer, "decode"):
        decoded = tokenizer.decode(predictions[0].tolist())
        logger.info(f"Decoded prediction: {decoded}")
    
    return predictions


def demo_text_generation():
    """Demonstrate text generation with a transformer model."""
    # Parse arguments
    args = parse_args()
    
    # Load model, configuration, and tokenizer
    model, config, tokenizer, _, _ = load_model(args.load_model_path or args.output_dir)
    
    # Set model to evaluation mode
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Check if the model is suitable for generation
    if not config.is_decoder and not config.is_encoder_decoder:
        logger.warning("Model is not a decoder or encoder-decoder model, text generation may not work")
    
    # Prompt text (replace with real prompt)
    prompt = "Once upon a time"
    
    # Tokenize prompt
    if tokenizer is not None:
        prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    else:
        # Without a tokenizer, use a dummy input
        prompt_ids = torch.tensor([[1, 2, 3, 4, 5]]).to(device)
    
    # Generate text
    generated_ids = model.generate(
        input_ids=prompt_ids,
        max_length=50,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )
    
    # Decode generated text
    if tokenizer is not None and hasattr(tokenizer, "decode"):
        generated_text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
        logger.info(f"Generated text: {generated_text}")
    else:
        logger.info(f"Generated IDs: {generated_ids[0].tolist()}")
    
    return generated_ids


def demo_sequence_classification():
    """Demonstrate sequence classification with a transformer model."""
    # Parse arguments
    args = parse_args()
    
    # Load model, configuration, and tokenizer
    model, config, tokenizer, _, _ = load_model(args.load_model_path or args.output_dir)
    
    # Set model to evaluation mode
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Example texts (replace with real examples)
    texts = [
        "This is a positive example.",
        "This is a negative example."
    ]
    
    # Tokenize texts
    if tokenizer is not None:
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    else:
        # Without a tokenizer, use dummy inputs
        inputs = {
            "input_ids": torch.tensor([
                [1, 2, 3, 4, 5, 0, 0],
                [6, 7, 8, 9, 0, 0, 0],
            ]).to(device),
            "attention_mask": torch.tensor([
                [1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0, 0, 0],
            ]).to(device),
        }
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Process outputs
    logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
    predictions = torch.argmax(logits, dim=-1)
    
    # Log results
    for i, text in enumerate(texts):
        logger.info(f"Text: {text}")
        logger.info(f"Prediction: {predictions[i].item()}")
    
    return predictions


def main():
    """Main function to demonstrate transformer usage."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    
    # Parse arguments
    args = parse_args()
    
    # Decide which demo to run
    if args.load_model_path is None:
        # If no model is provided, train a new one
        logger.info("No model path provided. Training a new model...")
        demo_training()
    else:
        # If a model is provided, run inference
        logger.info(f"Loading model from {args.load_model_path}...")
        
        # Run all demos
        logger.info("Running inference demo...")
        demo_inference()
        
        logger.info("Running text generation demo...")
        demo_text_generation()
        
        logger.info("Running sequence classification demo...")
        demo_sequence_classification()


if __name__ == "__main__":
    main()
