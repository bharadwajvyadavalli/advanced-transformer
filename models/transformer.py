"""
Main Transformer model implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, Union

from ..config import TransformerConfig
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder


class Transformer(nn.Module):
    """
    Full Transformer model with encoder and decoder.
    
    This implements the architecture described in "Attention Is All You Need" 
    with modern improvements and optimizations.
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Create token embeddings
        self.token_embedding = nn.Embedding(
            config.vocab_size, 
            config.d_model, 
            padding_idx=config.padding_idx
        )
        
        # Determine the model type based on configuration
        if config.is_encoder_decoder:
            # Full encoder-decoder architecture
            self.encoder = TransformerEncoder(config)
            self.decoder = TransformerDecoder(config)
            self.is_encoder_decoder = True
        elif config.is_decoder:
            # Decoder-only architecture (like GPT)
            self.encoder = None
            self.decoder = TransformerDecoder(config)
            self.is_encoder_decoder = False
        else:
            # Encoder-only architecture (like BERT)
            self.encoder = TransformerEncoder(config)
            self.decoder = None
            self.is_encoder_decoder = False
        
        # Final output layer (prediction head)
        # This could be tied to the input embeddings
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize the weights according to the configuration."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass of the transformer model.
        
        Args:
            input_ids: Input token IDs for the encoder
            attention_mask: Mask to avoid attention on padding tokens in the encoder
            decoder_input_ids: Input token IDs for the decoder
            decoder_attention_mask: Mask to avoid attention on padding tokens in the decoder
            encoder_outputs: Pre-computed encoder outputs (to avoid re-computing them)
            past_key_values: Cached key values for faster inference
            use_cache: Whether to return a cache for faster inference
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return a dictionary or a tuple
            
        Returns:
            A tuple or dictionary containing:
            - logits: The prediction logits
            - past_key_values: The key-value cache for faster inference (if use_cache=True)
            - encoder_outputs: The encoder outputs
            - encoder_hidden_states: All hidden states of the encoder (if output_hidden_states=True)
            - encoder_attentions: All attention weights of the encoder (if output_attentions=True)
            - decoder_hidden_states: All hidden states of the decoder (if output_hidden_states=True)
            - decoder_attentions: All attention weights of the decoder (if output_attentions=True)
            - cross_attentions: All cross-attention weights (if output_attentions=True)
        """
        # Set default values for optional arguments
        use_cache = use_cache if use_cache is not None else self.config.use_cache if hasattr(self.config, "use_cache") else False
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        return_dict = return_dict if return_dict is not None else True
        
        # Process through encoder (if available and not already computed)
        if self.is_encoder_decoder:
            if encoder_outputs is None and self.encoder is not None and input_ids is not None:
                encoder_outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            
            # Process through decoder with encoder outputs
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_outputs.last_hidden_state if hasattr(encoder_outputs, "last_hidden_state") else encoder_outputs[0],
                encoder_attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        
        elif self.decoder is not None:
            # Decoder-only architecture (e.g., GPT)
            decoder_outputs = self.decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        
        else:
            # Encoder-only architecture (e.g., BERT)
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            decoder_outputs = encoder_outputs  # For consistency in return handling
        
        # Get the sequence output from decoder or encoder
        sequence_output = (
            decoder_outputs.last_hidden_state if hasattr(decoder_outputs, "last_hidden_state") 
            else decoder_outputs[0]
        )
        
        # Project to vocabulary
        logits = self.lm_head(sequence_output)
        
        # Prepare outputs based on return_dict
        if not return_dict:
            outputs = (logits,)
            # Add cache for decoder-based models
            if use_cache:
                outputs += (
                    decoder_outputs.past_key_values if hasattr(decoder_outputs, "past_key_values") 
                    else decoder_outputs[1],
                )
            # Add encoder outputs for encoder-decoder models
            if self.is_encoder_decoder:
                outputs += (encoder_outputs,)
            # Add attention and hidden states if requested
            if output_hidden_states or output_attentions:
                # TODO: Add proper handling of attention and hidden states
                pass
            return outputs
        
        return {
            "logits": logits,
            "past_key_values": decoder_outputs.past_key_values if hasattr(decoder_outputs, "past_key_values") else None,
            "encoder_outputs": encoder_outputs if self.is_encoder_decoder else None,
            # TODO: Add other outputs like attentions and hidden states when requested
        }
    
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        max_length: int = 20,
        min_length: int = 0,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        do_sample: bool = False,
        num_return_sequences: int = 1,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """
        Generate sequences for models with a language modeling head.
        
        This is a simplified version of generation - for more advanced options,
        consider using the transformers library's generation utilities.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_length: Maximum length of generated sequences
            min_length: Minimum length of generated sequences
            temperature: Sampling temperature
            top_k: Top k sampling
            top_p: Nucleus sampling probability
            repetition_penalty: Repetition penalty
            do_sample: Whether to use sampling instead of greedy decoding
            num_return_sequences: Number of sequences to generate
            pad_token_id: Padding token ID
            eos_token_id: End of sequence token ID
            
        Returns:
            Generated sequences
        """
        # Set default token IDs if not provided
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.padding_idx
        eos_token_id = eos_token_id if eos_token_id is not None else pad_token_id
        
        batch_size = input_ids.shape[0]
        
        # Prepare encoder outputs for encoder-decoder models
        if self.is_encoder_decoder:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            decoder_input_ids = torch.full(
                (batch_size, 1),
                pad_token_id,
                dtype=torch.long,
                device=input_ids.device,
            )
            
            # Store input_ids and attention_mask for encoder-decoder generation
            model_kwargs = {
                "encoder_outputs": encoder_outputs,
                "attention_mask": attention_mask,
            }
            current_ids = decoder_input_ids
        else:
            # For decoder-only models, we use the input_ids as the starting point
            model_kwargs = {}
            current_ids = input_ids
        
        # Store past key values for faster generation
        past_key_values = None
        
        # Track generated sequences
        generated_ids = current_ids
        
        # Main generation loop
        while generated_ids.shape[1] < max_length:
            # Forward pass
            if self.is_encoder_decoder:
                outputs = self(
                    decoder_input_ids=current_ids[:, -1:],  # Only use the last token for the decoder
                    use_cache=True,
                    past_key_values=past_key_values,
                    **model_kwargs,
                )
            else:
                outputs = self(
                    input_ids=current_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
            
            # Get logits and update past key values
            if isinstance(outputs, dict):
                logits = outputs["logits"]
                past_key_values = outputs.get("past_key_values")
            else:
                logits = outputs[0]
                past_key_values = outputs[1] if len(outputs) > 1 else None
            
            # Only use the logits corresponding to the last token
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_token in generated_ids[i]:
                        # If the token is already generated, decrease its probability
                        if previous_token.item() in generated_ids[i]:
                            next_token_logits[i, previous_token] /= repetition_penalty
            
            # Mask tokens that shouldn't be generated (e.g., before min_length)
            if min_length > 0 and generated_ids.shape[1] < min_length:
                # Prevent EOS token generation if we're below min_length
                if eos_token_id is not None:
                    next_token_logits[:, eos_token_id] = -float("inf")
            
            # Sample or select next token
            if do_sample:
                # Top-k sampling
                if top_k > 0:
                    top_k_values, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                    next_token_logits = torch.full_like(next_token_logits, -float("inf"))
                    next_token_logits.scatter_(-1, top_k_indices, top_k_values)
                
                # Top-p (nucleus) sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep the first token above threshold
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    
                    # Apply filtering
                    for i in range(batch_size):
                        indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                        next_token_logits[i, indices_to_remove] = -float("inf")
                
                # Convert to probabilities and sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to generated ids
            generated_ids = torch.cat([generated_ids, next_tokens], dim=1)
            
            # For encoder-decoder models, update current_ids to just the new token
            if self.is_encoder_decoder:
                current_ids = next_tokens
            else:
                # For decoder-only models, update attention mask to include the new token
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [attention_mask, attention_mask.new_ones((batch_size, 1))], dim=1
                    )
                current_ids = generated_ids
            
            # Check if we've generated an EOS token
            if eos_token_id is not None and (next_tokens == eos_token_id).all():
                break
        
        return generated_ids
