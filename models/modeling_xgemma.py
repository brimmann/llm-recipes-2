import torch
import torch.nn as nn
import re
from transformers import GemmaForCausalLM, GemmaConfig
from typing import Optional, Union, List, Tuple


class XGemmaConfig(GemmaConfig):
    """
    Configuration class for XGemma model that extends GemmaConfig
    with additional parameters for retrieval embedding projection.
    """
    def __init__(
        self,
        projector_type='mlp2x_gelu',
        retriever_hidden_size=128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.projector_type = projector_type
        self.retriever_hidden_size = retriever_hidden_size


class Projector(nn.Module):
    """
    Projector module to map retrieval embeddings to model's hidden size.
    Supports configurable MLP depth with GELU activation.
    """
    def __init__(self, config):
        super().__init__()
        projector_type = config.projector_type
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(config.retriever_hidden_size, config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            self.projector = nn.Sequential(*modules)
        else:
            raise ValueError(f"Unsupported projector_type: {projector_type}")
    
    def forward(self, context_embedding):
        return self.projector(context_embedding)


class XGemmaForCausalLM(GemmaForCausalLM):
    """
    Extended Gemma model for causal language modeling with retrieval augmentation.
    Compatible with standard Gemma models while supporting retrieval embeddings.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Initialize projector if retriever_hidden_size is specified
        if hasattr(config, "retriever_hidden_size") and config.retriever_hidden_size > 0:
            self.projector = Projector(config)
            self.retriever_hidden_size = config.retriever_hidden_size
        else:
            self.projector = None
            self.retriever_hidden_size = None
            
        # Initialize xrag_token_id as None (will be set later)
        self.xrag_token_id = None
        
        # Post initialization
        self.post_init()
    
    def set_xrag_token_id(self, token_id: int):
        """Set the special token ID used for retrieval augmentation."""
        self.xrag_token_id = token_id
    
    def prepare_inputs_embeds(
        self, 
        input_ids: torch.Tensor, 
        retrieval_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Prepare input embeddings by replacing xRAG tokens with projected retrieval embeddings.
        
        Args:
            input_ids: Input token IDs tensor
            retrieval_embeds: Retrieval embeddings tensor
            
        Returns:
            Modified input embeddings tensor
        """
        # Get standard word embeddings
        inputs_embeds = self.model.embed_tokens(input_ids)
        
        # Reshape retrieval embeddings
        retrieval_embeds = retrieval_embeds.view(-1, self.retriever_hidden_size)
        
        # Sanity check: ensure number of xRAG tokens matches retrieval embeddings
        num_xrag_tokens = torch.sum(input_ids == self.xrag_token_id).item()
        num_retrieval_embeds = retrieval_embeds.shape[0]
        assert num_xrag_tokens == num_retrieval_embeds, \
            f"Mismatch: {num_xrag_tokens} xRAG tokens vs {num_retrieval_embeds} retrieval embeddings"
        
        # Project retrieval embeddings to model's hidden size
        retrieval_embeds = self.projector(retrieval_embeds.to(inputs_embeds.dtype))
        
        # Replace xRAG token embeddings with retrieval embeddings
        inputs_embeds[input_ids == self.xrag_token_id] = retrieval_embeds
        
        return inputs_embeds
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        retrieval_embeds: Optional[torch.FloatTensor] = None,  # Shape: [-1, retriever_hidden_size]
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Forward pass with optional retrieval augmentation.
        
        When retrieval_embeds is provided, xRAG tokens in input_ids are replaced
        with projected retrieval embeddings before processing.
        """
        # Check if we're at the beginning of generation
        # (inputs_embeds is passed during generation after the first step)
        at_the_beginning_of_generation = False
        if inputs_embeds is not None and retrieval_embeds is None:
            # During generation, after first step, inputs_embeds is passed
            assert not self.training, "inputs_embeds without retrieval_embeds only supported during generation"
            at_the_beginning_of_generation = True
        
        # Process retrieval embeddings if not in generation mode
        if not at_the_beginning_of_generation:
            if retrieval_embeds is not None:
                # Prepare embeddings with retrieval augmentation
                inputs_embeds = self.prepare_inputs_embeds(input_ids, retrieval_embeds)
                input_ids = None  # Clear input_ids to prevent double embedding
                
                # Verify attention mask shape matches
                if attention_mask is not None:
                    assert inputs_embeds.shape[1] == attention_mask.shape[1], \
                        f"Shape mismatch: inputs_embeds {inputs_embeds.shape} vs attention_mask {attention_mask.shape}"
        
        # Call parent's forward method
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        retrieval_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Generate text with optional retrieval augmentation.
        
        Args:
            input_ids: Input token IDs
            retrieval_embeds: Optional retrieval embeddings for xRAG tokens
            attention_mask: Attention mask
            **kwargs: Additional generation arguments
            
        Returns:
            Generated token IDs
        """
        # Ensure inputs_embeds is not passed directly (not supported)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not directly supported for generate. Use retrieval_embeds instead.")
        
        # Process retrieval embeddings if provided
        if retrieval_embeds is not None:
            # Prepare embeddings with retrieval augmentation
            inputs_embeds = self.prepare_inputs_embeds(input_ids, retrieval_embeds)
            
            # Verify attention mask shape
            if attention_mask is not None:
                assert inputs_embeds.shape[1] == attention_mask.shape[1], \
                    f"Shape mismatch: inputs_embeds {inputs_embeds.shape} vs attention_mask {attention_mask.shape}"
            
            # Generate with prepared embeddings
            return super().generate(
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs
            )
        else:
            # Standard generation without retrieval augmentation
            return super().generate(
                attention_mask=attention_mask,
                input_ids=input_ids,
                **kwargs
            )
