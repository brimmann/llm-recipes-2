"""
Example usage of XGemmaForCausalLM with retrieval augmentation.

This script demonstrates:
1. Loading the model with custom configuration
2. Setting up the xRAG token
3. Training with retrieval embeddings
4. Generating text with retrieval augmentation
"""

import torch
from transformers import AutoTokenizer
from modeling_xgemma import XGemmaForCausalLM, XGemmaConfig


def example_training():
    """Example of training XGemma with retrieval embeddings."""
    
    # Initialize configuration
    # For Gemma-2 1B, typical configuration would be:
    config = XGemmaConfig(
        vocab_size=256000,  # Gemma-2 uses 256k vocabulary
        hidden_size=2048,   # Hidden size for 1B model
        intermediate_size=16384,  # FFN intermediate size
        num_hidden_layers=18,  # Number of transformer layers
        num_attention_heads=8,  # Number of attention heads
        num_key_value_heads=1,  # GQA: number of key-value heads
        head_dim=256,  # Dimension per attention head
        max_position_embeddings=8192,  # Context length
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        # Custom XGemma parameters
        projector_type='mlp2x_gelu',
        retriever_hidden_size=128,
    )
    
    # Initialize model
    model = XGemmaForCausalLM(config)
    
    # Initialize tokenizer (you would load from pretrained in practice)
    # tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    
    # Set the special xRAG token ID (assuming 32001 as in your example)
    model.set_xrag_token_id(32001)
    
    # Example training batch (matching your data format)
    batch = {
        'input_ids': torch.tensor([[
            733, 16289, 28793, 24316, 28747, 32001, 28725, 690, 835, 2825,
            28747, 28792, 28748, 16289, 28793, 415, 2990, 302, 9143, 403,
            16783, 23799, 356, 4117, 28705, 28740, 28787, 28725, 28705, 28740,
            28787, 28750, 28750, 28725, 304, 403, 10806, 1987, 9143, 8897
        ]]),
        'attention_mask': torch.ones(1, 40),  # Simplified for example
        'retrieval_embeds': torch.randn(1, 128),  # Random retrieval embedding
        'labels': torch.tensor([[
            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, 415, 2990, 302, 9143, 403,
            16783, 23799, 356, 4117, 28705, 28740, 28787, 28725, 28705, 28740,
            28787, 28750, 28750, 28725, 304, 403, 10806, 1987, 9143, 8897
        ]]),
    }
    
    # Forward pass for training
    outputs = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        retrieval_embeds=batch['retrieval_embeds'],
        labels=batch['labels'],
    )
    
    loss = outputs.loss
    print(f"Training loss: {loss.item():.4f}")
    
    # Backward pass would follow in actual training
    # loss.backward()
    # optimizer.step()
    
    return model


def example_generation():
    """Example of text generation with retrieval augmentation."""
    
    # Load model (in practice, you'd load from checkpoint)
    config = XGemmaConfig(
        vocab_size=256000,
        hidden_size=2048,
        intermediate_size=16384,
        num_hidden_layers=18,
        num_attention_heads=8,
        num_key_value_heads=1,
        head_dim=256,
        projector_type='mlp2x_gelu',
        retriever_hidden_size=128,
    )
    
    model = XGemmaForCausalLM(config)
    model.set_xrag_token_id(32001)
    model.eval()
    
    # Example prompt with xRAG token
    input_ids = torch.tensor([[
        733, 16289, 28793, 24316, 28747, 32001, 28725, 690, 835, 2825,
        28747, 28792, 28748, 16289, 28793
    ]])
    
    # Retrieval embedding (would come from your retriever)
    retrieval_embeds = torch.randn(1, 128)
    
    # Generate with retrieval augmentation
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            retrieval_embeds=retrieval_embeds,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )
    
    print(f"Generated token IDs shape: {generated_ids.shape}")
    
    return generated_ids


def example_without_retrieval():
    """Example of using the model without retrieval (standard Gemma mode)."""
    
    config = XGemmaConfig(
        vocab_size=256000,
        hidden_size=2048,
        intermediate_size=16384,
        num_hidden_layers=18,
        num_attention_heads=8,
        num_key_value_heads=1,
        head_dim=256,
        retriever_hidden_size=0,  # No retriever
    )
    
    # This will work as a standard GemmaForCausalLM
    model = XGemmaForCausalLM(config)
    model.eval()
    
    # Standard input without xRAG tokens
    input_ids = torch.tensor([[
        733, 16289, 28793, 415, 2990, 302, 9143, 28792, 28748, 16289, 28793
    ]])
    
    # Generate without retrieval
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=30,
            temperature=0.7,
            do_sample=True,
        )
    
    print(f"Standard generation shape: {generated_ids.shape}")
    
    return generated_ids


if __name__ == "__main__":
    print("=" * 50)
    print("XGemma Model Examples")
    print("=" * 50)
    
    print("\n1. Training with retrieval embeddings:")
    model = example_training()
    
    print("\n2. Generation with retrieval augmentation:")
    generated = example_generation()
    
    print("\n3. Standard generation without retrieval:")
    standard_generated = example_without_retrieval()
    
    print("\n✅ All examples completed successfully!")
