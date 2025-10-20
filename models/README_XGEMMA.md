# XGemma: Gemma with Retrieval Augmentation

## Overview

XGemmaForCausalLM is an extended version of Google's Gemma model that supports retrieval-augmented generation (RAG) through special token replacement. This implementation is compatible with Gemma-2 1B and larger models.

## Key Features

- **Seamless RAG Integration**: Replace special `<xRAG>` tokens with retrieval embeddings
- **Flexible Projector**: Configurable MLP projector to map retrieval embeddings to model's hidden size
- **Backward Compatible**: Works as standard Gemma when retriever_hidden_size is 0
- **Training & Generation Support**: Full support for both training and inference with retrieval

## Model Architecture

The model extends `GemmaForCausalLM` with:
1. A projector module to transform retrieval embeddings
2. Special token replacement mechanism
3. Custom forward and generate methods

## Configuration

### XGemmaConfig Parameters

- All standard `GemmaConfig` parameters
- `projector_type` (str): Type of projector, default 'mlp2x_gelu'
- `retriever_hidden_size` (int): Size of retrieval embeddings, default 128

### Gemma-2 1B Typical Configuration

```python
config = XGemmaConfig(
    vocab_size=256000,          # Gemma-2 vocabulary
    hidden_size=2048,           # Model hidden dimension
    intermediate_size=16384,    # FFN intermediate size
    num_hidden_layers=18,       # Number of layers
    num_attention_heads=8,      # Attention heads
    num_key_value_heads=1,      # GQA key-value heads
    head_dim=256,              # Head dimension
    max_position_embeddings=8192,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    attention_bias=False,
    # XGemma specific
    projector_type='mlp2x_gelu',
    retriever_hidden_size=128,
)
```

## Usage

### Training with Retrieval Embeddings

```python
from modeling_xgemma import XGemmaForCausalLM, XGemmaConfig

# Initialize model
config = XGemmaConfig.from_pretrained("google/gemma-2-2b")
config.retriever_hidden_size = 128
config.projector_type = 'mlp2x_gelu'

model = XGemmaForCausalLM(config)
model.set_xrag_token_id(32001)  # Set your special token ID

# Training forward pass
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    retrieval_embeds=retrieval_embeds,  # Shape: [batch, retriever_hidden_size]
    labels=labels,  # With -100 for prompt tokens
)

loss = outputs.loss
```

### Generation with Retrieval

```python
# Generate with retrieval augmentation
generated_ids = model.generate(
    input_ids=prompt_ids,  # Contains <xRAG> token
    retrieval_embeds=retrieval_embeds,
    max_new_tokens=100,
    temperature=0.7,
    do_sample=True,
)
```

## Data Format

Your data loader should produce batches with:
- `input_ids`: Token IDs including special `<xRAG>` token
- `attention_mask`: Standard attention mask
- `retrieval_embeds`: Retrieval embeddings (shape: [batch, retriever_hidden_size])
- `labels`: Token IDs with -100 for non-target tokens

## Differences from XMistral

While maintaining the same interface, XGemma adapts to Gemma-specific features:
- Uses RMSNorm instead of LayerNorm
- Supports Grouped Query Attention (GQA)
- Compatible with RoPE positional embeddings
- Handles Gemma's 256k vocabulary

## Installation Requirements

```bash
pip install torch>=2.0.0
pip install transformers>=4.36.0
```

## Model Compatibility

- Gemma-2 1B (google/gemma-2-2b-it)
- Gemma-2 9B (google/gemma-2-9b-it)
- Gemma-2 27B (google/gemma-2-27b-it)

## Notes

1. The model automatically handles the projection of retrieval embeddings to match the model's hidden size
2. During generation, retrieval embeddings are only used in the first forward pass
3. The number of `<xRAG>` tokens must match the number of retrieval embeddings
4. When `retriever_hidden_size=0`, the model behaves as standard Gemma

## Citation

If you use XGemma in your research, please cite:
- The original Gemma paper
- Your retrieval augmentation method
- This implementation
