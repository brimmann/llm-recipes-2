from datasets import load_dataset

def tokenize_xrag(item, tokenizer):
    if tokenizer.name_or_path == "Hannibal046/xrag-7b":
        prompt = "[INST] Background: <xRAG>, which also means:[/INST]"
    elif tokenizer.name_or_path == "google/gemma-3-1b-it":
        prompt = "<start_of_turn>user\nBackground: <xRAG>, which also means:<end_of_turn>"
    

    context_tokens = tokenizer.encode(f"{prompt}", add_special_tokens=False)
    answer_tokens = tokenizer.encode(f"{item['generated_text']}", add_special_tokens=False)

    prompt_tokens = context_tokens+answer_tokens
    # print("len(prompt_tokens)", len(prompt_tokens))
    labels_tokens = (len(context_tokens)*[-100,])+answer_tokens
    # print("len(labels_tokens)", len(labels_tokens))

    combined_tokens = {
                "input_ids": prompt_tokens,
                "labels": labels_tokens
    }
    return dict(combined_tokens, attention_mask=[1]*len(combined_tokens["input_ids"]), retrieval_embeds=item["embeddings"])

def get_split(dataset_config, tokenizer, split):
    dataset = load_dataset("brimmann2/squad-v2-sampled", split=split)
    dataset = dataset.map(lambda item: tokenize_xrag(item, tokenizer), remove_columns=list(dataset.features))
    return dataset