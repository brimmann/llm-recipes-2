from transformers import AutoConfig, AutoTokenizer
from models.modeling_xmistral import XMistralForCausalLM
import torch
from models.modeling_xgemma import XGemmaForCausalLM, XGemmaConfig

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_teacher_model():
    teacher_model_name = "Hannibal046/xrag-7b"
    
    config = AutoConfig.from_pretrained(teacher_model_name)
    MODEL_CLASS = eval(config.architectures[0])
    teacher_model = MODEL_CLASS.from_pretrained(
        teacher_model_name,
        torch_dtype = torch.bfloat16,
        low_cpu_mem_usage = True,
    ).to(device)

    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    return teacher_model, teacher_tokenizer


def get_student_model():
    student_model_name = "google/gemma-3-1b-it"
    pretrained_config = AutoConfig.from_pretrained(student_model_name)

    config = XGemmaConfig(
    **pretrained_config.to_dict(),
    projector_type='mlp2x_gelu',
    retriever_hidden_size=4096,
    )

    student_model = XGemmaForCausalLM.from_pretrained(student_model_name, config=config).to(device)

    student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    student_tokenizer.add_special_tokens({"additional_special_tokens": ["<xRAG>"]})
    
    student_model.resize_token_embeddings(len(student_tokenizer))
    
    return student_model, student_tokenizer

    
    




