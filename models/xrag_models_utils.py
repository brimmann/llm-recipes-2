from transformers import AutoConfig, AutoTokenizer
from models.modeling_xmistral import XMistralForCausalLM
import torch
from models.modeling_xgemma import XGemmaForCausalLM, XGemmaConfig
from models.distillation_model import DistillationModelXrag

XRAG_TOKEN = "<xRAG>"

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

    teacher_model.set_xrag_token_id(teacher_tokenizer.convert_tokens_to_ids(XRAG_TOKEN))
    
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

    # Free all model except projector part
    for param in student_model.parameters():
        param.requires_grad = False

    for param in student_model.projector.parameters():
        param.requires_grad = True

    print("Trainable parameters in student model:")
    for name, param in student_model.named_parameters():
        if param.requires_grad:
            print(name)

    student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    student_tokenizer.add_special_tokens({"additional_special_tokens": [XRAG_TOKEN]})
    
    student_model.resize_token_embeddings(len(student_tokenizer))

    student_model.set_xrag_token_id(student_tokenizer.convert_tokens_to_ids(XRAG_TOKEN))
    
    return student_model, student_tokenizer

    
    
def get_xrag_models():
    teacher_model, teacher_tokenizer = get_teacher_model()
    student_model, student_tokenizer = get_student_model()
    return student_tokenizer, teacher_tokenizer, DistillationModelXrag(student_model, teacher_model)



