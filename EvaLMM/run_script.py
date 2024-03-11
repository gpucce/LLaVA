from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

model_path = "/leonardo_scratch/large/userexternal/gpuccett/models/hf_llava/llava-v1.6-mistral-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)
