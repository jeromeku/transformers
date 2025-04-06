import os
from contextlib import contextmanager

import torch
from bitsandbytes.nn import Linear4bit

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Llama4Config,
    Llama4ForConditionalGeneration,
    Llama4TextConfig,
    Llama4VisionConfig,
)
from transformers.quantizers import AutoHfQuantizer
from transformers.utils.logging import set_verbosity_info


def format_memory(memory):
    if memory < 1024:
        return f"{memory} B"
    elif memory < 1024 * 1024:
        return f"{memory / 1024:.2f} KB"
    elif memory < 1024 * 1024 * 1024:
        return f"{memory / 1024 / 1024:.2f} MB"
    else:
        return f"{memory / 1024 / 1024 / 1024:.2f} GB"


@contextmanager
def memory_context(description=""):
    before_reserved = torch.cuda.memory_reserved()
    before_max_r = torch.cuda.max_memory_reserved()
    delimiters = "=" * 100
    print(f"\n{delimiters} {description} {delimiters}\n")
    print(f"before_reserved: {format_memory(before_reserved)}, before_max_r: {format_memory(before_max_r)} ")
    yield
    after_reserved = torch.cuda.memory_reserved()
    after_max_r = torch.cuda.max_memory_reserved()
    print(f"after_reserved: {format_memory(after_reserved)}, after_max_r: {format_memory(after_max_r)}")
    print(f"\n{delimiters * 2}\n")


set_verbosity_info()
# model_name = "meta-llama/Llama-3.2-1B-Instruct"
model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    llm_int8_skip_modules=["language_model.model.layers.0.feed_forward.router"],
)

debug_text_config = Llama4TextConfig(
    num_hidden_layers=1,
)

vision_config = Llama4VisionConfig()
config = Llama4Config(text_config=debug_text_config, vision_config=vision_config)
quantizer = AutoHfQuantizer.from_config(bnb_config)

if not os.path.exists("llama4-scout-17b-16e-instruct-debug"):
    with memory_context("Llama4ForConditionalGeneration"):
        model = Llama4ForConditionalGeneration(config).to("cuda")
    # print(model)
    model.save_pretrained("llama4-scout-17b-16e-instruct-debug")

with memory_context("Llama4ForConditionalGeneration from_pretrained"):
    model = Llama4ForConditionalGeneration.from_pretrained(
        "llama4-scout-17b-16e-instruct-debug",
        torch_dtype=torch.bfloat16,
        # quantization_config=bnb_config,
        attn_implementation="sdpa",
        device_map="auto",
    )

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)

outputs = model.generate(**inputs.to(model.device), max_new_tokens=1)
print(outputs)
# for name, module in model.named_modules():
#     if "router" in name:
#         print(name, type(module).__name__)
#     if isinstance(module, Linear4bit):
#         w = module.weight
#         quant_state = module.quant_state
