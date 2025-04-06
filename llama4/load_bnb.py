import os
from contextlib import contextmanager

import torch
from bitsandbytes.functional import QuantState
from bitsandbytes.nn import Linear4bit, Params4bit

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Llama4Config,
    Llama4ForConditionalGeneration,
    Llama4TextConfig,
    Llama4VisionConfig,
)
from transformers.integrations.bitsandbytes import replace_with_bnb_linear
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


# set_verbosity_info()
# model_name = "meta-llama/Llama-3.2-1B-Instruct"
VISION_MODULES_SKIP_PATTERNS = ["embedding", "projector"]
LANGUAGE_MODULES_SKIP_PATTERNS = ["router", "lm_head"]
MODULES_SKIP_PATTERNS = VISION_MODULES_SKIP_PATTERNS + LANGUAGE_MODULES_SKIP_PATTERNS
DEVICE = "cuda"
DTYPE = torch.bfloat16
model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    llm_int8_skip_modules=["language_model.model.layers.0.feed_forward.router"],
)

text_config = Llama4TextConfig(
    num_hidden_layers=1,
)
vision_config = Llama4VisionConfig(num_hidden_layers=1)
config = Llama4Config(text_config=text_config, vision_config=vision_config)

quantizer = AutoHfQuantizer.from_config(bnb_config)

if not os.path.exists("llama4-scout-17b-16e-instruct-debug"):
    with memory_context("Llama4ForConditionalGeneration"):
        model = Llama4ForConditionalGeneration(config).to("cuda")
    model.save_pretrained("llama4-scout-17b-16e-instruct-debug")

    torch.save(model.state_dict(), "llama4-scout-17b-16e-instruct-debug/model.pt")

with memory_context("Llama4ForConditionalGeneration from_pretrained"), torch.device("meta"):
    # model = Llama4ForConditionalGeneration.from_pretrained(
    #     "llama4-scout-17b-16e-instruct-debug",
    #     torch_dtype=torch.bfloat16,
    #     # quantization_config=bnb_config,
    #     attn_implementation="sdpa",
    #     device_map="auto",
    # )
    model = Llama4ForConditionalGeneration(config)

    param_names = [name for name, _ in model.named_parameters()]
    module_names = [name for name, _ in model.named_modules()]
    modules_to_skip = list(filter(lambda x: any(pattern in x for pattern in MODULES_SKIP_PATTERNS), module_names))
    print(modules_to_skip)

    replace_with_bnb_linear(model, modules_to_not_convert=modules_to_skip, quantization_config=bnb_config)
    for name, module in model.named_modules():
        print(f"{name}: {type(module).__name__}")
    for name, param in model.named_parameters():
        if isinstance(param, Params4bit):
            print(f"{name}: {param.device} {param.dtype} {param.shape}")

model_state_dict = {}

with memory_context("Load params"):
    # Load params
    loaded_state_dict = torch.load(
        "llama4-scout-17b-16e-instruct-debug/model.pt", map_location="cpu", weights_only=True, mmap=True
    )
    for name, param in loaded_state_dict.items():
        assert name in param_names, f"{name} not in {param_names}"
        model_param = model.get_parameter(name)
        param = param.to(device=DEVICE, dtype=DTYPE)
        # if isinstance(model_param, Params4bit):
        #     model_state_dict[name] = Params4bit(param, quant_type="nf4", compress_statistics=True)
        # else:
        model_state_dict[name] = param

    model.load_state_dict(model_state_dict, assign=True)

# Need to do a second pass to convert Linear4bit params back to Params4bit
for name, module in model.named_modules():
    if isinstance(module, Linear4bit):
        print(f"{name}: {module.weight.device} {module.weight.dtype} {module.weight.shape}")
        module.weight = Params4bit(module.weight, quant_type="nf4", compress_statistics=True).to(DEVICE)


for name, param in model.named_parameters():
    print(f"{name} is Params4bit {isinstance(param, Params4bit)}: {param.device} {param.dtype} {param.shape}")
    if isinstance(param, Params4bit):
        param.to(DEVICE)
        quant_state: QuantState = param.quant_state
        print(f" ->: {param.device} {quant_state.dtype} {quant_state.shape} {hasattr(quant_state, 'state2')}")

    # text_model = model.language_model
# vision_model = model.vision_model

# text_model_size = sum(p.numel() * p.element_size() for p in text_model.parameters())
# vision_model_size = sum(p.numel() * p.element_size() for p in vision_model.parameters())
# print(f"text_model_size: {format_memory(text_model_size)}, vision_model_size: {format_memory(vision_model_size)}")

# for name, param in model.named_parameters():
#     if "router" in name or "expert" in name:
#         print(f"{name}: {param.device} {param.dtype} {param.shape}")
# # tokenizer = AutoTokenizer.from_pretrained(model_id)

# messages = [
#     {"role": "user", "content": "Who are you?"},
# ]
# inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)

# outputs = model.generate(**inputs.to(model.device), max_new_tokens=1)
# print(outputs)
# # for name, module in model.named_modules():
# #     if "router" in name:
# #         print(name, type(module).__name__)
# #     if isinstance(module, Linear4bit):
# #         w = module.weight
#         quant_state = module.quant_state
