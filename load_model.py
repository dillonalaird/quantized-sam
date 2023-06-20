import torch
import torch.nn as nn

from typing import Tuple, Dict, Callable
from sam.build_sam import sam_model_registry
from transformers import BitsAndBytesConfig
from transformers.utils.bitsandbytes import (
    replace_with_bnb_linear,
    set_module_quantized_tensor_to_device,
)


def load_params(model: nn.Module, state_dict: Dict):
    for param_name, param in state_dict.items():
        set_module_quantized_tensor_to_device(
            model,
            param_name,
            0,
            value=param,
            fp16_statistics=None,
        )


def warmup_model(model: nn.Module, input_shape: Tuple[int, ...]):
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        dummy_input = torch.randn(input_shape, dtype=torch.float16)
        dummy_input = dummy_input.to(torch.device("cuda", 0))
        model(dummy_input)
        model(dummy_input)
        model(dummy_input)


def load_quantized_model(
    model_type: str, model_path: str, quantization_config: BitsAndBytesConfig
) -> Tuple[nn.Module, Callable]:
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        model, preprocess = sam_model_registry[model_type]()
        model = replace_with_bnb_linear(model, quantization_config=quantization_config)
        state_dict = torch.load(model_path)
        load_params(model, state_dict)

        warmup_model(model, (1, 3, 1024, 1024))

    return model, preprocess
