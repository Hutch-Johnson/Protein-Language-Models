import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, EsmForMaskedLM
from tokenizers import Tokenizer
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def initialize_progen2_noeval(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    from transformers.modeling_utils import PreTrainedModel

    _original_adjust = PreTrainedModel._adjust_tied_keys_with_tied_pointers

    def safe_adjust(self, *args, **kwargs):
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
        if not hasattr(self, "_tied_weights_keys"):
            self._tied_weights_keys = {}
        return _original_adjust(self, *args, **kwargs)

    PreTrainedModel._adjust_tied_keys_with_tied_pointers = safe_adjust
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=False
        # device_map={"": "cpu"}
    )
    for name, module in model.named_modules():
        if hasattr(module, 'scale_attn') and isinstance(module.scale_attn, torch.Tensor):
            print(f"{name}: scale_attn device={module.scale_attn.device}, value={module.scale_attn}")
    # Force ALL tensors/buffers to CPU
    model = model.to("cpu")

    # Also move any remaining meta tensors explicitly
    for name, param in list(model.named_parameters()):
        if param.device.type == "meta":
            print(f"Moving {name} from meta to cpu")
            param.data = torch.empty(param.shape, dtype=param.dtype, device="cpu")

    for name, buf in list(model.named_buffers()):
        if buf.device.type == "meta":
            print(f"Moving buffer {name} from meta to cpu")
            # find the module and replace the buffer
            parts = name.split(".")
            module = model
            for p in parts[:-1]:
                module = getattr(module, p)
            module.register_buffer(parts[-1], torch.zeros(buf.shape, dtype=buf.dtype, device="cpu"))
    
    # scale_attn is an attribute, not a registered buffer, so it escapes the above loop
    for module in model.modules():
        if hasattr(module, 'scale_attn') and isinstance(module.scale_attn, torch.Tensor):
            if module.scale_attn.device.type == 'meta':
                # print("Fixing scale_attn on meta device")
                shape = module.scale_attn.shape
                dtype = module.scale_attn.dtype
                module.scale_attn = torch.ones(shape, dtype=dtype, device='cpu')
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     trust_remote_code=True,
    #     low_cpu_mem_usage=False  # force weights onto cpu, not meta device
    # )
    # model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    PreTrainedModel._adjust_tied_keys_with_tied_pointers = _original_adjust

    # Patch get_head_mask on BOTH the outer and inner model
    import types

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers
        return head_mask

    # Patch outer ProGenForCausalLM
    if not hasattr(model, "get_head_mask"):
        model.get_head_mask = types.MethodType(get_head_mask, model)

    # Patch inner ProGenModel (model.transformer)
    if not hasattr(model.transformer, "get_head_mask"):
        model.transformer.get_head_mask = types.MethodType(get_head_mask, model.transformer)

    return model, tokenizer

model_name = "hugohrban/progen2-medium"

initialize_progen2_noeval(model_name)