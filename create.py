import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

cfg_file="config-20b-40.json"
config = GPTNeoXConfig.from_json_file(cfg_file)
print("Making")
model = GPTNeoXForCausalLM(config)
n_params = count_parameters(model)
print("n_params:",f"{n_params:,}")
print("Saving")
model.save_pretrained("gpt-neox-20b-1")
print("Done")
