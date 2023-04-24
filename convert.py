import os, shutil
shutil.rmtree("aaa",ignore_errors=True)
os.mkdir("aaa")
print("aaa was created")
import torch
import torchinfo
from transformers import pipeline, set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer
set_seed(42)
#device="cuda"
device="cpu"
load_in_8bit=False
#model_name='gpt2'
#model_name="EleutherAI/gpt-neo-1.3B"
model_name="EleutherAI/gpt-neo-2.7B"
#model_name="EleutherAI/gpt-neo-125M"
#model_name="EleutherAI/gpt-neox-20b"
#model_name="gpt-neox-20b-1"
#model_name="gpt-neox-20b-41"
#model_name="gpt-neox-20b-16"
model=AutoModelForCausalLM.from_pretrained(model_name)
#tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loaded")
class CausalLMWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids,
          attention_mask=attention_mask,
          use_cache=False,
          return_dict = False
        )
        res = out[0]
        return res

m2=CausalLMWrapper(model)

input_ids = torch.tensor([[2233, 318, 213, 43]], dtype=torch.int64).to(device)
attention_mask = torch.tensor([[1,1,1,1]]).to(device)

print("Running")
out = m2(input_ids, attention_mask)

print("Model run done. out shape:", out.shape)


if 0: 
  input_size = 100
  input_tokens = torch.LongTensor([[42]*input_size])
  torchinfo.summary(model, input_data={"input_ids":input_tokens})
  print()
  torchinfo.summary(model, input_data={"input_ids":input_tokens}, depth=9)
  exit(0)

input_names=["input_ids", "attention_mask"]
output_names=["output0"]
dynamic_axes = {'input_ids': {0: 'batch', 1: 'token'}, 'attention_mask':{0:'batch', 1:'token'}, 'output0':{0: 'batch', 1:'token'}}

print("ONNX Exporting...")
onnx_file = model_name.split("/")[-1].lower()
torch.onnx.export(
    m2,
    (input_ids, attention_mask),
    f"aaa/{onnx_file}.onnx",
    verbose=True,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes,
)
print("================ ONNX Export Done! =======================")
