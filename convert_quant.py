import os, shutil
shutil.rmtree("aaa",ignore_errors=True)
os.mkdir("aaa")
print("aaa was created")
import torch
from transformers import pipeline
from transformers import convert_graph_to_onnx, GPTNeoForCausalLM, AutoConfig, AutoModelForCausalLM, AutoTokenizer
from torchinfo import summary

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor

# Set default QuantDescriptor to use histogram based calibration for activation
quant_desc_input = QuantDescriptor(calib_method='histogram')
quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

# Now new models will automatically have QuantConv2d layers instead of regular Conv2d
from pytorch_quantization import quant_modules
quant_modules.initialize()

#model_name='EleutherAI/gpt-neo-125M'
model_name='EleutherAI/gpt-neo-1.3B'
#model_name='EleutherAI/gpt-neo-2.7B'

device = torch.device("cuda:0")
print("Getting model", model_name)
model=AutoModelForCausalLM.from_pretrained(model_name).to(device)
model = model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)
prompt = "Germany is"
tokenized_input = tokenizer(prompt) #[27079, 318]
print(prompt, tokenized_input)
input_ids = torch.LongTensor([tokenized_input["input_ids"]]).to(device)
attention_mask = torch.Tensor([tokenized_input["attention_mask"]]).to(device)

print("Loaded")
@torch.no_grad()
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

m2=CausalLMWrapper(model).to(device)
m2=m2.eval()

#summary(m2, input_data={"input_ids": input_ids, "attention_mask": attention_mask}, depth=8, device=device)

print("Running")
with torch.no_grad():
    #out = m2(input_ids, attention_mask)
    #print("Model run done. out shape:", out.shape)
    #print("out dtype:", out.dtype)

    input_names=["input_ids", "attention_mask"]
    output_names=["output0"]
    dynamic_axes = {'input_ids': {0: 'batch', 1: 'token'}, 'attention_mask':{0:'batch', 1:'token'}, 'output0':{0: 'batch', 1:'token'}}

    print("ONNX Exporting...")
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    onnx_file = model_name.split("/")[-1].lower()
    torch.onnx.export(
        m2,
        (input_ids, attention_mask),
        f"aaa/{onnx_file}_fakequant.onnx",
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    print("================ ONNX Export Done! =======================")
