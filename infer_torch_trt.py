text="""One morning, when Gregor Samsa woke from troubled dreams, he found himself transformed in his bed into a horrible vermin. He lay on his armour-like back, and if he lifted his head a little he could see his brown belly, slightly domed and divided by arches into stiff sections. The bedding was hardly able to cover it and seemed ready to slide off any moment. His many legs, pitifully thin compared with the size of the rest of him, waved about helplessly as he looked. "What's happened to me?" he thought. It wasn't a dream. His room, a proper human room although a little too small, lay peacefully between its four familiar walls. A collection of textile samples lay spread out on the table - Samsa was a travelling salesman - and above it there hung a picture that he had recently cut out of an illustrated magazine and housed in a nice, gilded frame. It showed a lady fitted out with a fur hat and fur boa who sat upright, raising a heavy fur muff that covered the whole of her lower arm towards the viewer. Gregor then turned to look out the window at the dull weather. Drops of rain could be heard hitting the pane, which made him feel quite sad. "How about if I sleep a little bit longer and forget all this nonsense", he thought, but that was something he was unable to do because he was used to sleeping on his right, and in his present state couldn't get into"""

tokens=[3198, 3329, 11, 618, 8547, 273, 3409, 11400, 19092, 422, 17840, 10625, 11, 339, 1043, 2241, 14434, 287, 465, 3996, 656, 257, 12361, 3326, 1084, 13, 679, 3830, 319, 465, 18588, 12, 2339, 736, 11, 290, 611, 339, 13663, 465, 1182, 257, 1310, 339, 714, 766, 465, 7586, 19921, 11, 4622, 2401, 276, 290, 9086, 416, 610, 2052, 656, 15175, 9004, 13, 383, 3996, 12083, 373, 8941, 1498, 284, 3002, 340, 290, 3947, 3492, 284, 10649, 572, 597, 2589, 13, 2399, 867, 7405, 11, 6028, 17049, 7888, 3688, 351, 262, 2546, 286, 262, 1334, 286, 683, 11, 26834, 546, 21144, 306, 355, 339, 3114, 13, 366, 2061, 338, 3022, 284, 502, 1701, 339, 1807, 13, 632, 2492, 470, 257, 4320, 13, 2399, 2119, 11, 257, 1774, 1692, 2119, 3584, 257, 1310, 1165, 1402, 11, 3830, 30996, 1022, 663, 1440, 5385, 7714, 13, 317, 4947, 286, 45293, 8405, 3830, 4104, 503, 319, 262, 3084, 532, 3409, 11400, 373, 257, 16574, 42414, 532, 290, 2029, 340, 612, 9174, 257, 4286, 326, 339, 550, 2904, 2005, 503, 286, 281, 18542, 7093, 290, 23707, 287, 257, 3621, 11, 308, 46158, 5739, 13, 632, 3751, 257, 10846, 18235, 503, 351, 257, 9230, 6877, 290, 9230, 1489, 64, 508, 3332, 24826, 11, 8620, 257, 4334, 9230, 27563, 326, 5017, 262, 2187, 286, 607, 2793, 3211, 3371, 262, 19091, 13, 8547, 273, 788, 2900, 284, 804, 503, 262, 4324, 379, 262, 19222, 6193, 13, 41692, 286, 6290, 714, 307, 2982, 9008, 262, 37218, 11, 543, 925, 683, 1254, 2407, 6507, 13, 366, 2437, 546, 611, 314, 3993, 257, 1310, 1643, 2392, 290, 6044, 477, 428, 18149, 1600, 339, 1807, 11, 475, 326, 373, 1223, 339, 373, 5906, 284, 466, 780, 339, 373, 973, 284, 11029, 319, 465, 826, 11, 290, 287, 465, 1944, 1181, 3521, 470, 651, 656]


import torch
import torch_tensorrt
#import torchinfo
import numpy as np
from scipy.special import softmax
np.set_printoptions(precision=5)
#import GPUtil
#torch.backends.cudnn.benchmark = True
#torch.backends.cudnn.enabled = True
from transformers import pipeline, set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer
set_seed(42)
device="cpu"
load_in_fp16=False
load_in_8bit=False
#load_in_8bit=True
#model_name='gpt2'
#model_name="EleutherAI/gpt-neo-2.7B"
#model_name="EleutherAI/gpt-neo-1.3B"
model_name="EleutherAI/gpt-neo-125M"
#model_name="EleutherAI/gpt-neo-125M"
#model_name="EleutherAI/gpt-neox-20b"
#model_name = "gpt-neox-20b-16"
model=AutoModelForCausalLM.from_pretrained(model_name)
dtype = torch.float32
int_dtype = torch.int64
if load_in_fp16:
  model = model.half()
  dtype = torch.float16
  int_dtype = torch.int32

model = model.to(device)
model = model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)
#model=AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=load_in_8bit, device_map='auto').eval()
#tokenizer = AutoTokenizer.from_pretrained(model_name, load_in_8bit=load_in_8bit, device_map='auto')
#prompt="Germany is"
#prompt="France is"
#tokens = tokenizer.encode(prompt)
#L=len(tokens)
torch.cuda.empty_cache()
BATCH_SIZE = 1
N_tokens = 16
A_tokens = 16
F_tokens = N_tokens - A_tokens


class CausalLMWrapper(torch.nn.Module):
  def __init__(self, model):
    super().__init__()
    self.model = model
  def forward(self, input_ids, attention_mask):
    out = self.model(input_ids=input_ids,
                     attention_mask=attention_mask,
                     use_cache=False,
                     return_dict=False
                     )
    res = out[0]
    return res

m2 = CausalLMWrapper(model)
m2 = m2.to(device)
m2 = m2.eval()

input_ids = torch.tensor([[2233, 318, 213, 43]], dtype=torch.int64).to(device)
attention_mask = torch.tensor([[1,1,1,1]]).to(device)

print("Running")
out = m2(input_ids, attention_mask)

print("Model run done. out shape:", out.shape)

inputs = [
    torch_tensorrt.Input(
        min_shape=[1, 1],
        opt_shape=[1, 16],
        max_shape=[1, 512],
        dtype=torch.int32,
    ),
    torch_tensorrt.Input(
        min_shape=[1, 1],
        opt_shape=[1, 16],
        max_shape=[1, 512],
        dtype=torch.float32,
    )
]

enabled_precisions = {torch.float32}

trt_ts_module = torch_tensorrt.compile(
    m2, inputs=inputs, enabled_precisions=enabled_precisions
)

print("tensorrt compile Done")

