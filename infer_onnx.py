import onnx
import onnxruntime as ort
import numpy as np
from scipy.special import softmax

model_name='EleutherAI/gpt-neo-125M'
model_name='EleutherAI/gpt-neo-1.3b'
#model_file = "gpt-neo-125m.onnx"
model_file = "gpt-neo-125m_fakequant.onnx"
model_file = "gpt-neo-1.3b_fakequant/gpt-neo-1.3b_fakequant.onnx"
# onnx_model = onnx.load(model_name)
# onnx.checker.check_model(onnx_model)
print("Loading", model_file)

in_len = 2
input_ids = np.array([[27079, 318]], dtype="int64")
attention_mask = np.array([[1,1]], dtype="float32")

ort_sess = ort.InferenceSession(model_file, providers=["CUDAExecutionProvider"])
outputs = ort_sess.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})

next_token_logits = outputs[0]
logits = next_token_logits[0,in_len-1,:]
logits = softmax(logits)
sorted_ids = np.argsort(logits)
topk = np.flip(sorted_ids[-10:])

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("prompt:", tokenizer.decode(input_ids[0]))
for top1 in topk:
  print("id/prop/word", top1, logits[top1], f"'{tokenizer.decode(top1)}'")
