text="""One morning, when Gregor Samsa woke from troubled dreams, he found himself transformed in his bed into a horrible vermin. He lay on his armour-like back, and if he lifted his head a little he could see his brown belly, slightly domed and divided by arches into stiff sections. The bedding was hardly able to cover it and seemed ready to slide off any moment. His many legs, pitifully thin compared with the size of the rest of him, waved about helplessly as he looked. "What's happened to me?" he thought. It wasn't a dream. His room, a proper human room although a little too small, lay peacefully between its four familiar walls. A collection of textile samples lay spread out on the table - Samsa was a travelling salesman - and above it there hung a picture that he had recently cut out of an illustrated magazine and housed in a nice, gilded frame. It showed a lady fitted out with a fur hat and fur boa who sat upright, raising a heavy fur muff that covered the whole of her lower arm towards the viewer. Gregor then turned to look out the window at the dull weather. Drops of rain could be heard hitting the pane, which made him feel quite sad. "How about if I sleep a little bit longer and forget all this nonsense", he thought, but that was something he was unable to do because he was used to sleeping on his right, and in his present state couldn't get into"""

tokens=[3198, 3329, 11, 618, 8547, 273, 3409, 11400, 19092, 422, 17840, 10625, 11, 339, 1043, 2241, 14434, 287, 465, 3996, 656, 257, 12361, 3326, 1084, 13, 679, 3830, 319, 465, 18588, 12, 2339, 736, 11, 290, 611, 339, 13663, 465, 1182, 257, 1310, 339, 714, 766, 465, 7586, 19921, 11, 4622, 2401, 276, 290, 9086, 416, 610, 2052, 656, 15175, 9004, 13, 383, 3996, 12083, 373, 8941, 1498, 284, 3002, 340, 290, 3947, 3492, 284, 10649, 572, 597, 2589, 13, 2399, 867, 7405, 11, 6028, 17049, 7888, 3688, 351, 262, 2546, 286, 262, 1334, 286, 683, 11, 26834, 546, 21144, 306, 355, 339, 3114, 13, 366, 2061, 338, 3022, 284, 502, 1701, 339, 1807, 13, 632, 2492, 470, 257, 4320, 13, 2399, 2119, 11, 257, 1774, 1692, 2119, 3584, 257, 1310, 1165, 1402, 11, 3830, 30996, 1022, 663, 1440, 5385, 7714, 13, 317, 4947, 286, 45293, 8405, 3830, 4104, 503, 319, 262, 3084, 532, 3409, 11400, 373, 257, 16574, 42414, 532, 290, 2029, 340, 612, 9174, 257, 4286, 326, 339, 550, 2904, 2005, 503, 286, 281, 18542, 7093, 290, 23707, 287, 257, 3621, 11, 308, 46158, 5739, 13, 632, 3751, 257, 10846, 18235, 503, 351, 257, 9230, 6877, 290, 9230, 1489, 64, 508, 3332, 24826, 11, 8620, 257, 4334, 9230, 27563, 326, 5017, 262, 2187, 286, 607, 2793, 3211, 3371, 262, 19091, 13, 8547, 273, 788, 2900, 284, 804, 503, 262, 4324, 379, 262, 19222, 6193, 13, 41692, 286, 6290, 714, 307, 2982, 9008, 262, 37218, 11, 543, 925, 683, 1254, 2407, 6507, 13, 366, 2437, 546, 611, 314, 3993, 257, 1310, 1643, 2392, 290, 6044, 477, 428, 18149, 1600, 339, 1807, 11, 475, 326, 373, 1223, 339, 373, 5906, 284, 466, 780, 339, 373, 973, 284, 11029, 319, 465, 826, 11, 290, 287, 465, 1944, 1181, 3521, 470, 651, 656]
tokens=[27079, 318, 1576]

import numpy as np
from scipy.special import softmax
np.set_printoptions(precision=5)
from transformers import pipeline, set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer
set_seed(42)
#model_name="EleutherAI/gpt-neo-2.7B"
model_name="EleutherAI/gpt-neo-1.3B"
#model_name="EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
#input_ids=tokenizer(text)["input_ids"]
#attention_mask=tokenizer(text)["attention_mask"]
N_tokens = 2#16
A_tokens = 2#16
F_tokens = N_tokens - A_tokens
input_ids = np.array([tokens[:N_tokens]], dtype="int32")
attention_mask = np.array([[1]*A_tokens+[0]*F_tokens], dtype="float32")

import tensorrt as trt
from cuda import cuda, nvrtc

def ASSERT_DRV(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("Nvrtc Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))

#import pudb; pu.db
err, = cuda.cuInit(0)
ASSERT_DRV(err)
# Retrieve handle for device 0
err, cuDevice = cuda.cuDeviceGet(0)
ASSERT_DRV(err)
# Create context
err, cuCtx = cuda.cuCtxCreate(0, cuDevice)
ASSERT_DRV(err)


trt_logger = trt.Logger(trt.Logger.INFO)
trt_logger.min_severity = trt.Logger.Severity.VERBOSE
runtime = trt.Runtime(trt_logger)
#fpath="gpt-neox-1b.trt"
#fpath="gpt-neox-1b-quant.trt"
#fpath="gpt-neox-1b-8bit.trt"
#fpath="gpt-neo-125m-8bit.trt"
#fpath="gpt-neo-125m-best.trt"
fpath="gpt-neo-125m_int8.trt"
#fpath="gpt-neo-125m_fp16.trt"
#fpath="gpt-neo-125m_fp32.trt"
#fpath="gpt-neo-125m.trt"
#fpath="gpt-neo-125m_py.trt"
#fpath="gpt-neo-125m_fp16_py.trt"
#fpath="gpt-neo-125m_fp16_fast_py.trt"
#fpath="gpt-neo-125m_fp16_fast2_py.trt"
#fpath="gpt-neo-125m_int8_py.trt"
#fpath="gpt-neo-125m_trtexec.trt"
#fpath="gpt-neo-125m_fp16_trtexec.trt"

fpath="gpt-neo-1.3b_int8.trt"
#fpath="gpt-neo-1.3b_fp16_fast.trt"
#fpath="gpt-neo-2.7b_fp16_fast.trt"
#fpath="gpt-neo-2.7b_int8_fast.trt"

#fpath="gpt-neox-20b-16.trt"


with open(fpath, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())

print("Engine Info:")
for i, name in enumerate(engine):
    shape = engine.get_tensor_shape(name)
    dtype = trt.nptype(engine.get_tensor_dtype(name))
    volume = abs(trt.volume(engine.get_tensor_shape(name)))
    is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
    if is_input:
        desc = "input"
    else:
        desc = "output"
    print(f"{i} type:    {desc}\n  name: {name} \n  data:    {np.dtype(dtype).name}\n  shape:   {shape} => {volume} \n")



context: trt.IExecutionContext = engine.create_execution_context()
BATCH_SIZE = 1

unspecified_tensors = context.infer_shapes()
print("unspecified_tensors:", unspecified_tensors)

context.set_input_shape("input_ids", (BATCH_SIZE, N_tokens))
context.set_input_shape("attention_mask", (BATCH_SIZE, N_tokens))

out_shapes = []
out_dtypes = []
print("num_optimization_profiles:", engine.num_optimization_profiles)
for i in range(engine.num_bindings):
    name = engine.get_tensor_name(i)
    is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
    dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
    shape = context.get_tensor_shape(name)
    print("from context:", name, dtype, shape)
    if is_input:
        profile_shape = engine.get_tensor_profile_shape(name, 0)
        print("    profile_shape:", name, profile_shape)
    else:
        out_shapes.append(shape)
        out_dtypes.append(dtype)

print("out_shapes:", out_shapes)
print("out_dtypes:", out_dtypes)

unspecified_tensors = context.infer_shapes()
print("unspecified_tensors:", unspecified_tensors)
assert(len(unspecified_tensors) == 0)

output = np.empty(out_shapes[0], dtype = out_dtypes[0])

# allocate device memory

err, d_input_ids = cuda.cuMemAlloc(1 * input_ids.nbytes)
ASSERT_DRV(err)
err, d_attention_mask = cuda.cuMemAlloc(1 * attention_mask.nbytes)
ASSERT_DRV(err)
err, d_output = cuda.cuMemAlloc(1 * output.nbytes)
ASSERT_DRV(err)
bindings = [int(d_input_ids), int(d_attention_mask), int(d_output)]
#bindings = [int(d_input_ids), int(d_output)]
err, stream = cuda.cuStreamCreate(0)
ASSERT_DRV(err)

def predict(input_ids, attention_mask): # result gets copied into output
    # transfer input data to device
    err, = cuda.cuMemcpyHtoDAsync(d_input_ids, input_ids, 1 * input_ids.nbytes, stream)
    ASSERT_DRV(err)
    #if attention_mask is not None:
    err, = cuda.cuMemcpyHtoDAsync(d_attention_mask, attention_mask, 1 * attention_mask.nbytes, stream)
    ASSERT_DRV(err)
    # execute model
    exec_res = context.execute_async_v2(bindings, stream)
    print("execute_v2:", exec_res)
    assert exec_res
    # transfer predictions back
    err, = cuda.cuMemcpyDtoHAsync(output, d_output, 1 * output.nbytes, stream)
    ASSERT_DRV(err)
    # synchronize threads
    err, = cuda.cuStreamSynchronize(stream)
    ASSERT_DRV(err)


predict(input_ids, attention_mask)
logits = output[0,A_tokens-1,:]
logits = softmax(logits)

sorted_ids = np.argsort(logits)
topk = np.flip(sorted_ids[-30:])

print(tokenizer.decode(tokens[:A_tokens + 1]))
for top1 in topk:
  print("id/prop/word", top1, logits[top1], f"'{tokenizer.decode(top1)}'")

if 0:
    insp = engine.create_engine_inspector()
    eng_info = insp.get_engine_information(trt.LayerInformationFormat.JSON)
    l_info = insp.get_layer_information(3, trt.LayerInformationFormat.JSON)
    print(l_info)
    exit()

if 1:
    # Warmup
    for i in range(100):
      predict(input_ids, attention_mask)

    #exit()
    # Measure Latency
    import time
    TT=[]
    for i in range(1):
      t0=time.time()
      predict(input_ids, attention_mask)
      t1=time.time()
      TT.append((t1-t0)*1000)

    print("AVG time (ms):",np.mean(TT))
    print("P50 time (ms):",np.percentile(TT, 50))
    print("P95 time (ms):",np.percentile(TT, 95))

err, = cuda.cuStreamDestroy(stream)
err, = cuda.cuMemFree(d_input_ids)
err, = cuda.cuMemFree(d_attention_mask)
err, = cuda.cuMemFree(d_output)
