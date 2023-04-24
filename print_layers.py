import logging
import os
import tensorrt as trt
import builtins

from net_utils import set_fp32

logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
log = logging.getLogger("EngineBuilder")

onnx_path = "gpt-neo-125m.onnx"
engine_path = "./gpt-neo-125m_fp16.trt"

trt_logger = trt.Logger(trt.Logger.INFO)

trt.init_libnvinfer_plugins(trt_logger, namespace="")

builder = trt.Builder(trt_logger)
config = builder.create_builder_config()
config.flags=0
config.set_flag(trt.BuilderFlag.FP16)

max_ws_size = 35 * (2 ** 30)
print("================== max_ws_size", type(max_ws_size), max_ws_size)
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_ws_size)

network_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
network = builder.create_network(network_flags)
parser = trt.OnnxParser(network, trt_logger)

onnx_path = os.path.realpath(onnx_path)

with open(onnx_path, "rb") as f:
    if not parser.parse(f.read()):
        log.error("Failed to load ONNX file: {}".format(onnx_path))
        for error in range(parser.num_errors):
            log.error(parser.get_error(error))
        exit(1)

print(f"=========================== {onnx_path} Loaded")

def desc_layer(l, i):
    print(i, l.name, l.type, l.num_inputs, l.num_outputs, l.precision, l.precision_is_set)
    for j in range(l.num_outputs):
        t = l.get_output(j)
        print("    output", j, t.name, t.shape, l.get_output_type(j), l.output_type_is_set(j))


set_fp32(network)
num_layers = network.num_layers
for i in range(num_layers-1):
    l = network.get_layer(i)
    l1 = network.get_layer(i+1)
    if not(l.type == trt.LayerType.ELEMENTWISE and l1.type == trt.LayerType.REDUCE):
        continue
    desc_layer(l, i)
    desc_layer(l1, i+1)
