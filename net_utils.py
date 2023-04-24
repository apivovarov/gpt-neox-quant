tokens=[3198, 3329, 11, 618, 8547, 273, 3409, 11400, 19092, 422, 17840, 10625, 11, 339, 1043, 2241, 14434, 287, 465, 3996, 656, 257, 12361, 3326, 1084, 13, 679, 3830, 319, 465, 18588, 12, 2339, 736, 11, 290, 611, 339, 13663, 465, 1182, 257, 1310, 339, 714, 766, 465, 7586, 19921, 11, 4622, 2401, 276, 290, 9086, 416, 610, 2052, 656, 15175, 9004, 13, 383, 3996, 12083, 373, 8941, 1498, 284, 3002, 340, 290, 3947, 3492, 284, 10649, 572, 597, 2589, 13, 2399, 867, 7405, 11, 6028, 17049, 7888, 3688, 351, 262, 2546, 286, 262, 1334, 286, 683, 11, 26834, 546, 21144, 306, 355, 339, 3114, 13, 366, 2061, 338, 3022, 284, 502, 1701, 339, 1807, 13, 632, 2492, 470, 257, 4320, 13, 2399, 2119, 11, 257, 1774, 1692, 2119, 3584, 257, 1310, 1165, 1402, 11, 3830, 30996, 1022, 663, 1440, 5385, 7714, 13, 317, 4947, 286, 45293, 8405, 3830, 4104, 503, 319, 262, 3084, 532, 3409, 11400, 373, 257, 16574, 42414, 532, 290, 2029, 340, 612, 9174, 257, 4286, 326, 339, 550, 2904, 2005, 503, 286, 281, 18542, 7093, 290, 23707, 287, 257, 3621, 11, 308, 46158, 5739, 13, 632, 3751, 257, 10846, 18235, 503, 351, 257, 9230, 6877, 290, 9230, 1489, 64, 508, 3332, 24826, 11, 8620, 257, 4334, 9230, 27563, 326, 5017, 262, 2187, 286, 607, 2793, 3211, 3371, 262, 19091, 13, 8547, 273, 788, 2900, 284, 804, 503, 262, 4324, 379, 262, 19222, 6193, 13, 41692, 286, 6290, 714, 307, 2982, 9008, 262, 37218, 11, 543, 925, 683, 1254, 2407, 6507, 13, 366, 2437, 546, 611, 314, 3993, 257, 1310, 1643, 2392, 290, 6044, 477, 428, 18149, 1600, 339, 1807, 11, 475, 326, 373, 1223, 339, 373, 5906, 284, 466, 780, 339, 373, 973, 284, 11029, 319, 465, 826, 11, 290, 287, 465, 1944, 1181, 3521, 470, 651, 656]

import os
import logging

import numpy as np
import tensorrt as trt
from cuda import cuda

log = logging.getLogger("EngineBuilder")

def set_fp32_gpt(network: trt.INetworkDefinition):
  for i in range(network.num_layers - 1):
    l = network.get_layer(i)
    l_next = network.get_layer(i + 1)

    if not all([l.get_output(i).is_execution_tensor for i in range(l.num_outputs)]):
      continue

    if l.get_output_type(0) != trt.float32:
      continue

    if l.type == trt.LayerType.ELEMENTWISE and l_next.type == trt.LayerType.REDUCE:
      l.__class__ = getattr(trt, "IElementWiseLayer")
      if l.op == trt.ElementWiseOperation.POW:
        l.precision = trt.float32
        l.set_output_type(0, trt.float32)

      l_next.precision = trt.float32
      l_next.set_output_type(0, trt.float32)


def set_fp32(network: trt.INetworkDefinition):
  """
  Force operations involved in layer norm to run in FP32 precision.
  """
  pow_ops = {}
  for layer_index, layer in enumerate(network):
    if layer.type == trt.LayerType.IDENTITY:
      all_fp32 = all(
        [layer.output_type_is_set(o) and layer.get_output_type(o) == trt.float32 for o in range(layer.num_outputs)])
      if all_fp32:
        if layer.get_input(0).dtype == trt.float32:
          layer.precision = trt.float32

    if layer.type == trt.LayerType.ELEMENTWISE:
      layer.__class__ = getattr(trt, "IElementWiseLayer")
      if layer.op == trt.ElementWiseOperation.POW:
        pow_ops[layer] = layer_index
        layer.precision = trt.float32
        layer.set_output_type(0, trt.float32)

  for _, index in pow_ops.items():
    # Iterate from few layers before pow to include residual add and cast op.
    # Iterate till 10 layers after pow op to include all operations included in layer norm.
    START_OFFSET = 4
    END_OFFSET = 12
    for i in range(index - START_OFFSET, index + END_OFFSET):
      l = network.get_layer(i)
      if l.type == trt.LayerType.REDUCE:
        l.precision = trt.float32
        l.set_output_type(0, trt.float32)

      if l.type == trt.LayerType.ELEMENTWISE:
        l.__class__ = getattr(trt, "IElementWiseLayer")
        if l.op == trt.ElementWiseOperation.SUM:
          l.precision = trt.float32
          l.set_output_type(0, trt.float32)

      if l.type == trt.LayerType.UNARY:
        l.__class__ = getattr(trt, "IUnaryLayer")
        if l.op == trt.UnaryOperation.SQRT:
          l.precision = trt.float32
          l.set_output_type(0, trt.float32)

      if l.type == trt.LayerType.ELEMENTWISE:
        l.__class__ = getattr(trt, "IElementWiseLayer")
        if l.op == trt.ElementWiseOperation.DIV:
          l.precision = trt.float32
          l.set_output_type(0, trt.float32)

      if l.type == trt.LayerType.ELEMENTWISE:
        l.__class__ = getattr(trt, "IElementWiseLayer")
        if l.op == trt.ElementWiseOperation.PROD:
          l.precision = trt.float32
          l.set_output_type(0, trt.float32)


def _cuda_error_check(args):
    """CUDA error checking."""
    err, ret = args[0], args[1:]
    if isinstance(err, cuda.CUresult):
      if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError("Cuda Error: {}".format(err))
    else:
      raise RuntimeError("Unknown error type: {}".format(err))
    # Special case so that no unpacking is needed at call-site.
    if len(ret) == 1:
      return ret[0]
    return ret


class EngineCalibrator(trt.IInt8MinMaxCalibrator):
    """
    Implements the INT8 MinMax Calibrator.
    """

    def __init__(self, cache_file):
      """
      :param cache_file: The location of the cache file.
      """
      super().__init__()
      self.cache_file = cache_file
      N_tokens = 16
      A_tokens = 16
      F_tokens = N_tokens - A_tokens
      self.input_ids = np.array([tokens[:N_tokens]], dtype="int32")
      self.attention_mask = np.array([[1] * A_tokens + [0] * F_tokens], dtype="int32")
      self.cnt = 0
      _cuda_error_check(cuda.cuInit(0))
      cuDevice = _cuda_error_check(cuda.cuDeviceGet(0))
      cuCtx = _cuda_error_check(cuda.cuCtxCreate(0, cuDevice))
      self.input_ids_allocation = _cuda_error_check(cuda.cuMemAlloc(1 * self.input_ids.nbytes))
      self.attention_mask_allocation = _cuda_error_check(cuda.cuMemAlloc(1 * self.attention_mask.nbytes))


    def get_batch_size(self):
      """
      Overrides from trt.IInt8MinMaxCalibrator.
      Get the batch size to use for calibration.
      :return: Batch size.
      """
      return 1

    def get_batch(self, names):
      """
      Overrides from trt.IInt8MinMaxCalibrator.
      Get the next batch to use for calibration, as a list of device memory pointers.
      :param names: The names of the inputs, if useful to define the order of inputs.
      :return: A list of int-casted memory pointers.
      """
      if self.cnt > 20:
        return None

      try:
        log.info(f"Calibrating data for inputs {names}")
        _cuda_error_check(
          cuda.cuMemcpyHtoD(
            self.input_ids_allocation,
            np.ascontiguousarray(self.input_ids),
            1 * self.input_ids.nbytes))

        _cuda_error_check(
          cuda.cuMemcpyHtoD(
            self.attention_mask_allocation,
            np.ascontiguousarray(self.attention_mask),
            1 * self.attention_mask.nbytes))

        self.cnt += 1
        return [int(self.input_ids_allocation), int(self.attention_mask_allocation)]
      except StopIteration:
        log.info("Finished calibration batches")
        return None

    def read_calibration_cache(self):
      """
      Overrides from trt.IInt8MinMaxCalibrator.
      Read the calibration cache file stored on disk, if it exists.
      :return: The contents of the cache file, if any.
      """
      if os.path.exists(self.cache_file):
        with open(self.cache_file, "rb") as f:
          log.info("Using calibration cache file: {}".format(self.cache_file))
          return f.read()

    def write_calibration_cache(self, cache):
      """
      Overrides from trt.IInt8MinMaxCalibrator.
      Store the calibration cache to a file on disk.
      :param cache: The contents of the calibration cache to store.
      """
      if self.cache_file is None:
        return
      with open(self.cache_file, "wb") as f:
        log.info("Writing calibration cache data to: {}".format(self.cache_file))
        f.write(cache)

