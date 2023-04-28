trtexec \
--int8 \
--fp16 \
--verbose \
--onnx=gpt-neo-125m_fakequant.onnx \
--saveEngine=gpt-neo-125m_int8_trtexec.trt \
--minShapes=input_ids:1x16,attention_mask:1x16 \
--optShapes=input_ids:1x16,attention_mask:1x16 \
--maxShapes=input_ids:1x16,attention_mask:1x16
