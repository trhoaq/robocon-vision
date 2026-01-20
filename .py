import onnxruntime as ort

session = ort.InferenceSession("./experiments/models/ssd_lite_mobilenet_v2_voc.onnx")
model_inputs = session.get_inputs()
for input in model_inputs:
    print(f"Input Name: {input.name}")
    print(f"Input Shape: {input.shape}") # Should show [batch, 3, 640, 360] or similar