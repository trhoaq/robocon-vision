import onnxruntime as ort
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ---------- CONFIG ----------
MODEL_PATH = "ssd-mobilenet-v2.onnx"
IMAGE_PATH = "img.jpg"
CONF_THRESH = 0.4
INPUT_SIZE = 300  # SSD MobileNet V2 dùng 300x300
# ----------------------------

# Load image
image = Image.open(IMAGE_PATH).convert("RGB")
orig_w, orig_h = image.size

# Preprocess (chuẩn SSD MobileNet)
transform = T.Compose([
    T.Resize((INPUT_SIZE, INPUT_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5])
])

input_tensor = transform(image).unsqueeze(0).numpy()

# Load ONNX session
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: input_tensor})

# --------------------------------------------------
# SSD OUTPUT (phổ biến):
# outputs[0] = boxes  [1, N, 4]
# outputs[1] = scores [1, N, num_classes]
# --------------------------------------------------

boxes = outputs[0][0]
scores = outputs[1][0]

# COCO labels (80 classes)
COCO_CLASSES = [
    "background","person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse",
    "sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase",
    "frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard",
    "surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana",
    "apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]

# Plot
fig, ax = plt.subplots(1, figsize=(10, 8))
ax.imshow(image)
print(outputs)

# detections = outputs[0][0]  # shape: [N, 6]

# for det in detections:
#     x1, y1, x2, y2, score, class_id = det

#     score = float(score)
#     class_id = int(class_id)

#     if score < CONF_THRESH or class_id == 0:
#         continue

#     # box đã là normalized [0,1]
#     x1 *= orig_w
#     x2 *= orig_w
#     y1 *= orig_h
#     y2 *= orig_h

#     rect = patches.Rectangle(
#         (x1, y1),
#         x2 - x1,
#         y2 - y1,
#         linewidth=2,
#         edgecolor="red",
#         facecolor="none"
#     )
#     ax.add_patch(rect)

#     ax.text(
#         x1,
#         y1,
#         f"{CLASSES[class_id]}: {score:.2f}",
#         color="white",
#         fontsize=10,
#         bbox=dict(facecolor="red", alpha=0.5)
#     )


# plt.axis("off")
# plt.show()
