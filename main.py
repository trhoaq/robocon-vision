import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from transformers import AutoImageProcessor, AutoModelForObjectDetection

# Load image
image = Image.open("image.jpg").convert("RGB")

# Load processor & model từ Hugging Face
processor = AutoImageProcessor.from_pretrained(
    "hustvl/ssd-mobilenet-v2"
)
model = AutoModelForObjectDetection.from_pretrained(
    "hustvl/ssd-mobilenet-v2"
)

model.eval()

# Preprocess
inputs = processor(images=image, return_tensors="pt")

# Inference
with torch.no_grad():
    outputs = model(**inputs)

# Post-process (convert sang bbox gốc)
target_sizes = torch.tensor([image.size[::-1]])  # (H, W)
results = processor.post_process_object_detection(
    outputs,
    target_sizes=target_sizes,
    threshold=0.4
)[0]

fig, ax = plt.subplots(1, figsize=(10, 8))
ax.imshow(image)

for score, label, box in zip(
    results["scores"],
    results["labels"],
    results["boxes"]
):
    box = box.tolist()
    x1, y1, x2, y2 = box

    rect = patches.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        linewidth=2,
        edgecolor="red",
        facecolor="none"
    )
    ax.add_patch(rect)

    class_name = model.config.id2label[label.item()]
    ax.text(
        x1, y1,
        f"{class_name}: {score:.2f}",
        color="white",
        fontsize=10,
        bbox=dict(facecolor="red", alpha=0.5)
    )

plt.axis("off")
plt.show()
