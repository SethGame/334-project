import os
import sys
import torch
import cv2
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# Resolve paths relative to this file so the script works no matter the CWD.
_HERE = os.path.dirname(os.path.abspath(__file__))

# Checkpoint paths, only image checkpoint is needed
image_ckpt_path = os.path.join(_HERE, "checkpoints", "sam3.pt")
# no need for video checkpoint
#video_ckpt_path = os.path.join(_HERE, "checkpoints", "sam3.1_multiplex.pt")

# Device to use for the model
device = "cuda" if torch.cuda.is_available() else "cpu"

sam_model = build_sam3_image_model(
    checkpoint_path=image_ckpt_path,
    load_from_HF=False,
    device=device,
    eval_mode=True,
)

# processor to load the model
_processor = Sam3Processor(sam_model, device=device)
print("SAM3 model loaded successfully")

image = cv2.imread(os.path.join(_HERE, "apple.jpg"))
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# On CUDA, run SAM3 in bf16 end-to-end to avoid dtype mismatches
with torch.inference_mode():
    if device == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            state = _processor.set_image(image_rgb)
            state = _processor.set_text_prompt(prompt="apple", state=state)
    else:
        state = _processor.set_image(image_rgb)
        state = _processor.set_text_prompt(prompt="apple", state=state)

masks = state["masks"]  # bool tensor [N, H, W]
scores = state["scores"]
boxes = state["boxes"]
print({"num_masks": int(masks.shape[0]), "scores": scores.tolist(), "boxes": boxes.tolist()})