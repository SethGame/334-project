import os
import sys
import torch
import cv2
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# Resolve paths relative to this file so the script works no matter the CWD.
_HERE = os.path.dirname(os.path.abspath(__file__))

# Checkpoint paths.
# NOTE: `build_sam3_image_model` expects the SAM3 *image* checkpoint (`sam3.pt`).
# The `sam3.1_multiplex.pt` checkpoint is for the SAM3.1 *video* predictor API
# (built via `build_sam3_predictor`), and will not cleanly load into the image model.
image_ckpt_path = os.path.join(_HERE, "checkpoints", "sam3.pt")
video_ckpt_path = os.path.join(_HERE, "checkpoints", "sam3.1_multiplex.pt")

# Device to use for the model
device = "cuda" if torch.cuda.is_available() else "cpu"

# Import smoke test (verifies the package is installed and importable).
print("SAM3 imports OK")

# Only attempt to load weights if they are present locally.
if not os.path.exists(image_ckpt_path):
    if os.path.exists(video_ckpt_path):
        print(
            "Found SAM3.1 video checkpoint but not SAM3 image checkpoint.\n"
            f"- video checkpoint: {video_ckpt_path!r}\n"
            f"- expected image checkpoint for this script: {image_ckpt_path!r}\n"
            "For still-image text-prompted masks with `Sam3Processor`, download/place `sam3.pt`.\n"
            "If you specifically want SAM3.1, use the video predictor API instead."
        )
    else:
        print(
            f"Checkpoint not found at {image_ckpt_path!r}. Skipping weight loading for now."
        )
    sys.exit(0)

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