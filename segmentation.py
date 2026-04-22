import numpy as np
import torch
import cv2
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image
from typing import Union

# class to initiate and do segmentation via SAM3
class Segmentation:
    def __init__(self, model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model_path = model_path
        self.device = device
        self.model = build_sam3_image_model(checkpoint_path=model_path, load_from_HF=False, device=device, eval_mode=True)
        self.processor = Sam3Processor(self.model, device=device)

    def _to_pil_rgb(self, image: Union[str, np.ndarray]) -> Image.Image:
        """
        Convert an image path or numpy array into a PIL RGB image.

        - If `image` is a path, it is read with OpenCV (BGR) then converted to RGB.
        - If `image` is an array, it is assumed to already be RGB uint8 (H, W, 3).
        """
        if isinstance(image, str):
            bgr = cv2.imread(image)
            if bgr is None:
                raise ValueError(f"Could not read image at path: {image}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("image must have shape (H, W, 3)")
            rgb = image
        return Image.fromarray(rgb)

    def segment(self, image: Union[str, np.ndarray], prompt: str):
        """
        Run SAM3 segmentation with a text prompt.

        Args:
            image: path to an image file, or an RGB numpy array (H, W, 3).
            prompt: text prompt for SAM3.

        Returns:
            masks: (N, H, W) bool numpy array
            scores: (N,) float numpy array
            boxes: (N, 4) float numpy array
        """
        image_pil = self._to_pil_rgb(image)

        with torch.inference_mode():
            if self.device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    state = self.processor.set_image(image_pil)
                    state = self.processor.set_text_prompt(prompt=prompt, state=state)
            else:
                state = self.processor.set_image(image_pil)
                state = self.processor.set_text_prompt(prompt=prompt, state=state)

        masks = state["masks"].detach().cpu().numpy().astype(bool)
        # Some SAM variants include a singleton channel dimension: (N, 1, H, W).
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0, :, :]
        # Under CUDA autocast, SAM3 may output bf16 tensors; NumPy can't convert bf16.
        scores = state["scores"].detach().to(dtype=torch.float32).cpu().numpy()
        boxes = state["boxes"].detach().to(dtype=torch.float32).cpu().numpy()
        return masks, scores, boxes