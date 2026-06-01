import os
from pathlib import Path

import numpy as np
import torch
import cv2
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image
from typing import Optional, Union


def resolve_sam3_bpe_path() -> str:
    """
    PyPI `sam3` wheels omit tokenizer assets; the library defaults to a path under
    site-packages that does not exist. Use a local copy next to this project.
    Override with env SAM3_BPE_PATH if needed.
    """
    env = os.environ.get("SAM3_BPE_PATH")
    if env:
        p = Path(env).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"SAM3_BPE_PATH is set but file not found: {p}")
        return str(p.resolve())
    p = Path(__file__).resolve().parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    if not p.is_file():
        raise FileNotFoundError(
            f"Missing SAM3 BPE vocab at {p}. "
            "Download: https://github.com/facebookresearch/sam3/raw/main/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
        )
    return str(p)


# class to initiate and do segmentation via SAM3
class Segmentation:
    # initialize the model path and device
    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        bpe_path: Optional[str] = None,
    ):
        self.model_path = model_path
        self.device = device
        bpe = bpe_path or resolve_sam3_bpe_path()
        self.model = build_sam3_image_model(
            bpe_path=bpe,
            checkpoint_path=model_path,
            load_from_HF=False,
            device=device,
            eval_mode=True,
        )
        self.processor = Sam3Processor(self.model, device=device)

    def _to_pil_rgb(self, image: Union[str, np.ndarray]) -> Image.Image:
        """
        Convert an image path or numpy array into a PIL RGB image.

        - If `image` is a path, it is read with OpenCV (BGR) then converted to RGB.
        - If `image` is an array, it is assumed to already be RGB uint8 (H, W, 3).
        """
        # if the image is a path, read it with OpenCV (BGR) then convert to RGB
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
            # if using cuda, use autocast to cast the tensors to bf16
            if self.device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    state = self.processor.set_image(image_pil)
                    state = self.processor.set_text_prompt(prompt=prompt, state=state)
            # if not using cuda, use regular inference
            else:
                state = self.processor.set_image(image_pil)
                state = self.processor.set_text_prompt(prompt=prompt, state=state)

        masks = state["masks"].detach().cpu().numpy().astype(bool)
        # Some SAM variants include a singleton channel dimension: (N, 1, H, W).
        # this is to remove the singleton channel dimension
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0, :, :]
        # Under CUDA autocast, SAM3 may output bf16 tensors; NumPy can't convert bf16.
        # this prevents the error of converting bf16 to float32
        scores = state["scores"].detach().to(dtype=torch.float32).cpu().numpy()
        boxes = state["boxes"].detach().to(dtype=torch.float32).cpu().numpy()
        return masks, scores, boxes