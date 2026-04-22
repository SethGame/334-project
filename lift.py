from __future__ import annotations

import numpy as np
import cv2

from segmentation import Segmentation
from depth import DepthEstimator

def lift(image_path: str, prompt: str):
    """
    Produces a segmentation mask and depth map for an image.

    Args:
        image_path: path to the image on disk.
        prompt: SAM3 text prompt.

    Returns:
        best_mask: (H, W) bool numpy array
        depth_map: (H, W) float numpy array
    """

    # call segmentation and depth estimation
    segmentation = Segmentation(model_path="checkpoints/sam3.pt")
    depth = DepthEstimator(model_type="DPT_Hybrid")
    masks, scores, boxes = segmentation.segment(image_path, prompt)
    best_idx = int(np.argmax(scores)) if scores.size else 0
    best_mask = masks[best_idx]

    # reads the image, converts to RGB, and predicts the depth
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not read image at path: {image_path}")
    image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth_map = depth.predict_depth(image_rgb)
    return best_mask, depth_map