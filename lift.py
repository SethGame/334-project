from __future__ import annotations

import numpy as np
import cv2

from segmentation import Segmentation
from depth import DepthEstimator

def estimate_intrinsics(
    width: int,
    height: int,
    fov_degrees: float = 60.0,
) -> tuple[float, float, float, float]:
    """
    Estimate pinhole intrinsics (fx, fy, cx, cy) from image size using an assumed FOV.
    """

    # calculate the center of the image
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    # calculate the focal length
    fov_rad = np.deg2rad(fov_degrees)
    fx = (width / 2.0) / np.tan(fov_rad / 2.0)
    fy = fx
    return fx, fy, cx, cy

# function to lift the masked points into a 3D point cloud
def lift_masked_points(
    image_rgb: np.ndarray,
    depth_map: np.ndarray,
    mask: np.ndarray,
    *,
    fx: float | None = None,
    fy: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
    depth_scale: float = 1.0,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Lift masked pixels + depth into a colored 3D point cloud.

    Returns:
        points_xyz: (N, 3) float32
        colors_rgb: (N, 3) uint8 (same as input image)
    """
    # if the image does not have the correct shape, raise an error
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must have shape (H, W, 3)")
    h, w = image_rgb.shape[:2]
    if depth_map.shape[:2] != (h, w):
        raise ValueError(f"depth_map must have shape (H, W) matching image; got {depth_map.shape} vs {(h, w)}")
    if mask.shape[:2] != (h, w):
        raise ValueError(f"mask must have shape (H, W) matching image; got {mask.shape} vs {(h, w)}")

    if stride < 1:
        raise ValueError("stride must be >= 1")

    # if fx, fy, cx, cy are not provided, estimate them from the image size
    if fx is None or fy is None or cx is None or cy is None:
        _fx, _fy, _cx, _cy = estimate_intrinsics(w, h)
        fx = _fx if fx is None else fx
        fy = _fy if fy is None else fy
        cx = _cx if cx is None else cx
        cy = _cy if cy is None else cy

    mask_bool = mask.astype(bool)
    # if the depth map is finite and greater than 0, and the stride is greater than 1, then the valid points are the points that are in the mask and the depth map is finite and greater than 0 and the stride is greater than 1
    valid = mask_bool & np.isfinite(depth_map) & (depth_map > 0)
    if stride > 1:
        valid &= ((np.arange(h)[:, None] % stride) == 0) & ((np.arange(w)[None, :] % stride) == 0)

    ys, xs = np.where(valid)
    if ys.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    # calculate the z, x, y coordinates
    z = (depth_map[ys, xs].astype(np.float32) * float(depth_scale))
    x = ((xs.astype(np.float32) - float(cx)) * z) / float(fx)
    y = ((ys.astype(np.float32) - float(cy)) * z) / float(fy)
    # conver the coordinates to a numpy array, and convert the coordinates to float32
    points_xyz = np.stack([x, y, z], axis=1).astype(np.float32, copy=False)
    colors_rgb = image_rgb[ys, xs].astype(np.uint8, copy=False)
    return points_xyz, colors_rgb

def lift(image_path: str, prompt: str):
    """
    Full lift step for the pipeline:
    segmentation mask + depth map + pinhole unprojection -> colored point cloud.

    Args:
        image_path: path to the image on disk.
        prompt: SAM3 text prompt.

    Returns:
        points_xyz: (N, 3) float32
        colors_rgb: (N, 3) uint8
        best_mask: (H, W) bool
        depth_map: (H, W) float
    """
    # call the segmentation and depth estimation models
    segmentation = Segmentation(model_path="checkpoints/sam3.pt")
    depth = DepthEstimator(model_type="DPT_Hybrid")
    masks, scores, boxes = segmentation.segment(image_path, prompt)
    best_idx = int(np.argmax(scores)) if scores.size else 0
    best_mask = masks[best_idx]

    # read the image and convert it to RGB
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not read image at path: {image_path}")
    image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth_map = depth.predict_depth(image_rgb)
    # lift the masked points into a 3D point cloud
    points_xyz, colors_rgb = lift_masked_points(image_rgb, depth_map, best_mask, stride=2)
    return points_xyz, colors_rgb, best_mask, depth_map