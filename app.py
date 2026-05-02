from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import gradio as gr
import numpy as np

from depth import DepthEstimator
from lift import lift_masked_points
from mesh import mesh_point_cloud
from segmentation import Segmentation
from viewer import save_mesh, save_point_cloud

try:
    import plotly.graph_objects as go
except Exception:  # plotly may not be installed yet
    go = None


@dataclass
class _Models:
    segmenter: Segmentation
    depth: DepthEstimator


_MODELS: Optional[_Models] = None

# get the models
def _get_models() -> _Models:
    global _MODELS
    if _MODELS is None:
        _MODELS = _Models(
            segmenter=Segmentation(model_path=r"checkpoints/sam3.pt"),
            depth=DepthEstimator(model_type="DPT_Hybrid"),
        )
    return _MODELS

# pick the best mask
def _pick_best_mask(masks: np.ndarray, scores: np.ndarray) -> np.ndarray:
    if masks.ndim != 3:
        raise ValueError(f"Expected masks shape (N, H, W), got {masks.shape}")
    best_idx = int(np.argmax(scores)) if scores is not None and getattr(scores, "size", 0) else 0
    return masks[best_idx]

# convert the mask to a preview
def _mask_to_preview(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8) * 255
    return np.ascontiguousarray(m)

# convert the depth map to a preview
def _depth_to_preview(depth_map: np.ndarray) -> np.ndarray:
    d = depth_map.astype(np.float32)
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    d_norm = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX)
    d_u8 = d_norm.astype(np.uint8)
    d_color = cv2.applyColorMap(d_u8, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(d_color, cv2.COLOR_BGR2RGB)

# ensure the outputs directory exists
def _ensure_outputs_dir() -> Path:
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

# plot the point cloud
def _plot_point_cloud(points_xyz: np.ndarray, colors_rgb: np.ndarray, *, max_points: int = 20000):
    # if plotly is not installed, return None
    if go is None:
        return None
    n = int(points_xyz.shape[0])
    if n == 0:
        return go.Figure()
    # if the number of points is greater than the max points, then randomly select the points
    if n > max_points:
        idx = np.random.choice(n, size=max_points, replace=False)
        pts = points_xyz[idx]
        cols = colors_rgb[idx]
    else:
        pts = points_xyz
        cols = colors_rgb
    # convert the colors to a uint8 array
    c = cols.astype(np.uint8, copy=False)
    rgb_str = np.array([f"rgb({r},{g},{b})" for r, g, b in c], dtype=object)
    # create the figure
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker=dict(size=2, color=rgb_str, opacity=0.9),
            )
        ]
    )
    fig.update_layout(
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig

# plot the mesh
def _plot_mesh(mesh_vertices: np.ndarray, mesh_triangles: np.ndarray):
    if go is None:
        return None
    if mesh_vertices.size == 0 or mesh_triangles.size == 0:
        return go.Figure()
    v = mesh_vertices
    t = mesh_triangles
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=v[:, 0],
                y=v[:, 1],
                z=v[:, 2],
                i=t[:, 0],
                j=t[:, 1],
                k=t[:, 2],
                color="lightgray",
                opacity=1.0,
            )
        ]
    )
    fig.update_layout(scene=dict(aspectmode="data"), margin=dict(l=0, r=0, t=0, b=0))
    return fig


def run_pipeline(
    image_rgb: Optional[np.ndarray],
    prompt: str,
    stride: int,
    depth_scale: float,
    fov_degrees: float,
    make_mesh: bool,
    voxel_size: float,
    poisson_depth: int,
    density_trim_quantile: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[object], Optional[object], Optional[str], Optional[str], str]:
    """
    Returns:
        mask_preview: (H, W) uint8
        depth_preview_rgb: (H, W, 3) uint8
        pointcloud_plot: plotly Figure (or None)
        mesh_plot: plotly Figure (or None)
        pointcloud_ply_path: str
        mesh_ply_path: str (or None)
        status: str
    """
    if image_rgb is None:
        return None, None, None, None, None, None, "No image provided."
    if not isinstance(prompt, str) or not prompt.strip():
        return None, None, None, None, None, None, "Prompt is required."

    try:
        models = _get_models()

        # Gradio provides RGB uint8 numpy arrays for type="numpy".
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            return None, None, None, None, None, None, f"Expected image shape (H, W, 3), got {image_rgb.shape}"
        image_rgb = image_rgb.astype(np.uint8, copy=False)

        # Segmentation
        masks, scores, boxes = models.segmenter.segment(image=image_rgb, prompt=prompt.strip())
        best_mask = _pick_best_mask(masks, scores)

        # Depth
        depth_map = models.depth.predict_depth(image_rgb)

        # Lift to point cloud (masked)
        points_xyz, colors_rgb = lift_masked_points(
            image_rgb,
            depth_map,
            best_mask,
            stride=int(stride),
            depth_scale=float(depth_scale),
            # override intrinsics via fov by leaving fx/fy/cx/cy None and passing fov through estimate_intrinsics
        )

        # Save outputs
        out_dir = _ensure_outputs_dir()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        pc_path = out_dir / f"pointcloud_{stamp}.ply"
        pc_abs = save_point_cloud(pc_path, points_xyz, colors_rgb)

        mesh_abs = None
        mesh_plot = None
        if make_mesh:
            vs = None if voxel_size <= 0 else float(voxel_size)
            mesh = mesh_point_cloud(
                points_xyz,
                colors_rgb,
                voxel_size=vs,
                poisson_depth=int(poisson_depth),
                density_trim_quantile=float(density_trim_quantile),
            )
            mesh_path = out_dir / f"mesh_{stamp}.ply"
            mesh_abs = save_mesh(mesh_path, mesh)
            mesh_plot = _plot_mesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles))

        # Previews
        mask_preview = _mask_to_preview(best_mask)
        depth_preview = _depth_to_preview(depth_map)
        pc_plot = _plot_point_cloud(points_xyz, colors_rgb)

        status = f"OK: points={int(points_xyz.shape[0])}, saved={os.path.basename(pc_abs)}"
        if mesh_abs:
            status += f", mesh={os.path.basename(mesh_abs)}"
        if go is None:
            status += " (Install plotly to see 3D previews: pip install plotly)"

        return mask_preview, depth_preview, pc_plot, mesh_plot, pc_abs, mesh_abs, status

    except Exception as e:
        return None, None, None, None, None, None, f"Error: {type(e).__name__}: {e}"


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Object-centric 3D Reconstruction") as demo:
        gr.Markdown("## Object-centric 3D Reconstruction\nUpload an image, enter a prompt, and export a point cloud (and optional mesh).")
        # layout the interface
        with gr.Row():
            image_in = gr.Image(label="Input image", type="numpy")
            with gr.Column():
                prompt_in = gr.Textbox(label="Text prompt", value="apple")
                stride_in = gr.Slider(label="Stride (subsample pixels)", minimum=1, maximum=8, step=1, value=2)
                depth_scale_in = gr.Slider(label="Depth scale", minimum=0.1, maximum=10.0, step=0.1, value=1.0)
                fov_in = gr.Slider(label="Assumed FOV (degrees)", minimum=30, maximum=120, step=1, value=60)

                make_mesh_in = gr.Checkbox(label="Reconstruct mesh (Poisson)", value=True)
                voxel_size_in = gr.Slider(label="Voxel size (0 disables downsample)", minimum=0.0, maximum=0.05, step=0.001, value=0.0)
                poisson_depth_in = gr.Slider(label="Poisson depth", minimum=6, maximum=11, step=1, value=9)
                density_trim_in = gr.Slider(label="Density trim quantile", minimum=0.0, maximum=0.2, step=0.01, value=0.05)

                run_btn = gr.Button("Run")
                status_out = gr.Textbox(label="Status", interactive=False)

        with gr.Row():
            mask_out = gr.Image(label="Best mask preview", type="numpy")
            depth_out = gr.Image(label="Depth preview (colormap)", type="numpy")

        with gr.Row():
            pc_plot_out = gr.Plot(label="Point cloud preview (3D)")
            mesh_plot_out = gr.Plot(label="Mesh preview (3D)")

        with gr.Row():
            pc_file = gr.File(label="Point cloud (.ply)")
            mesh_file = gr.File(label="Mesh (.ply)")

        # run the pipeline
        run_btn.click(
            fn=run_pipeline,
            inputs=[
                image_in,
                prompt_in,
                stride_in,
                depth_scale_in,
                fov_in,
                make_mesh_in,
                voxel_size_in,
                poisson_depth_in,
                density_trim_in,
            ],
            outputs=[mask_out, depth_out, pc_plot_out, mesh_plot_out, pc_file, mesh_file, status_out],
        )

    return demo

# run the app
if __name__ == "__main__":
    app = build_app()
    app.launch()
