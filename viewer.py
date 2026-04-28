from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import open3d as o3d


def pointcloud_from_arrays(points_xyz: np.ndarray, colors_rgb: Optional[np.ndarray] = None) -> o3d.geometry.PointCloud:
    """
    Build an Open3D point cloud from numpy arrays.

    Args:
        points_xyz: (N, 3) float array
        colors_rgb: optional (N, 3) uint8/float array
    """
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"points_xyz must have shape (N, 3); got {points_xyz.shape}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64, copy=False))

    if colors_rgb is not None:
        if colors_rgb.shape != points_xyz.shape:
            raise ValueError(f"colors_rgb must have shape (N, 3) matching points; got {colors_rgb.shape}")
        cols = colors_rgb.astype(np.float64, copy=False)
        if cols.max() > 1.0:
            cols = cols / 255.0
        cols = np.clip(cols, 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(cols)

    return pcd


def show_point_cloud(points_xyz: np.ndarray, colors_rgb: Optional[np.ndarray] = None) -> None:
    """Display a point cloud interactively."""
    pcd = pointcloud_from_arrays(points_xyz, colors_rgb)
    o3d.visualization.draw_geometries([pcd])


def show_mesh(mesh: o3d.geometry.TriangleMesh) -> None:
    """Display a mesh interactively."""
    o3d.visualization.draw_geometries([mesh])


def save_point_cloud(path: Union[str, Path], points_xyz: np.ndarray, colors_rgb: Optional[np.ndarray] = None) -> str:
    """
    Save point cloud to .ply using Open3D 
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pcd = pointcloud_from_arrays(points_xyz, colors_rgb)
    o3d.io.write_point_cloud(str(out), pcd)
    return str(out.resolve())


def save_mesh(path: Union[str, Path], mesh: o3d.geometry.TriangleMesh) -> str:
    """Save mesh to .ply/.obj/etc."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(out), mesh)
    return str(out.resolve())

