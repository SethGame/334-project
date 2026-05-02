from __future__ import annotations

import numpy as np
import open3d as o3d


def pointcloud_from_arrays(points_xyz: np.ndarray, colors_rgb: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Build an Open3D point cloud from numpy arrays.

    Args:
        points_xyz: (N, 3) float array
        colors_rgb: (N, 3) uint8 or float array

    Returns:
        pcd: Open3D PointCloud with colors in [0, 1].
    """
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"points_xyz must have shape (N, 3); got {points_xyz.shape}")
    if colors_rgb.ndim != 2 or colors_rgb.shape[1] != 3 or colors_rgb.shape[0] != points_xyz.shape[0]:
        raise ValueError(f"colors_rgb must have shape (N, 3) matching points; got {colors_rgb.shape}")

    # get the points and colors
    pts = points_xyz.astype(np.float64, copy=False)
    cols = colors_rgb.astype(np.float64, copy=False)
    # if the colors are greater than 1, divide by 255
    if cols.max() > 1.0:
        cols = cols / 255.0
    cols = np.clip(cols, 0.0, 1.0)
    # create the point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    return pcd


def clean_point_cloud(pcd: o3d.geometry.PointCloud, *, voxel_size: float | None = None, nb_neighbors: int = 20, std_ratio: float = 2.0, normal_radius: float | None = None, normal_max_nn: int = 30) -> o3d.geometry.PointCloud:
    """
    Downsample, remove outliers, and estimate normals.
    """
    if voxel_size is not None and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # Statistical outlier removal.
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    # Estimate normals (needed for Poisson/BPA).
    if normal_radius is None:
        # Heuristic: scale normal radius with point cloud size.
        bbox = pcd.get_axis_aligned_bounding_box()
        diag = np.linalg.norm(bbox.get_extent())
        normal_radius = max(diag * 0.01, 1e-6)

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=float(normal_radius), max_nn=int(normal_max_nn)))
    try:
        pcd.orient_normals_consistent_tangent_plane(50)
    except Exception:
        # Orientation can fail on sparse/noisy clouds; normals are still usable.
        pass

    return pcd


def poisson_reconstruction(pcd: o3d.geometry.PointCloud, *, depth: int = 9, density_trim_quantile: float = 0.05) -> o3d.geometry.TriangleMesh:
    """
    Poisson surface reconstruction with density-based trimming.
    """
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=int(depth))
    # if the densities are not None, and the length of the densities is greater than 0, and the density trim quantile is between 0 and 1, then the vertices to remove are the vertices that are less than the threshold
    if densities is not None and len(densities) > 0 and 0.0 < density_trim_quantile < 1.0:
        dens = np.asarray(densities)
        thresh = np.quantile(dens, float(density_trim_quantile))
        verts_to_remove = dens < thresh
        mesh.remove_vertices_by_mask(verts_to_remove)

    mesh.compute_vertex_normals()
    return mesh


def mesh_point_cloud(points_xyz: np.ndarray, colors_rgb: np.ndarray, *, voxel_size: float | None = None, poisson_depth: int = 9, density_trim_quantile: float = 0.05) -> o3d.geometry.TriangleMesh:
    """
    End-to-end meshing: arrays -> cleaned point cloud -> Poisson mesh.
    """
    # call all of the functions, and return the mesh
    pcd = pointcloud_from_arrays(points_xyz, colors_rgb)
    pcd = clean_point_cloud(pcd, voxel_size=voxel_size)
    mesh = poisson_reconstruction(pcd, depth=poisson_depth, density_trim_quantile=density_trim_quantile)
    return mesh