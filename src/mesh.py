"""
Module for computing SDF and creating meshes.
"""

import time

import numpy as np
import trimesh
from skimage.measure import marching_cubes
import torch


def create_mesh(model, N=256, max_batch=32**3, verbose=False):
    """
    Create a mesh with the model.
    
    Parameters
    ----------
    model: PyTorch model
    N: int (default=256)
        Resolution (in each dimension) used for Marching Cubes.
    max_batch: int (default=32^3)
        Maximum number of samples predicted at once with the model.
        Useful to avoid memory errors on the GPU.
    """
    sdf_grid = compute_sdf_grid(model, N=N, max_batch=max_batch, verbose=verbose)
    mesh = convert_sdf_grid_to_mesh(sdf_grid, voxel_size=2. / (N - 1))
    return mesh, sdf_grid


def compute_sdf_grid(model, N=256, max_batch=32**3, bbox=[(-1., -1., -1.), (1., 1., 1.)], verbose=False):
    """
    Compute the SDF values over a grid in the given bounding box.

    Parameters
    ----------
    model: PyTorch model
    N: int (default=256)
        Resolution (in each dimension) used for Marching Cubes.
    max_batch: int (default=32^3)
        Maximum number of samples predicted at once with the model.
        Useful to avoid memory errors on the GPU.
    bbox: array-like, 2x3
        Limits of the grid as lowest/highest corner points.
    """
    n_points = N ** 3
    sdf = torch.zeros(n_points)

    # Create points on a 3D grid
    xx = torch.linspace(bbox[0][0], bbox[1][0], N)
    yy = torch.linspace(bbox[0][1], bbox[1][1], N)
    zz = torch.linspace(bbox[0][2], bbox[1][2], N)
    xyz = torch.meshgrid(xx, yy, zz, indexing='ij')
    xyz = torch.stack(xyz, dim=-1).view(-1, 3)

    # Predict the SDF values on the grid
    sdf = compute_sdf(model, xyz, max_batch, verbose)  
    return sdf.view(N, N, N)


def compute_sdf(model, xyz, max_batch=32**3, verbose=False):
    """
    Compute the SDF values at the given positions.
    
    Parameters
    ----------
    model: PyTorch model
    xyz: tensor
        3D coordinates where the SDF is predicted.
    max_batch: int (default=32^3)
        Maximum number of samples predicted at once with the model.
        Useful to avoid memory errors on the GPU.
    """
    if verbose:
        start_time = time.time()
    model.eval()

    # Prepare data
    xyz_all = xyz.view(-1, 3)
    n_points = len(xyz_all)
    sdf = torch.zeros(n_points)

    # Predict SDF on a subset of points at a time
    head = 0
    while head < n_points:
        xyz_subset = xyz_all[head : head + max_batch].cuda()

        sdf[head : head + max_batch] = model(xyz_subset).squeeze(1).detach().cpu()

        head += max_batch
    
    if verbose:
        print(f"sdf-prediction took {time.time() - start_time:.3f}s.")    
    return sdf.view(xyz.shape[:-1] + (1,))


def convert_sdf_grid_to_mesh(sdf_grid, voxel_size=1., voxel_origin=[-1., -1., -1.]):
    """
    Convert the grid of SDF values to a mesh using Marching Cubes.
    
    Parameters
    ----------
    sdf_grid: tensor/ndarray
        3D grid of SDF values.
    voxel_size: float
        Size of a voxel (:= a cell of the SDF grid) in 
        the 3D world.
    voxel_origin: array-like, 3
        3D world coordinate of the origin/lowest corner 
        of the grid.
    """
    sdf_grid_np = sdf_grid
    if not isinstance(sdf_grid_np, np.ndarray):
        sdf_grid_np = sdf_grid_np.numpy()
    if isinstance(voxel_size, (int, float)):
        voxel_size = [voxel_size] * 3

    try:
        verts, faces, normals, values = marching_cubes(sdf_grid_np, level=0., spacing=voxel_size)
        verts += np.array(voxel_origin)
    except ValueError:  # no surface within range
        verts, faces = None, None

    return trimesh.Trimesh(verts, faces)