"""
Generate sdf samples for given meshes.
It can also save normals (as computed by IGL), but that changes
the SDF computation to use pseudo-normals which is not robust to unclean meshes.
"""

import os, os.path
import argparse
import json
from math import sqrt

import numpy as np
from scipy.spatial.distance import cdist
import trimesh
import igl



class Sampler():
    """Select adequate sampler and generate XYZ and SDF samples."""

    def __init__(self, mesh, pseudo_normal=False, to_float64=False):
        self.mesh = mesh
        self.pseudo_normal = pseudo_normal
        self.to_float64 = to_float64
        self.xyz = None
        self.sdf = None
        self.normals = None
        self.results = None

    
    @classmethod
    def get_sampler(cls, sampling, mesh, pseudo_normal=False):
        """Get a sampler for the mesh."""
        if sampling.startswith("uniform"):
            if "_" in sampling:
                shape = sampling.split("_")[1]
            else:
                shape="cube"
            return UniformSampler(mesh, pseudo_normal, shape)

        elif sampling.startswith("nearsurface"):
            if "_" in sampling:
                variance = sampling.split("_")[1]
            else:
                variance = 0.005
            return NearSurfaceSampler(mesh, pseudo_normal, variance)

        elif sampling == "surface":
            return SurfaceSampler(mesh, pseudo_normal)
            
        elif sampling.startswith("voxel"):
            resolution = sampling.split("_")[1]
            return VoxelSampler(mesh, pseudo_normal, resolution)


    def __call__(self, n_samples):
        """Sample coordinates and compute their SDF (+normals)."""
        # Sample coordinates
        self.xyz = self.sample(n_samples)
        # Compute SDF values
        results = igl.signed_distance(self.xyz, self.mesh.vertices, self.mesh.faces, 
                                      return_normals=self.pseudo_normal)
        self.sdf = results[0]
        if self.pseudo_normal:
            self.normals = results[3]
        # Convert to float32
        if not self.to_float64:
            self.xyz = self.xyz.astype(np.float32)
            self.sdf = self.sdf.astype(np.float32)
            if self.pseudo_normal:
                self.normals = self.normals.astype(np.float32)
        self._remove_nans()
        self._prepare_results()

    def sample(self, n_samples):
        """Sampling method, should be defined in children classes."""
        raise NotImplementedError("Children classes must implement this function.")
    

    def _remove_nans(self):
        """Remove samples with NaN SDF."""
        nan_indices = np.isnan(self.sdf)
        self.xyz = self.xyz[~nan_indices, :]
        self.sdf = self.sdf[~nan_indices]
        if self.pseudo_normal:
            self.normals = self.normals[~nan_indices, :]

    def _prepare_results(self):  # can be overwritten in children classes
        """Arrange samples for saving results."""
        pos_idx = self.sdf >= 0.
        neg_idx = ~pos_idx
        pos = np.concatenate([self.xyz[pos_idx], self.sdf[pos_idx, np.newaxis]], axis=1)
        neg = np.concatenate([self.xyz[neg_idx], self.sdf[neg_idx, np.newaxis]], axis=1)
        self.results = {
            "pos": pos,
            "neg": neg
        }
        if self.pseudo_normal:
            self.results["pos_normal"] = self.normals[pos_idx]
            self.results["neg_normal"] = self.normals[neg_idx]
    

    def get_validity(self):  # can be overwritten in children classes
        """Return a dicitonary with validity information."""
        if self.results is None:
            raise RuntimeError("No results available to check validity of sampling.")
        return {
            "enough_interior": self._enough_interior_samples(),
            "continuous_sdf": self._continuous_sdf()
        }

    def _enough_interior_samples(self):
        """At least 1% negative/interior samples."""
        frac_neg = len(self.results["neg"]) / (len(self.results["neg"]) + len(self.results["pos"])) * 100.
        return bool(frac_neg >= 1.)

    def _continuous_sdf(self, n_points=25000, tol=1e-4):
        """Difference in SDF between samples must be <= to their distance (i.e. 1-Lipschitz).
           Approximated on a sub-sample of all points."""
        rand_idx = np.random.permutation(len(self.xyz))[:n_points]
        sdf_diff = cdist(self.sdf[rand_idx, None], self.sdf[rand_idx, None])
        xyz_dist = cdist(self.xyz[rand_idx], self.xyz[rand_idx])
        return bool((sdf_diff <= xyz_dist + tol).all())
    

    def get_result_dict(self):  # can be overwritten in children classes
        """Return a dictionary with results."""
        if self.results is None:
            raise RuntimeError("No results available to return.")

        results = {"method": "pseudo-normal" if self.pseudo_normal else "winding-number"}
        results.update(self.results)
        return results



class UniformSampler(Sampler):
    """
    Samples uniformly in a given shape.

    Cube sampling is uniform in [-1, 1]^3
    Sphere sampling is uniform in ball of radius 1.
    """

    def __init__(self, mesh, pseudo_normal, shape="cube"):
        super().__init__(mesh, pseudo_normal)
        self.shape = shape.lower()
    
    def sample(self, n_samples):
        if self.shape in ["", "cube"]:
            samples = np.random.uniform(-1., 1., (n_samples, 3))
        elif self.shape in ["sphere", "ball"]:
            samples = self.sphere_sampling(n_samples)
        return samples
    
    def sphere_sampling(self, n_samples):
        """Uniformly sample 3D coordinate within the unit ball."""
        rand_samples = np.random.rand(3, n_samples)

        r = rand_samples[0]
        phi = rand_samples[1] * np.pi * 2.
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        cos_theta = rand_samples[2] * 2. - 1.
        sin_theta = np.sqrt(1. - cos_theta ** 2)

        x = r * cos_phi * sin_theta
        y = r * sin_phi * sin_theta
        z = r * cos_theta

        return np.stack([x, y, z], axis=-1)


class NearSurfaceSampler(Sampler):
    """Samples around the mesh surface."""
    def __init__(self, mesh, pseudo_normal, variance):
        super().__init__(mesh, pseudo_normal)
        self.variance = variance
    
    def sample(self, n_samples):
        """Sample on the surface, then add gaussian noise (twice)."""
        surf_samples = self.mesh.sample(n_samples)
        samples = np.concatenate([surf_samples + np.random.normal(scale=sqrt(self.variance), size=surf_samples.shape),
                                  surf_samples + np.random.normal(scale=sqrt(self.variance/10.), size=surf_samples.shape)], axis=0)
        return samples



class SurfaceSampler(Sampler):
    """Samples uniformly on the mesh surface."""
    def __init__(self, mesh, pseudo_normal):
        super().__init__(mesh, pseudo_normal)
    
    def sample(self, n_samples):
        return self.mesh.sample(n_samples)
    
    def _prepare_results(self):  # can be overwritten in children classes
        """Arrange samples for saving results."""
        all = np.concatenate([self.xyz, self.sdf[:, np.newaxis]], axis=1)
        self.results = {
            "all": all
        }
        if self.pseudo_normal:
            self.results["all_normal"] = self.normals
    
    def get_validity(self):
        """Return a dicitonary with validity information."""
        if self.results is None:
            raise RuntimeError("No results available to check validity of sampling.")
        return {
            "zero_sdf": self._zero_sdf()
        }
    
    def _zero_sdf(self, tol=1e-4):
        """Verify that SDF on surface is nearly zero."""
        return bool((np.abs(self.sdf) < tol).all())



class VoxelSampler(Sampler):
    """Return a voxel grid."""
    def __init__(self, mesh, pseudo_normal, resolution):
        super().__init__(mesh, pseudo_normal)
        self.resolution = int(resolution)

        # Build voxel grid
        xx = np.linspace(-1., 1., self.resolution)
        xyz = np.meshgrid(xx, xx, xx, indexing='ij')
        self._voxels = np.stack(xyz, axis=-1).reshape(-1, 3)
    
    def sample(self, n_samples):
        return self._voxels

    def _prepare_results(self):
        """Arrange samples for saving results."""
        pos_idx = self.sdf >= 0.
        neg_idx = ~pos_idx
        pos = np.concatenate([self.xyz[pos_idx], self.sdf[pos_idx, np.newaxis]], axis=1)
        neg = np.concatenate([self.xyz[neg_idx], self.sdf[neg_idx, np.newaxis]], axis=1)
        self.results = {
            "resolution": self.resolution,
            "pos": pos,
            "neg": neg
        }
        if self.pseudo_normal:
            self.results["pos_normal"] = self.normals[pos_idx]
            self.results["neg_normal"] = self.normals[neg_idx]
        # Add indices of points to form the voxel grid back
        indices = np.arange(len(self.xyz))
        self.results["pos_idx"] = indices[pos_idx]
        self.results["neg_idx"] = indices[neg_idx]

        # Indices can be used like the following to get the grid:
        # voxel = np.zeros((RES ** 3, 4))
        # voxel[npz['pos_idx']] = npz['pos']
        # voxel[npz['neg_idx']] = npz['neg']
        # voxel = voxel.reshape(RES, RES, RES, 4)


    
class WeightedSampler():
    """Weighted sampling using rejection sampling"""
    """Based on https://github.com/u2ni/ICML2021/blob/main/neuralImplicitTools/src/geometry.py"""  
    
    def __init__(self, mesh, beta):
        self.beta = beta
        return
    
    def sphere_sampling(self, n_samples):
        """Sample uniformly from a unit sphere"""
        rand_samples = np.random.rand(3, n_samples)

        r = rand_samples[0]
        phi = rand_samples[1] * np.pi * 2.
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        cos_theta = rand_samples[2] * 2. - 1.
        sin_theta = np.sqrt(1. - cos_theta ** 2)

        x = r * cos_phi * sin_theta
        y = r * sin_phi * sin_theta
        z = r * cos_theta

        return np.stack([x, y, z], axis=-1)

    
    def weight_func(self, dist_to_surf):
        """Calculate importance weight depending on abs(sdf)"""
        return np.exp(- self.beta * np.abs(dist_to_surf))
    
    
    def sample(self, mesh, n_uniform_samples, n_final_samples):
        """Perform weighted sampling"""
        uniform_samples = self.sphere_sampling(n_uniform_samples)
        
        pseudo_normal = False
        dist = igl.signed_distance(uniform_samples, mesh.vertices, mesh.faces, 
                                   return_normals=pseudo_normal)[0]
        samples_weight = self.weight_func(dist)
        
        # probabilities to choose each
        probs = samples_weight / np.sum(samples_weight)
        # exclusive sum
        C = np.concatenate(([0],np.cumsum(probs)))
        C = C[0:-1]
        # choose N random buckets
        R = np.random.rand(n_final_samples)
        # histc
        I = np.array(np.digitize(R,C)) - 1

        xyz = uniform_samples[I,:]
        sdf = dist[I].reshape(-1,1)

        return np.concatenate((xyz, sdf), axis=1)
    
    

# Utility functions for standardizing meshes
def mesh_mass_center(mesh):
    """Center the mesh at (0,0,0)"""
    return mesh.vertices.sum(axis=0) / mesh.vertices.shape[0]
        
def mesh_max_norm(mesh):
    """Normalize the mesh to a unit sphere"""
    return np.sqrt(np.power(np.array(mesh.vertices), 2).sum(axis=1)).max()
    
def make_mesh_canonical(mesh):
    """Canonalize the mesh = center + normalize to unit sphere"""
    mass_center = mesh_mass_center(mesh)
    mesh.vertices -= mass_center

    max_norm = mesh_max_norm(mesh)
    mesh.vertices /= max_norm
    return mesh



# Utility functions for standardizing meshes
def sample_sdf_from_mesh(
    mesh, 
    surface_sample_num      = 25000, 
    near_surface_sample_num = 250000,
    bigger_variance         = 0.005
):
    ps_normal = False # pseudo normal test
    mesh = make_mesh_canonical(mesh)
    
    sampler1 = SurfaceSampler(mesh, ps_normal)
    sampler1(surface_sample_num)

    # samples with bigger_variance, and with 1/10 of that
    # final size = 2*near_surface_sample_num
    sampler2 = NearSurfaceSampler(mesh, ps_normal, bigger_variance)
    sampler2(near_surface_sample_num) 

    dict1 = sampler1.get_result_dict()
    dict2 = sampler2.get_result_dict()
    
    all_sdf_samples = np.concatenate((
        dict1['all'],
        dict2['pos'],
        dict2['neg']
    ))
    
    # for newer versions of numpy
    #rng = np.random.default_rng()
    #rng.shuffle(all_sdf_samples)
    np.random.shuffle(all_sdf_samples)
    
    return all_sdf_samples


def weighted_sample_sdf_from_mesh(
    mesh, 
    beta,
    n_uniform_samples = 1000000,
    n_final_samples   =  500000
):
    mesh = make_mesh_canonical(mesh)    

    sampler = WeightedSampler(mesh, beta)
    all_sdf_samples = sampler.sample(mesh, n_uniform_samples, n_final_samples) 

    np.random.shuffle(all_sdf_samples)
    return all_sdf_samples


