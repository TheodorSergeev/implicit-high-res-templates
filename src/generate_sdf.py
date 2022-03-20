"""
Generate sdf samples for given meshes.
It can also save normals (as computed by IGL), but that changes
the SDF computation to use pseudo-normals which is not robust to unclean meshes.

Sampling types should be amongst:
- "uniform" / "uniform_{SHAPE}"   where {SHAPE} is "cube" or "sphere" / "ball" (default="cube")
- "nearsurface" / "nearsurface_{VAR}"   where {VAR} is the variance of the gaussian noise (default=0.005)
- "surface"
- "voxel_{RES}"   where {RES} is the desired resolution.

Used on meshes contained in `1_normalized`.
"""

# TODO: add multiprocessing to parallelized over meshes
# (can be done outside of this script by creating multiple non-overlapping splits)

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


def main(args):
    np.random.seed(args.seed)

    source, dest = args.source, args.dest
    filenames = os.listdir(source)
    os.makedirs(dest, exist_ok=True)

    if args.split is not None:
        with open(args.split) as f:
            split = json.load(f)
        filenames = [fn for fn in filenames if fn in split]

    n_shapes = len(filenames)
    if args.verbose:
        print(f"{n_shapes} shapes to process:")
    for i, filename in enumerate(filenames):
        if args.verbose and (i+1) % 100 == 0:
            print(f"Generating for shape {i+1}/{n_shapes}...")

        mesh = trimesh.load(os.path.join(source, filename))
        destdir = os.path.join(dest, os.path.splitext(filename)[0])
        os.makedirs(destdir, exist_ok=True)

        # Mesh validity (watertight, enough negative SDF samples, ...)
        valid_fn = os.path.join(destdir, "valid.json")
        if os.path.isfile(valid_fn):
            with open(valid_fn) as f:
                validity = json.load(f)
        else:
            validity = {
                "watertight": mesh.is_watertight,
                "winding_consistent": mesh.is_winding_consistent,
                "sdf" : {}
            }

        for sampling in args.sampling:
            sample_fn = os.path.join(destdir, sampling + ".npz")

            if args.skip and os.path.isfile(sample_fn):
                continue

            # Sample points and compute their SDF
            sampler = Sampler.get_sampler(sampling, mesh, args.pseudo_normal)
            sampler(args.n_samples)

            # Test validity of sampling-based SDF
            validity["sdf"][sampling] = sampler.get_validity()

            # Save samples
            tosave = sampler.get_result_dict()
            np.savez(sample_fn, **tosave)
        
        # Save validity
        with open(valid_fn, "w") as f:
            json.dump(validity, f, indent=2)

    if args.verbose:
        print(f"Results saved in {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SDF samples for meshes contained in a directory.")
    parser.add_argument("source", help="source directory containing the meshes")
    parser.add_argument("dest", help="destination directory to save the generated samples (one subdir will be created for each mesh in source)")
    parser.add_argument("-s", "--sampling", nargs="+", required=True, help="all the types of sampling to use for generation, "
                                                                           "results are saved in separate files (see script docstring)")

    parser.add_argument("--double", action="store_true", help="save samples in double resolution (float64 instead of float32)")
    parser.add_argument("-n", "--n-sample", default=250000, help="number of samples to generate per sampling type (unless fixed, e.g. for voxels)")
    parser.add_argument("--pseudo-normal", action="store_true", help="compute the SDF using the pseudo-normal test, and also saves the normals")
    parser.add_argument("--split", default=None, help="path to a JSON file containing a list of mesh to process from `source` (MUST containt the extension, e.g. '.obj')")
    parser.add_argument("--seed", default=0, help="seed for the RNGs")
    parser.add_argument("--skip", action="store_true", help="skip a type of generation if the result file already exists (note: this will skip it "
                                                            "anyway even if samples were generated with a different method, see --pseudo-normal)")
    parser.add_argument("-v", "--verbose", action="store_true", help="increase script verbosity")

    args = parser.parse_args()

    main(args)