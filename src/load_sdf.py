"""
Data module for SDF samples.
"""

import os, os.path
import numpy as np


# new version
def load_shuffled_sdf_samples_from_file(filename, n_samples):
    npz = np.load(filename)
    samples = np.random.permutation(npz)[:n_samples]

    xyz = samples[:, 0:3]
    sdf = samples[:, 3:4]

    return xyz, sdf


# old version
def load_samples_from_file(filename, n_samples):
    """
    Load samples from a file.
    
    Samples are:
        - 500K near-surface points
        - 25K uniform points
    """
    # Load the samples files
    npz = np.load(filename)
    pos, neg = npz['pos'], npz['neg']

    # Samples a balance between pos and neg
    pos = np.random.permutation(pos)[:n_samples//2]
    neg = np.random.permutation(neg)[:n_samples//2]

    samples = np.concatenate([pos, neg], 0)
    xyz = samples[:, 0:3]
    sdf = samples[:, 3:4]

    return xyz, sdf