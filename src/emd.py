from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import sample_sdf

def emd(X_source, X_target):
    d = cdist(X_source, X_target)
    assignment = linear_sum_assignment(d)
    return d[assignment].sum() / min(len(X_source), len(X_target))

def emd_mesh(gt_surf_samples, mesh_target, N_target=10000):
    sample_arr2 = sample_sdf.sample_sdf_from_mesh_surface(mesh_target, N_target)[:,0:3]
    return emd(gt_surf_samples, sample_arr2)