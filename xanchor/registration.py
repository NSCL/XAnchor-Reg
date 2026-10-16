from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch

from xanchor.anchors import Anchors
from xanchor.sc2pcr import SC2PCRMatcherOriginal

LEVELS = {
    'scan': dict(cell_size=1.0, min_support=1, refine_max_distance=0.5),
    'submap': dict(cell_size=3.0, min_support=3, refine_max_distance=1.0),
    'map': dict(cell_size=3.0, min_support=3, refine_max_distance=1.0),
}


@dataclass
class RegistrationResult:
    transform: np.ndarray
    coarse_transform: np.ndarray
    num_correspondences: int
    num_inliers: int
    refined: bool


@torch.no_grad()
def find_mnn_correspondences(desc_src, desc_tgt, budget: int = 8000, device=None, rows: int = 2048):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A = torch.as_tensor(np.asarray(desc_src, np.float32), device=device)
    B = torch.as_tensor(np.asarray(desc_tgt, np.float32), device=device)
    if len(A) == 0 or len(B) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    row_v, row_j = [], []
    col_v = torch.full((len(B),), -float('inf'), device=device)
    col_i = torch.zeros(len(B), dtype=torch.long, device=device)
    for i in range(0, len(A), rows):
        S = A[i:i + rows] @ B.T
        v, j = S.max(1)
        row_v.append(v); row_j.append(j)
        cv, ci = S.max(0)
        better = cv > col_v
        col_v = torch.where(better, cv, col_v)
        col_i = torch.where(better, ci + i, col_i)
    row_v = torch.cat(row_v); row_j = torch.cat(row_j)
    mutual = torch.nonzero(col_i[row_j] == torch.arange(len(A), device=device)).squeeze(1)
    sel = mutual[torch.argsort(row_v[mutual], descending=True, stable=True)[:budget]]
    return sel.cpu().numpy(), row_j[sel].cpu().numpy()


def estimate_transform(p_src: np.ndarray, q_tgt: np.ndarray, inlier_threshold: float = 1.0, budget: int = 8000, device=None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    matcher = SC2PCRMatcherOriginal(inlier_threshold=inlier_threshold, d_thre=inlier_threshold, max_points=max(budget, 8000))
    T, _ = matcher.SC2_PCR(torch.tensor(p_src, dtype=torch.float32, device=device)[None],
                           torch.tensor(q_tgt, dtype=torch.float32, device=device)[None])
    T = T[0].detach().cpu().numpy().astype(np.float64)
    res = np.linalg.norm(p_src.astype(np.float64) @ T[:3, :3].T + T[:3, 3] - q_tgt, axis=1)
    return T, int((res < inlier_threshold).sum())


def refine_gicp(points_src: np.ndarray, points_tgt: np.ndarray, init: np.ndarray, max_distance: float, num_threads: int = 0):
    try:
        import small_gicp as sg
    except ImportError:
        import open3d as o3d
        ps = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points_src, np.float64)))
        pt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points_tgt, np.float64)))
        est = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()
        return np.asarray(o3d.pipelines.registration.registration_generalized_icp(ps, pt, max_distance, init, est).transformation)
    n = num_threads if num_threads > 0 else min(os.cpu_count() or 1, 24)
    clouds = []
    for xyz in (points_tgt, points_src):
        pc = sg.PointCloud(np.ascontiguousarray(xyz, dtype=np.float64))
        tree = sg.KdTree(pc, num_threads=n)
        sg.estimate_covariances(pc, tree, num_neighbors=20, num_threads=n)
        clouds.append((pc, tree))
    (target, target_tree), (source, _) = clouds
    r = sg.align(target, source, target_tree, np.asarray(init, np.float64), registration_type='GICP',
                 max_correspondence_distance=float(max_distance), num_threads=n, max_iterations=30)
    return np.asarray(r.T_target_source, dtype=np.float64)


def register(source: Anchors, target: Anchors, min_support: int = 3, budget: int = 8000, inlier_threshold: float = 1.0,
             points_source: np.ndarray | None = None, points_target: np.ndarray | None = None,
             refine_max_distance: float = 1.0, device=None) -> RegistrationResult | None:
    xyz_s, desc_s = source.select(min_support)
    xyz_t, desc_t = target.select(min_support)
    if len(xyz_s) < 3 or len(xyz_t) < 3:
        return None
    i, j = find_mnn_correspondences(desc_s, desc_t, budget, device)
    if len(i) < 3:
        return None
    T, n_inliers = estimate_transform(xyz_s[i], xyz_t[j], inlier_threshold, budget, device)
    T_final, refined = T, False
    if points_source is not None and points_target is not None:
        T_final, refined = refine_gicp(points_source, points_target, T, refine_max_distance), True
    return RegistrationResult(T_final, T, int(len(i)), n_inliers, refined)
