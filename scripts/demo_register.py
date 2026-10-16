import argparse
import time
from pathlib import Path

import numpy as np
import torch

from xanchor import LEVELS, DescriptorEncoder, compute_anchors, merge_anchors, register


def load_points(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix == '.npy':
        return np.load(p)[:, :3].astype(np.float32)
    if p.suffix == '.bin':
        return np.fromfile(p, dtype=np.float32).reshape(-1, 4)[:, :3]
    import open3d as o3d
    return np.asarray(o3d.io.read_point_cloud(str(p)).points, dtype=np.float32)


def encode(encoder, paths, cell_size):
    clouds = [load_points(p) for p in paths]
    parts = [compute_anchors(x, encoder(x), cell_size) for x in clouds]
    return merge_anchors(parts), np.concatenate(clouds)


def main():
    ap = argparse.ArgumentParser(description='Register a source point cloud to a target point cloud with XAnchor-Reg.')
    ap.add_argument('--weights', required=True)
    ap.add_argument('--source', nargs='+', required=True, help='one or more files (.npy, .bin, .ply, .pcd) in a common frame')
    ap.add_argument('--target', nargs='+', required=True, help='one or more files (.npy, .bin, .ply, .pcd) in a common frame')
    ap.add_argument('--level', choices=sorted(LEVELS), default='submap')
    ap.add_argument('--refine', action='store_true', help='refine the coarse estimate with GICP')
    ap.add_argument('--show', action='store_true', help='visualize the result with Open3D')
    ap.add_argument('--out', help='save the 4x4 transform (source -> target) as .txt')
    a = ap.parse_args()
    cfg = LEVELS[a.level]

    encoder = DescriptorEncoder(a.weights)
    t0 = time.perf_counter()
    src, pts_src = encode(encoder, a.source, cfg['cell_size'])
    tgt, pts_tgt = encode(encoder, a.target, cfg['cell_size'])
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    refine = dict(points_source=pts_src, points_target=pts_tgt) if a.refine else {}
    result = register(src, tgt, min_support=cfg['min_support'], refine_max_distance=cfg['refine_max_distance'], **refine)
    t2 = time.perf_counter()
    if result is None:
        print('registration failed: fewer than 3 anchors or correspondences')
        return
    np.set_printoptions(precision=4, suppress=True)
    print(f'anchors: source {len(src)}, target {len(tgt)}')
    print(f'correspondences: {result.num_correspondences}, inliers: {result.num_inliers}')
    print(f'time: anchors {t1 - t0:.2f} s, registration {t2 - t1:.2f} s')
    print('transform (source -> target):')
    print(result.transform)
    if a.out:
        np.savetxt(a.out, result.transform)
    if a.show:
        import open3d as o3d
        s = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_src.astype(np.float64)))
        t = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_tgt.astype(np.float64)))
        s.paint_uniform_color([0.95, 0.6, 0.1]); t.paint_uniform_color([0.1, 0.5, 0.9])
        o3d.visualization.draw_geometries([s.transform(result.transform), t])


if __name__ == '__main__':
    main()
