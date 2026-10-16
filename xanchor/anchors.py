from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Anchors:
    xyz: np.ndarray
    feat: np.ndarray
    support: np.ndarray
    cell: np.ndarray
    cell_size: float

    def __len__(self) -> int:
        return len(self.xyz)

    def select(self, min_support: int = 1) -> tuple[np.ndarray, np.ndarray]:
        keep = self.support >= min_support
        f = self.feat[keep].astype(np.float32)
        f = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)
        return self.xyz[keep], f


def compute_anchors(xyz: np.ndarray, feat, cell_size: float) -> Anchors:
    xyz = np.ascontiguousarray(np.asarray(xyz)[:, :3], dtype=np.float32)
    cell = np.floor(xyz.astype(np.float64) / cell_size).astype(np.int64)
    keys, inverse, counts = np.unique(cell, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    n = len(keys)
    centroid = np.zeros((n, 3), np.float64)
    np.add.at(centroid, inverse, xyz.astype(np.float64))
    centroid /= counts[:, None]
    f = feat if torch.is_tensor(feat) else torch.as_tensor(np.asarray(feat, np.float32))
    inv = torch.as_tensor(inverse, dtype=torch.long, device=f.device)
    s = torch.zeros((n, f.shape[1]), dtype=torch.float32, device=f.device).index_add_(0, inv, f.float())
    mean = (s / torch.as_tensor(counts, dtype=torch.float32, device=f.device)[:, None]).cpu().numpy()
    return Anchors(centroid.astype(np.float32), mean, counts.astype(np.int64), keys, float(cell_size))


def merge_anchors(parts: list[Anchors]) -> Anchors:
    sizes = {a.cell_size for a in parts}
    if len(sizes) != 1:
        raise ValueError(f'cannot merge anchors with different cell sizes: {sorted(sizes)}')
    return _merge(np.concatenate([a.xyz for a in parts]), np.concatenate([a.feat for a in parts]),
                  np.concatenate([a.support for a in parts]), np.concatenate([a.cell for a in parts]), sizes.pop())


def _merge(xyz, feat, support, cell, cell_size) -> Anchors:
    keys, inverse = np.unique(np.asarray(cell, np.int64), axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    n = len(keys)
    w = np.asarray(support, np.float64)[:, None]
    xs = np.zeros((n, 3)); fs = np.zeros((n, feat.shape[1])); c = np.zeros(n, np.int64)
    np.add.at(xs, inverse, np.asarray(xyz, np.float64) * w)
    np.add.at(fs, inverse, np.asarray(feat, np.float64) * w)
    np.add.at(c, inverse, np.asarray(support, np.int64))
    return Anchors((xs / c[:, None]).astype(np.float32), (fs / c[:, None]).astype(np.float32), c, keys, float(cell_size))
