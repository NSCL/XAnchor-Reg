from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

INPUT_GRID_M = 0.3
BACKBONE_DIM = 1386


def _multiscale_levels(point):
    feats, invs = [point.feat], []
    while 'pooling_parent' in point.keys():
        invs.append(point['pooling_inverse'])
        point = point['pooling_parent']
        feats.append(point.feat)
    return feats, invs


def _project_per_scale(feats, invs, weight, bias):
    off = int(weight.shape[1])
    acc = None
    for k, f in enumerate(feats):
        c = int(f.shape[1])
        off -= c
        part = nn.functional.linear(f.float(), weight[:, off:off + c])
        acc = part if acc is None else part + acc[invs[k - 1]]
    return acc + bias


def _load_state_dict(path):
    sd = torch.load(str(path), map_location='cpu', weights_only=False)
    sd = sd.get('model', sd)
    if 'head_a.weight' not in sd:
        raise ValueError(f'{path}: not an XAnchor-Reg checkpoint (head_a.weight missing)')
    backbone = {k[len('backbone.'):]: v for k, v in sd.items() if k.startswith('backbone.')}
    head = {k[len('head_a.'):]: v for k, v in sd.items() if k.startswith('head_a.')}
    return backbone, head


class DescriptorEncoder:
    def __init__(self, weights: str | Path, device: str = 'cuda', enable_flash: bool | None = None):
        import utonia
        from utonia.transform import Compose
        if enable_flash is None:
            try:
                import flash_attn  # noqa: F401
                enable_flash = True
            except ImportError:
                enable_flash = False
        self.device = torch.device(device)
        backbone_sd, head_sd = _load_state_dict(weights)
        self.dim = int(head_sd['weight'].shape[0])
        self.backbone = utonia.model.load('utonia', repo_id='Pointcept/Utonia', custom_config=dict(enable_flash=enable_flash))
        missing, unexpected = self.backbone.load_state_dict(backbone_sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f'backbone weights mismatch: {len(missing)} missing, {len(unexpected)} unexpected')
        self.head = nn.Linear(BACKBONE_DIM, self.dim)
        self.head.load_state_dict(head_sd)
        self.backbone.to(self.device).eval()
        self.head.to(self.device).eval()
        self.transform = Compose([
            dict(type='GridSample', grid_size=INPUT_GRID_M, hash_type='fnv', mode='train',
                 return_grid_coord=True, return_inverse=True),
            dict(type='NormalizeColor'), dict(type='ToTensor'),
            dict(type='Collect', keys=('coord', 'grid_coord', 'inverse'), feat_keys=('coord', 'color', 'normal'))])

    @torch.inference_mode()
    def __call__(self, xyz: np.ndarray) -> torch.Tensor:
        x = np.ascontiguousarray(np.asarray(xyz)[:, :3], dtype=np.float32)
        if len(x) == 0:
            return torch.zeros((0, self.dim), device=self.device)
        center = np.array([*x[:, :2].mean(0), 0.0], dtype=np.float32)
        np.random.seed(0)
        data = self.transform({'coord': x - center, 'color': np.zeros_like(x), 'normal': np.zeros_like(x)})
        data = {k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in data.items()}
        data['batch'] = torch.zeros(len(data['coord']), dtype=torch.long, device=self.device)
        data['grid_size'] = INPUT_GRID_M
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            point = self.backbone(data)
        feats, invs = _multiscale_levels(point)
        return _project_per_scale(feats, invs, self.head.weight, self.head.bias)[data['inverse']].float()
