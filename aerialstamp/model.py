"""The AerialStamp network (DOYModel of the training code, inference only) and
the frozen DINOv2 patch embedding."""
import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import BACKBONE, NBINS

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")


class ViewEncoder(nn.Module):
    """Patch embeddings -> layer norm, projection, attention pooling (one query)."""

    def __init__(self, emb_dim, D, pool="attn", n_scales=1):
        super().__init__()
        self.pool = pool
        self.emb = nn.Sequential(nn.LayerNorm(emb_dim), nn.Dropout(0.0), nn.Linear(emb_dim, D))
        self.scale_emb = nn.Parameter(torch.zeros(n_scales, D))
        if pool == "attn":
            self.q = nn.Parameter(torch.zeros(1, 1, D))
            self.att = nn.MultiheadAttention(D, 4, batch_first=True)
            self.out = nn.Linear(D, D)
        else:
            self.out = nn.Linear(D * n_scales, D)

    def forward(self, e, v):
        t = self.emb(e.float()) + self.scale_emb[0]
        if self.pool == "attn":
            allmiss = ~v.any(1)
            M = v.clone()
            M[allmiss, 0] = True
            o, _ = self.att(self.q.expand(t.shape[0], -1, -1), t, t, key_padding_mask=~M)
            return self.out(o[:, 0] * (~allmiss).float()[:, None])
        mf = v.float()[..., None]
        return self.out((t * mf).sum(1) / mf.sum(1).clamp(min=1))


class DOYModel(nn.Module):
    """Seasonal head (+ Sentinel-2 head when cfg['s2']) -> 365 day logits."""

    def __init__(self, cfg, emb_dim, s2_dim, static_dim):
        super().__init__()
        D = cfg["D"]
        self.cfg = cfg
        self.views = ViewEncoder(emb_dim, D, cfg["pool"])
        self.static = nn.Sequential(nn.Linear(static_dim, 64), nn.GELU(), nn.Linear(64, 64))
        self.trunk = nn.Sequential(nn.Linear(D + 64, 256), nn.GELU(), nn.Dropout(0.1),
                                   nn.Linear(256, 256), nn.GELU())
        ang = 2 * np.pi * (np.arange(NBINS) + 0.5) / NBINS
        four = np.concatenate([np.stack([np.sin(k * ang), np.cos(k * ang)], 1)
                               for k in range(1, 5)], 1)
        self.register_buffer("four", torch.tensor(four, dtype=torch.float32))
        self.day_in = nn.Linear(8, 96)
        self.day_ctx = nn.Linear(256, 96)
        self.day_out = nn.Sequential(nn.GELU(), nn.Linear(96, 96), nn.GELU(), nn.Linear(96, 1))
        if cfg["s2"]:
            self.wk_in = nn.Linear(s2_dim + 3, 128)
            self.wk_ctx = nn.Linear(256, 128)
            self.conv1 = nn.Conv1d(128, 128, 5, padding=2)
            self.conv2 = nn.Conv1d(128, 128, 5, padding=2)
            self.wk_out = nn.Linear(128, 1)
            self.log_bw = nn.Parameter(torch.tensor(np.log(4.0), dtype=torch.float32))
            self.register_buffer("wk_keep", torch.ones(s2_dim))

    def forward(self, static, emb, valid, s2w=None, s2pos=None, s2m=None):
        z_s = self.static(static)
        h = self.trunk(torch.cat([self.views(emb, valid), z_s], 1))
        B = h.shape[0]
        x = self.day_in(self.four[None].expand(B, -1, -1)) + self.day_ctx(h)[:, None]
        logits = self.day_out(x)[..., 0]
        if self.cfg["s2"]:
            ang = 2 * np.pi * s2pos / NBINS
            wk = torch.cat([s2w.float() * self.wk_keep, torch.sin(ang)[..., None],
                            torch.cos(ang)[..., None], (s2pos / NBINS)[..., None]], -1)
            t = F.gelu(self.wk_in(wk) + self.wk_ctx(h)[:, None])
            t = (t * s2m[..., None]).transpose(1, 2)
            t = F.gelu(self.conv1(t)) * s2m[:, None]
            t = F.gelu(self.conv2(t)) * s2m[:, None]
            s = self.wk_out(t.transpose(1, 2))[..., 0]
            days = torch.arange(NBINS, device=s2pos.device).float() + 0.5
            bw = self.log_bw.exp()
            k = torch.exp(-0.5 * ((days[None, :, None] - s2pos[:, None, :]) / bw)**2)
            k = k * s2m[:, None, :].float()
            logits = logits + (k * s[:, None, :]).sum(-1) / (k.sum(-1) + 0.1)
        return logits


def load_ensemble(kind, device="cpu", weights_dir=WEIGHTS_DIR):
    """kind 's2' or 'image' -> list of models, calibration dict."""
    files = sorted(glob.glob(os.path.join(weights_dir, kind, "seed*.pt")))
    if not files:
        raise FileNotFoundError(f"no weights in {weights_dir}/{kind}")
    models = []
    for f in files:
        ck = torch.load(f, map_location="cpu", weights_only=False)
        m = DOYModel(ck["cfg"], ck["emb_dim"], ck["s2_dim"], ck["static_dim"])
        m.load_state_dict(ck["state"])
        models.append(m.to(device).eval())
    cal = json.load(open(os.path.join(weights_dir, "calibration.json")))[kind]
    return models, cal


@torch.no_grad()
def predict(models, static, emb, valid, s2w=None, s2pos=None, s2m=None, device="cpu"):
    """Mean of the seeds' softmax distributions, (365,) float64."""
    def tt(a, dtype=torch.float32):
        return None if a is None else torch.as_tensor(np.asarray(a)[None], dtype=dtype,
                                                      device=device)
    args = (tt(static), tt(emb), tt(valid, torch.bool), tt(s2w), tt(s2pos), tt(s2m, torch.bool))
    ps = [F.softmax(m(*args), 1)[0].double().cpu().numpy() for m in models]
    p = np.mean(ps, 0)
    return p / p.sum()


def load_backbone(device="cpu"):
    """Frozen DINOv2 ViT-B/14 with registers -> embed(imgs (N, 224, 224, 3) uint8)
    -> (N, 1536): class token and mean patch token."""
    import timm
    half = str(device).startswith("cuda")
    model = timm.create_model(BACKBONE, pretrained=True, num_classes=0, img_size=224)
    model = model.to(device).eval()
    if half:
        model = model.half()
    cfg = model.default_cfg
    mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
    npre = model.num_prefix_tokens

    @torch.no_grad()
    def embed(imgs):
        x = torch.from_numpy(np.ascontiguousarray(imgs)).to(device).permute(0, 3, 1, 2).float() / 255
        x = (x - mean) / std
        tok = model.forward_features(x.half() if half else x)
        e = torch.cat([tok[:, 0], tok[:, npre:].mean(1)], 1)
        return e.float().cpu().numpy()

    return embed
