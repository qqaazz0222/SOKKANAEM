"""PLAN §7's quality-guarantee track: DA2's shape, our metric scale.

The native 4.19M track measured its own ceiling (REPORT §4.44): capacity does
not move the error (M0 at 10.8M is worse), the boundary term buys sharpness
only up to a ringing wall, and the Bonn gap is uniform across every region --
1.55-1.79x DA2 in edge, flat, dynamic, static, near, mid and far alike. A
uniform ratio is not something a region-weighted loss can close; it says the
shape model itself is weaker. So this track stops trying to grow one and
starts from a shape model that already passes: Depth Anything V2 Small.

Q0 (this file, and all that is needed to seal the floor): the DA2 stack is
FROZEN and produces relative disparity exactly as the published model does;
the only trained parameters are a per-frame calibration head that maps that
disparity to metres,

    disp_metric = softplus(scale) * disp_rel + shift
    depth       = 1 / clamp(disp_metric, eps)

with scale/shift read from the backbone's own pooled features. Two numbers per
frame, ~25k parameters. If this does not clear the accuracy gate, no amount of
fine-tuning the shape branch will -- and if it does, the gate is reachable and
Q1/Q2 (unfreezing with a retention loss, then distilling into a small student)
have something to aim at.

Inference runs at DA2's official 518px and the depth is resampled to the
caller's resolution, which is exactly how the baseline rows in this repo were
measured -- the comparison is then about the model, not about a resize.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

HF_ID = "depth-anything/Depth-Anything-V2-Small-hf"
# ImageNet statistics, the normalization DA2's own processor applies
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class TemporalAdapter(nn.Module):
    """Zero-init residual over a per-stage running state (PLAN §4.4).

    Sits between DA2's backbone and its neck, so the frozen shape branch is
    untouched and a fresh adapter reproduces Q0's output exactly -- the floor
    this track sealed cannot be lost by adding streaming. The state is a
    learned-rate EMA of each stage's feature map; the residual reads current
    and remembered features together. What it is FOR is temporal stability and,
    later, sparse updates: REPORT §4.43 measured that recurrent state does not
    create per-frame accuracy, so it is not asked to."""

    def __init__(self, chans):
        super().__init__()
        # the backbone hands back TOKEN sequences (N, L, C), not feature maps;
        # the neck does the reshape, so the adapter stays token-wise
        self.mix = nn.ModuleList(nn.Linear(2 * c, c) for c in chans)
        for m in self.mix:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        # sigmoid(0) = 0.5: half current frame, half memory, and trainable
        self.rate = nn.Parameter(torch.zeros(len(chans)))

    def forward(self, feats, state):
        out, new = [], []
        for i, (f, mix) in enumerate(zip(feats, self.mix)):
            s = f if state is None else state[i]
            out.append(f + mix(torch.cat([f, s], -1)))
            a = torch.sigmoid(self.rate[i])
            new.append(a * f + (1 - a) * s)
        return out, new


class QDepth(nn.Module):
    """Frozen DA2 shape + trained metric calibration. Streaming-compatible:
    `forward_clip` and `step` mirror SOKKANAEM's signatures so the existing
    evaluators, which know how to drive our model, drive this one unchanged."""

    def __init__(self, hf_id=HF_ID, infer_size=518, freeze_shape=True,
                 d_min=0.3, d_max=150.0, temporal=False, chunk=8):
        super().__init__()
        from transformers import AutoModelForDepthEstimation

        self.net = AutoModelForDepthEstimation.from_pretrained(hf_id)
        self.hf_id, self.infer_size = hf_id, infer_size
        # frames per forward. The long-stream manifest hands over 256-frame
        # clips and DA2's head at 518px wants 5.35 GiB for that in one batch --
        # which is how the first streaming evaluation died. Chunking is exact:
        # frames are independent on this path.
        self.chunk = chunk
        self.freeze_shape = freeze_shape
        self.d_min, self.d_max = d_min, d_max
        if freeze_shape:
            self.net.eval()
            for p in self.net.parameters():
                p.requires_grad_(False)
        hidden = self.net.config.backbone_config.hidden_size
        self.calib = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, 64), nn.GELU(),
            nn.Linear(64, 2))
        # The calibration acts on PER-FRAME NORMALIZED disparity (median
        # centre, mean-absolute-deviation scale), not on DA2's raw output: the
        # raw scale is arbitrary, so a head reading it would have to learn each
        # frame's gauge before it could learn any metric. Normalized, the two
        # numbers mean something fixed, and the init below (scale 0.2, shift
        # 0.3 in disparity) starts every frame at a ~3.3 m median depth --
        # indoor range, where TUM and Bonn live.
        nn.init.zeros_(self.calib[-1].weight)
        with torch.no_grad():
            self.calib[-1].bias.copy_(torch.tensor([-1.5, 0.3]))
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))
        self.p = 16          # evaluators ask for a patch size; masks are dense
        self.dim = hidden
        self.temporal = None
        if temporal:
            chans = self.net.config.backbone_config.hidden_size
            n_stage = len(self.net.config.backbone_config.out_indices) \
                if getattr(self.net.config.backbone_config, "out_indices", None) \
                else 4
            self.temporal = TemporalAdapter([chans] * n_stage)

    def _stages(self, x):
        """Backbone feature maps for a normalized batch, and the patch grid."""
        out = self.net.backbone.forward_with_filtered_kwargs(
            x, output_hidden_states=False, output_attentions=False)
        ph = x.shape[-2] // self.net.config.patch_size
        pw = x.shape[-1] // self.net.config.patch_size
        return list(out.feature_maps), (ph, pw)

    def _decode(self, feats, grid):
        h = self.net.neck(feats, *grid)
        d = self.net.head(h, *grid)
        return d.unsqueeze(1) if d.dim() == 3 else d

    def _shape(self, frames):
        """frames (N,3,H,W) in [0,1] -> (relative disparity at HxW, pooled feat).

        Run at DA2's own 518px and resampled back: the published quality is a
        property of that resolution, and every baseline row in this repo was
        measured the same way."""
        H, W = frames.shape[-2:]
        x = F.interpolate(frames, size=(self.infer_size, self.infer_size),
                          mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        ctx = torch.no_grad() if self.freeze_shape else torch.enable_grad()
        with ctx:
            out = self.net(pixel_values=x, output_hidden_states=True)
        disp = out.predicted_depth                      # (N, h, w), relative
        if disp.dim() == 3:
            disp = disp.unsqueeze(1)
        disp = F.interpolate(disp, size=(H, W), mode="bilinear",
                             align_corners=False)
        feat = out.hidden_states[-1][:, 1:].mean(1)     # drop CLS, mean patches
        return disp, feat

    def _prep(self, frames):
        H, W = frames.shape[-2:]
        x = F.interpolate(frames, size=(self.infer_size, self.infer_size),
                          mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std, (H, W)

    def _calibrate(self, disp, feat, size):
        from .sharpness import norm_field

        disp = F.interpolate(disp, size=size, mode="bilinear",
                             align_corners=False)
        dn = norm_field(disp, torch.ones_like(disp))
        s, b = self.calib(feat.float()).unbind(-1)
        disp_m = (F.softplus(s).view(-1, 1, 1, 1) * dn + b.view(-1, 1, 1, 1))
        return (1.0 / disp_m.clamp(min=1.0 / self.d_max)).clamp(
            min=self.d_min, max=self.d_max)

    def forward_stream(self, frame, state=None):
        """One frame with the temporal adapter in the loop.

        Returns (depth, state). The backbone runs under no_grad whenever the
        shape branch is frozen, so an adapter run costs one DA2 forward plus
        four 1x1 convs."""
        x, size = self._prep(frame)
        ctx = torch.no_grad() if self.freeze_shape else torch.enable_grad()
        with ctx:
            feats, grid = self._stages(x)
        if self.temporal is not None:
            feats, state = self.temporal(feats, state)
            # truncated BPTT: the state carries VALUES across frames, not the
            # graph. Keeping the graph meant one 518px DPT head activation set
            # per frame of the clip and 8 frames did not fit in 24 GB; the
            # adapter still learns, because each frame's residual is
            # differentiable where it is applied.
            state = [x.detach() for x in state]
        with ctx if self.temporal is None else torch.enable_grad():
            disp = self._decode(feats, grid)
        pooled = feats[-1][:, 1:].mean(1) if feats[-1].shape[1] % 2 \
            else feats[-1].mean(1)
        return self._calibrate(disp, pooled, size), state

    def forward(self, frames):
        """frames (N,3,H,W) in [0,1] -> metric depth (N,1,H,W)."""
        from .sharpness import norm_field

        if self.temporal is not None:      # single-frame call, no memory
            return self.forward_stream(frames)[0]
        if frames.shape[0] > self.chunk:
            return torch.cat([self(frames[i:i + self.chunk])
                              for i in range(0, frames.shape[0], self.chunk)])
        disp, feat = self._shape(frames)
        dn = norm_field(disp, torch.ones_like(disp))
        s, b = self.calib(feat.float()).unbind(-1)
        disp_m = (F.softplus(s).view(-1, 1, 1, 1) * dn
                  + b.view(-1, 1, 1, 1))
        return (1.0 / disp_m.clamp(min=1.0 / self.d_max)).clamp(
            min=self.d_min, max=self.d_max)

    def forward_clip(self, clip, force_mask=None, return_tokens=False):
        """clip (B,T,3,H,W) -> (depths (B,T,1,H,W), masks (B,T,N) all ones).

        Q0 is dense by construction: PLAN §7 seals the quality floor with every
        frame on the DA2 path before any sparsity is reintroduced."""
        B, T = clip.shape[:2]
        n = (clip.shape[-2] // self.p) * (clip.shape[-1] // self.p)
        if self.temporal is None:
            d = self(clip.reshape(B * T, *clip.shape[-3:]))
        else:
            state, out = None, []
            for t in range(T):
                dep, state = self.forward_stream(clip[:, t], state)
                out.append(dep)
            d = torch.stack(out, 1).reshape(B * T, 1, *clip.shape[-2:])
        return (d.reshape(B, T, 1, *clip.shape[-2:]),
                torch.ones(B, T, n, device=clip.device))

    @torch.no_grad()
    def step(self, frame, state=None):
        """Single-frame streaming API, carrying the adapter's state when it
        exists (and holding None when it does not, which is Q0's dense floor)."""
        if self.temporal is not None:
            d, state = self.forward_stream(frame, state)
        else:
            d = self(frame)
        n = (frame.shape[-2] // self.p) * (frame.shape[-1] // self.p)
        return d, state, {"mask": torch.ones(frame.shape[0], n,
                                             device=frame.device)}


def from_q_checkpoint(ckpt, device="cpu", **overrides):
    from pathlib import Path

    from .model import checkpoint_config

    kw = dict(checkpoint_config(ckpt).get("model", {}))
    kw.pop("kind", None)
    kw.update(overrides)
    model = QDepth(**kw).to(device)
    state = torch.load(ckpt, map_location=device)
    sd = state.get("ema") or state.get("model") or state
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f"{Path(ckpt).name}: unexpected tensors {unexpected[:3]}"
    return model
