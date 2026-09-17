"""sokkanaem.qflat_state.flat_band_loss: only the win_lo..win_hi pixel band is charged.

Run directly (pytest collection trips over the dead paper_work symlink):
  python tests/test_flat_band_loss.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sokkanaem.qflat_state import flat_band_loss, flat_plane_loss

H = W = 200


def depths(field):
    """A (1,1,H,W) depth map whose normalized disparity follows `field`."""
    return 1.0 / (field + 2.0)


def main():
    y, x = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
    flat = torch.zeros(1, 1, H, W)
    ramp = (0.004 * x + 0.002 * y).reshape(1, 1, H, W)
    # period ~4px: finer than win_lo=5, should pass through mostly untaxed (boundary/noise scale)
    fine = (0.05 * torch.sin(x * 1.57) * torch.sin(y * 1.57)).reshape(1, 1, H, W)
    # period ~10px: inside the 5..17 band, should be taxed hardest
    mid = (0.05 * torch.sin(x * 0.63) * torch.sin(y * 0.63)).reshape(1, 1, H, W)
    # period ~120px: much coarser than win_hi=17, should pass through mostly untaxed (slope-like)
    coarse = (0.05 * torch.sin(x * 0.0524) * torch.sin(y * 0.0524)).reshape(1, 1, H, W)
    valid = torch.ones(1, 1, H, W)
    mask = torch.zeros(1, 1, H, W, dtype=torch.bool)
    mask[..., 20:-20, 20:-20] = True     # interior only: border windows are truncated

    same = flat_band_loss(depths(ramp), depths(ramp), valid, mask)
    assert same < 1e-6, f"identical maps must cost nothing, got {same}"

    slope = flat_band_loss(depths(ramp), depths(flat), valid, mask)
    assert slope < 1e-3, f"a pure ramp must be free (both box scales track it equally), got {slope}"

    fine_cost = flat_band_loss(depths(fine), depths(flat), valid, mask)
    mid_cost = flat_band_loss(depths(mid), depths(flat), valid, mask)
    coarse_cost = flat_band_loss(depths(coarse), depths(flat), valid, mask)
    assert mid_cost > 5 * fine_cost, f"in-band content must cost far more than sub-band, got {mid_cost} vs {fine_cost}"
    assert mid_cost > 5 * coarse_cost, f"in-band content must cost far more than supra-band, got {mid_cost} vs {coarse_cost}"

    # flat_plane_loss's high-pass has no upper cutoff: fine content costs it almost as much as mid
    plane_fine = flat_plane_loss(depths(fine), depths(flat), valid, mask, win=9)
    plane_mid = flat_plane_loss(depths(mid), depths(flat), valid, mask, win=9)
    assert plane_fine > 0.3 * plane_mid, (
        f"sanity: flat_plane_loss should NOT reject fine content the way flat_band_loss does, "
        f"got fine {plane_fine} vs mid {plane_mid}")

    print("flat_band_loss: identical %.2e | ramp %.2e | fine %.4f | mid %.4f | coarse %.4f | "
          "(plane_loss fine %.4f mid %.4f, for contrast)"
          % (same, slope, fine_cost, mid_cost, coarse_cost, plane_fine, plane_mid))
    print("TEST_OK")


if __name__ == "__main__":
    main()
