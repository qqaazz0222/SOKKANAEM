import torch

from sokkanaem.ema import ema_update_


def test_ema_update_moves_toward_model_weights():
    ema = {"w": torch.zeros(3)}
    model = {"w": torch.ones(3)}
    ema_update_(ema, model, decay=0.9)
    assert torch.allclose(ema["w"], torch.full((3,), 0.1))
    ema_update_(ema, model, decay=0.9)
    assert torch.allclose(ema["w"], torch.full((3,), 0.19), atol=1e-6)


def test_ema_update_copies_non_float_buffers():
    ema = {"n": torch.tensor(0)}
    model = {"n": torch.tensor(7)}
    ema_update_(ema, model, decay=0.9)
    assert ema["n"].item() == 7
