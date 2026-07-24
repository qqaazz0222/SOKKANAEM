import math

from sokkanaem.schedule import lr_at, parse_size_schedule, size_for_step


def test_lr_at_warmup_ramps_from_zero():
    assert lr_at(0, 3e-4, 100, warmup=10) < 3e-4
    assert lr_at(9, 3e-4, 100, warmup=10) == 3e-4  # last warmup step hits base


def test_lr_at_cosine_decays_to_near_zero():
    assert math.isclose(lr_at(10, 3e-4, 100, warmup=10), 3e-4, rel_tol=1e-9)
    assert lr_at(100, 3e-4, 100, warmup=10) < 1e-9  # end of cosine


def test_lr_at_monotone_after_warmup():
    xs = [lr_at(s, 3e-4, 100, warmup=10) for s in range(10, 101)]
    assert all(a >= b - 1e-12 for a, b in zip(xs, xs[1:]))


def test_parse_size_schedule_default_is_fixed_size():
    assert parse_size_schedule(None, 256) == [(0, 256)]


def test_parse_size_schedule_inserts_step_zero():
    assert parse_size_schedule("30000:256", 128) == [(0, 128), (30000, 256)]


def test_parse_size_schedule_sorts():
    assert parse_size_schedule("30000:256,0:128", 128) == [(0, 128), (30000, 256)]


def test_size_for_step():
    sched = [(0, 128), (30000, 256)]
    assert size_for_step(sched, 0) == 128
    assert size_for_step(sched, 29999) == 128
    assert size_for_step(sched, 30000) == 256
    assert size_for_step(sched, 100000) == 256
