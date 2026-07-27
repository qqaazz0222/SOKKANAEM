from sokkanaem.collapse import update_streak


def test_update_streak_increments_below_eps():
    s = 0
    for _ in range(5):
        s = update_streak(s, std=1e-6, eps=1e-4)
    assert s == 5


def test_update_streak_resets_above_eps():
    s = update_streak(3, std=1e-6, eps=1e-4)
    assert s == 4
    s = update_streak(s, std=0.5, eps=1e-4)
    assert s == 0
