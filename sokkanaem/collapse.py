"""Constant-output collapse detection. A mis-weighted loss term with a
trivial degenerate minimizer (e.g. temporal_loss's "predict a constant")
can make training converge to zero-variance output; catch it early rather
than burning the full run on a checkpoint that's useless afterward."""


def update_streak(streak, std, eps):
    return streak + 1 if std < eps else 0
