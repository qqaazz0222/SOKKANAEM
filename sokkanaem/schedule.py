"""Training-time schedules (step -> value). None of these touch the model
or inference path — resolution curriculum picks the dataloader size, LR
schedule picks the step size. Both are pure functions of the global step
so resume just recomputes them (no scheduler state to serialize)."""
import math


def lr_at(step, base, total, warmup=2000):
    """Linear warmup then cosine decay to ~0. Stateless."""
    if step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


def parse_size_schedule(spec, default_size):
    """'step:size,step:size,...' -> sorted [(step, size), ...], step-0
    entry defaulted to default_size if the spec doesn't start there."""
    if not spec:
        return [(0, default_size)]
    pairs = sorted((int(s), int(sz))
                   for s, sz in (part.split(":") for part in spec.split(",")))
    if pairs[0][0] != 0:
        pairs.insert(0, (0, default_size))
    return pairs


def size_for_step(schedule, step):
    size = schedule[0][1]
    for thresh, sz in schedule:
        if step >= thresh:
            size = sz
    return size
