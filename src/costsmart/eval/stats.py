"""Stats stubs: bootstrap confidence intervals + McNemar paired test.

Stdlib-only implementations (random + math) so the pilot runs without
scipy; swap in scipy when the full analysis slice lands.
"""

from __future__ import annotations

import math
import random


def bootstrap_ci(
    values: list[float],
    stat: str = "mean",
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Bootstrap percentile CI for the mean (or median) of values."""
    if not values:
        return {"estimate": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    rng = random.Random(seed)
    n = len(values)
    if stat == "median":
        estimate = sorted(values)[n // 2]
    else:
        estimate = sum(values) / n
    resampled = []
    for _ in range(max(1, n_resamples)):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        resampled.append(sorted(sample)[n // 2] if stat == "median" else sum(sample) / n)
    resampled.sort()
    alpha = 1 - ci
    lo = resampled[int(alpha / 2 * len(resampled))]
    hi = resampled[min(len(resampled) - 1, int((1 - alpha / 2) * len(resampled)))]
    return {"estimate": estimate, "lo": lo, "hi": hi, "n": n}


def mcnemar(b: int, c: int) -> dict:
    """McNemar test stub for paired correct/incorrect outcomes.

    b = A wrong / B right, c = A right / B wrong. Uses the chi-square
    approximation with continuity correction; exact binomial for b + c < 25.
    """
    n = b + c
    if n == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n_discordant": 0}
    if n < 25:
        # Exact two-sided binomial p-value.
        prob = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n
        return {"statistic": float(abs(b - c)), "p_value": min(1.0, 2 * prob / 2), "n_discordant": n}
    stat = (abs(b - c) - 1) ** 2 / n
    # chi2(1) survival via erfc approximation.
    p_value = math.erfc(math.sqrt(stat / 2))
    return {"statistic": stat, "p_value": p_value, "n_discordant": n}
