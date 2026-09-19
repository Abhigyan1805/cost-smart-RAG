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
    """McNemar test for paired correct/incorrect outcomes.

    b = A wrong / B right, c = A right / B wrong. Uses the chi-square
    approximation with continuity correction; exact two-sided binomial
    for b + c < 25.
    """
    n = b + c
    if n == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n_discordant": 0}
    if n < 25:
        # Exact two-sided binomial p-value: 2 * lower-tail at min(b, c).
        tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n
        return {"statistic": float(abs(b - c)), "p_value": min(1.0, 2 * tail),
                "n_discordant": n}
    stat = (abs(b - c) - 1) ** 2 / n
    # chi2(1) survival via erfc approximation.
    p_value = math.erfc(math.sqrt(stat / 2))
    return {"statistic": stat, "p_value": p_value, "n_discordant": n}


# ---------------------------------------------------------------------------
# Headroom inference (costheadroom-09): bootstrap CIs + paired comparisons.
# ---------------------------------------------------------------------------

#: Headroom resample budget: 10k draws per the firstmate spec.
HEADROOM_RESAMPLES = 10_000


def _percentile(sorted_vals: list[float], ci: float) -> tuple[float, float]:
    alpha = 1 - ci
    m = len(sorted_vals)
    lo = sorted_vals[max(0, int(alpha / 2 * m))]
    hi = sorted_vals[min(m - 1, int((1 - alpha / 2) * m))]
    return lo, hi


def bootstrap_proportion_ci(
    k: int,
    n: int,
    n_resamples: int = HEADROOM_RESAMPLES,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Bootstrap percentile CI for a binomial proportion k/n.

    Resamples Bernoulli(p=k/n) draws (equivalent to resampling the 0/1
    outcomes); exact Clopper-Pearson-style coverage is NOT claimed at
    small n - the pilot gate states the small-n caveat explicitly.
    """
    if n <= 0:
        return {"estimate": 0.0, "lo": 0.0, "hi": 0.0, "n": 0,
                "n_resamples": n_resamples, "ci": ci}
    rng = random.Random(seed)
    p = k / n
    estimate = p
    draws = []
    for _ in range(max(1, n_resamples)):
        draws.append(sum(1 for _ in range(n) if rng.random() < p) / n)
    draws.sort()
    lo, hi = _percentile(draws, ci)
    return {"estimate": estimate, "lo": lo, "hi": hi, "n": n,
            "n_resamples": n_resamples, "ci": ci}


def bootstrap_paired_diff_ci(
    a_correct: list[bool | int],
    b_correct: list[bool | int],
    n_resamples: int = HEADROOM_RESAMPLES,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Bootstrap CI for the paired accuracy difference mean(B - A).

    Resamples query indices WITH replacement (paired: both arms move
    together), so the CI reflects the paired design; pairs with a length
    mismatch raise instead of silently truncating.
    """
    if len(a_correct) != len(b_correct):
        raise ValueError(
            f"paired lists required, got {len(a_correct)} vs {len(b_correct)}")
    n = len(a_correct)
    if n == 0:
        return {"estimate": 0.0, "lo": 0.0, "hi": 0.0, "n": 0,
                "n_resamples": n_resamples, "ci": ci}
    diffs = [float(b) - float(a) for a, b in zip(a_correct, b_correct)]
    rng = random.Random(seed)
    draws = []
    for _ in range(max(1, n_resamples)):
        draws.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    draws.sort()
    lo, hi = _percentile(draws, ci)
    return {"estimate": sum(diffs) / n, "lo": lo, "hi": hi, "n": n,
            "n_resamples": n_resamples, "ci": ci}


def bootstrap_saving_ci(
    per_query_strong: list[float],
    per_query_oracle: list[float],
    n_resamples: int = HEADROOM_RESAMPLES,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Bootstrap CI for the saving fraction 1 - sum(oracle)/sum(strong).

    Resamples queries (paired strong/oracle costs move together); resamples
    with zero strong-cost total contribute a 0.0 saving instead of NaN.
    """
    if len(per_query_strong) != len(per_query_oracle):
        raise ValueError("paired per-query cost lists required")
    n = len(per_query_strong)
    strong_total = sum(per_query_strong)
    oracle_total = sum(per_query_oracle)
    estimate = ((strong_total - oracle_total) / strong_total) if strong_total else 0.0
    if n == 0:
        return {"estimate": estimate, "lo": estimate, "hi": estimate, "n": 0,
                "n_resamples": n_resamples, "ci": ci}
    rng = random.Random(seed)
    draws = []
    for _ in range(max(1, n_resamples)):
        idx = [rng.randrange(n) for _ in range(n)]
        s = sum(per_query_strong[i] for i in idx)
        o = sum(per_query_oracle[i] for i in idx)
        draws.append(((s - o) / s) if s else 0.0)
    draws.sort()
    lo, hi = _percentile(draws, ci)
    return {"estimate": estimate, "lo": lo, "hi": hi, "n": n,
            "n_resamples": n_resamples, "ci": ci}
