#!/usr/bin/env python3
"""tiersweep-16: break-even analysis over progressively stronger cheap tiers.

Reads per-tier telemetry (one sweep DB + one repeats DB per served model,
named ``sweep-<tag>.db`` / ``repeats-<tag>.db`` in ``--db-dir``), rebuilds the
stable-oracle matrix per tier (3x majority; ties incorrect - same rule as
``scripts/stable_oracle.py``), and computes the headroom gate for each cheap
route (L0/L1/C0) against the C4 strong stub:

  coverage (routable fraction) + bootstrap CI, C4-vs-tier gap + McNemar,
  measured amortized GPU-seconds and USD per query, and the cheap/strong
  measured cost ratio.

It then writes the tier-vs-coverage-vs-cost table and the break-even
statement: at the measured cost ratio r, the no-loss saving at coverage f is
``f * (1 - r)``, so the plan's 0.10 exploitable-separation gate is the
break-even coverage X; requiring a saving of at least s raises it to
``X(s) = s / (1 - r)``.

Usage:
    python scripts/tier_sweep.py --db-dir results/tiersweep-16 \\
        --out-dir results/tiersweep-16
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from costsmart.eval import stability  # noqa: E402
from costsmart.eval.metrics import CORRECT_THRESHOLD, _attempt_cost  # noqa: E402
from costsmart.telemetry import attestation  # noqa: E402
from costsmart.telemetry.schema import N_REPEATS  # noqa: E402
from make_plots import ROUTABLE_GATE, compute_headroom  # noqa: E402

#: Cheap-tier routes swept per served model (oracle_sweep ROUTE_SPECS).
CHEAP_ROUTES = ("L0", "L1", "C0")
STRONG_ROUTE = "C4"


def load_rows(db_path: str, table: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        try:
            return [dict(r) for r in conn.execute(f"SELECT * FROM {table}")]
        except sqlite3.OperationalError:
            return []
    finally:
        conn.close()


def build_stable_matrix(base_rows: list[dict],
                        repeat_rows: list[dict]) -> tuple[list[dict], dict]:
    """Representative-row matrix for the stable recount + flip report.

    Mirrors scripts/stable_oracle.py: repeated pairs are graded by 3x
    majority (ties incorrect) and represented by a draw on the majority side;
    every other row (cloud/C0 stubs) passes through unchanged. Refuses a
    sweep/repeats mismatch so two different runs are never blended.
    """
    base_by_pair = {(r["query_id"], r["route_id"]): r for r in base_rows}
    repeats_by_pair: dict[tuple[str, str], list[dict]] = {}
    for r in repeat_rows:
        repeats_by_pair.setdefault((r["query_id"], r["route_id"]), []).append(r)
    orphans = sorted(set(repeats_by_pair) - set(base_by_pair))
    if orphans:
        sample = ", ".join(f"{q}/{r}" for q, r in orphans[:5])
        raise SystemExit(
            f"refusing mixed inputs: {len(orphans)} repeated pair(s) not in the "
            f"base sweep matrix ({sample}); sweep and repeats DBs do not "
            "describe the same run")
    stable_rows: list[dict] = []
    for (qid, route), reps in sorted(repeats_by_pair.items()):
        verdict = stability.stable_label_for_pair(reps, base_by_pair.get((qid, route)))
        rep = verdict["representative"]
        if rep is None:
            raise SystemExit(f"no representative row for {qid}/{route}")
        row = dict(rep)
        row.update({"stable_correct": verdict["stable_correct"],
                    "stable_n_correct": verdict["n_correct"],
                    "stable_n_total": verdict["n_total"],
                    "stable_complete": verdict["complete"],
                    "stable_tie": verdict["tie"]})
        stable_rows.append(row)
    repeated = set(repeats_by_pair)
    stable_rows += [r for (qid, route), r in base_by_pair.items()
                    if (qid, route) not in repeated]
    return stable_rows, {"n_repeated_pairs": len(repeats_by_pair),
                         "n_repeat_rows": len(repeat_rows)}


_MODEL_PARAMS = re.compile(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])")


def model_param_billions(model_version: str) -> float | None:
    """Parse parameter count (billions) out of a model id, e.g. 3B -> 3.0."""
    match = _MODEL_PARAMS.search(model_version)
    return float(match.group(1)) if match else None


def _mean(rows: list[dict], key: str) -> float:
    vals = [float(r.get(key) or 0.0) for r in rows]
    return sum(vals) / len(vals) if vals else 0.0


def analyse_tier(model_version: str, base_rows: list[dict],
                 stable_matrix: list[dict], repeat_rows: list[dict],
                 resamples: int, seed: int) -> dict:
    """Gate + cost payload for one served model across the cheap routes."""
    measured = [r for r in base_rows
                if r.get("generator_mode") == "measured"
                and r.get("model_version") == model_version
                and r.get("route_id") in CHEAP_ROUTES]
    strong_rows = [r for r in stable_matrix if r.get("route_id") == STRONG_ROUTE]
    strong_per_query = _mean(strong_rows, "cloud_spend_usd") + _mean(
        strong_rows, "amortized_usd")
    att = attestation.audit(
        measured + [r for r in repeat_rows
                    if r.get("model_version") == model_version
                    and r.get("route_id") in CHEAP_ROUTES])
    routes: dict[str, dict] = {}
    for route in CHEAP_ROUTES:
        rows = [r for r in measured if r.get("route_id") == route]
        if not rows:
            continue
        gate = compute_headroom(stable_matrix, cheap_route=route,
                               strong_route=STRONG_ROUTE,
                               n_resamples=resamples, seed=seed)
        cheap_per_query = _mean(rows, "amortized_usd")
        ratio = (cheap_per_query / strong_per_query) if strong_per_query else 0.0
        routes[route] = {
            "n_queries": gate["n_paired_queries"],
            "coverage": gate["routable_fraction"]["estimate"],
            "coverage_ci": [gate["routable_fraction"]["ci_lo"],
                            gate["routable_fraction"]["ci_hi"]],
            "gap_c4_minus_tier": gate["paired"]["accuracy_gap"],
            "gap_ci": [gate["paired"]["gap_ci_lo"], gate["paired"]["gap_ci_hi"]],
            "mcnemar_b": gate["paired"]["mcnemar_b"],
            "mcnemar_c": gate["paired"]["mcnemar_c"],
            "mcnemar_p": gate["paired"]["mcnemar_p"],
            "verdict": gate["gate"]["verdict"],
            "gpu_seconds_per_query": _mean(rows, "gpu_seconds"),
            "amortized_usd_per_query": cheap_per_query,
            "strong_usd_per_query": strong_per_query,
            "cost_ratio_cheap_over_strong": ratio,
            "saving_at_gate": ROUTABLE_GATE * (1.0 - ratio),
            "gate": gate,
        }
    best_route = None
    if routes:
        best_route = max(routes, key=lambda r: routes[r]["coverage"])
    return {
        "model_version": model_version,
        "params_billions": model_param_billions(model_version),
        "n_base_measured": len(measured),
        "n_repeat_measured": len(repeat_rows),
        "attestation": att,
        "routes": routes,
        "best_cheap_route": best_route,
        "best_coverage": routes[best_route]["coverage"] if best_route else 0.0,
    }


def _decision(point: float, ci_lo: float) -> str:
    """Project gate rule: point estimate decides GO (see make_plots).

    The CI lower bound is annotated so a marginal GO (point clears the gate
    but the lower bound does not) is never over-read as a comfortable win.
    """
    if point >= ROUTABLE_GATE:
        return ("GO" if ci_lo >= ROUTABLE_GATE
                else "GO (marginal: CI lower bound < gate)")
    return "NO-GO"


def break_even_statement(tiers: list[dict]) -> dict:
    """Which tier crosses the gate (per route), and the break-even rule.

    The project's gate is on the point estimate (``make_plots``), so that is
    what decides GO; the CI lower bound is reported alongside so a marginal
    GO is never over-read.
    """
    ordered = sorted(
        [t for t in tiers if t["params_billions"] is not None],
        key=lambda t: t["params_billions"])
    entries = []
    for tier in ordered:
        for route in CHEAP_ROUTES:
            if route not in tier["routes"]:
                continue
            e = tier["routes"][route]
            entries.append({
                "model_version": tier["model_version"],
                "params_billions": tier["params_billions"],
                "route": route,
                "coverage": e["coverage"],
                "coverage_ci": e["coverage_ci"],
                "cost_ratio": e["cost_ratio_cheap_over_strong"],
                "gpu_seconds_per_query": e["gpu_seconds_per_query"],
                "decision": _decision(e["coverage"], e["coverage_ci"][0]),
            })
    first_by_route: dict[str, dict | None] = {}
    for route in CHEAP_ROUTES:
        first_by_route[route] = next(
            (e for e in entries
             if e["route"] == route and e["decision"].startswith("GO")), None)
    first_l0 = first_by_route.get("L0")
    first_any = next((e for e in entries if e["decision"].startswith("GO")), None)
    if first_l0 is not None:
        primary = (f"The closed-book cheap route L0 first clears the gate at "
                   f"{first_l0['model_version']} "
                   f"(coverage {first_l0['coverage']:.3f} "
                   f"[{first_l0['coverage_ci'][0]:.3f}, "
                   f"{first_l0['coverage_ci'][1]:.3f}], cost ratio "
                   f"r={first_l0['cost_ratio']:.4f}).")
    elif first_any is not None:
        primary = (f"No closed-book L0 tier clears the gate up to the largest "
                   f"swept model; the first route/tier to clear it is "
                   f"{first_any['model_version']} / {first_any['route']} "
                   f"(coverage {first_any['coverage']:.3f}, cost ratio "
                   f"r={first_any['cost_ratio']:.4f}).")
    else:
        primary = "No swept cheap tier/route clears the gate."
    return {
        "gate_threshold": ROUTABLE_GATE,
        "rule": ("coverage (routable fraction) must clear the 0.10 gate; at "
                 "measured cost ratio r the no-loss saving at coverage f is "
                 "f*(1-r), so requiring saving >= s gives break-even coverage "
                 "X(s) = s/(1-r); at the gate the guaranteed saving is "
                 "0.10*(1-r)"),
        "entries": entries,
        "first_go_by_route": first_by_route,
        "first_go_l0": first_l0,
        "first_go_any": first_any,
        "first_go": first_l0 or first_any,
        "statement": (
            f"Routing pays when cheap-tier coverage exceeds the "
            f"{ROUTABLE_GATE:.2f} exploitable-separation gate. {primary}"),
    }


def render_tiers_svg(tiers: list[dict], break_even: dict) -> str:
    """Coverage bars per (tier, best route) with CI whiskers + gate line."""
    rows = [t for t in sorted(
        [t for t in tiers if t["params_billions"] is not None],
        key=lambda t: t["params_billions"])]
    width, row_h, top = 720, 46, 96
    height = top + row_h * len(rows) + 96
    x0, w = 250, 420

    def px(v: float) -> float:
        return x0 + max(0.0, min(1.0, v)) * w

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
        '<rect x="0" y="0" width="720" height="52" fill="#0b5394"/>',
        '<text x="20" y="33" font-size="19" font-weight="bold" fill="white" '
        'font-family="sans-serif">Cheap-tier break-even: coverage vs gate '
        '(real Tier-A)</text>',
        f'<text x="20" y="74" font-size="12" font-family="sans-serif">'
        f'Correctness = token_f1 &gt;= {CORRECT_THRESHOLD}; whiskers = 95% '
        'bootstrap CI; per best cheap route.</text>',
        f'<line x1="{px(ROUTABLE_GATE):.1f}" y1="{top - 16}" '
        f'x2="{px(ROUTABLE_GATE):.1f}" y2="{top + row_h * len(rows)}" '
        'stroke="#b42318" stroke-dasharray="5,4" stroke-width="2"/>',
        f'<text x="{px(ROUTABLE_GATE):.1f}" y="{top - 22}" font-size="12" '
        'fill="#b42318" font-family="sans-serif" text-anchor="middle">gate '
        f'{ROUTABLE_GATE:.2f}</text>',
    ]
    for i, tier in enumerate(rows):
        y = top + i * row_h
        route = tier["best_cheap_route"]
        entry = tier["routes"].get(route, {})
        cov = entry.get("coverage", 0.0)
        lo, hi = entry.get("coverage_ci", [0.0, 0.0])
        ratio = entry.get("cost_ratio_cheap_over_strong", 0.0)
        gpu = entry.get("gpu_seconds_per_query", 0.0)
        decision = "GO" if entry.get("verdict") == "GO" else "NO-GO"
        color = "#1a7f37" if decision == "GO" else "#8250df"
        parts += [
            f'<text x="20" y="{y + 24}" font-size="13" font-family="sans-serif">'
            f'{tier["model_version"].split("/")[-1]} ({route})</text>',
            f'<rect x="{x0}" y="{y + 8}" width="{w}" height="20" fill="#eee"/>',
            f'<rect x="{x0}" y="{y + 8}" width="{px(cov) - x0:.1f}" height="20" '
            f'fill="{color}"/>',
            f'<line x1="{px(lo):.1f}" y1="{y + 4}" x2="{px(lo):.1f}" '
            f'y2="{y + 32}" stroke="#111" stroke-width="2"/>',
            f'<line x1="{px(hi):.1f}" y1="{y + 4}" x2="{px(hi):.1f}" '
            f'y2="{y + 32}" stroke="#111" stroke-width="2"/>',
            f'<text x="{x0 + w + 10}" y="{y + 24}" font-size="12" '
            f'font-family="monospace">cov {cov:.3f} | {decision} | '
            f'r={ratio:.4f} | {gpu:.1f}s/q</text>',
        ]
    first = break_even.get("first_go")
    footer = (f"Break-even: first GO = {first['model_version']} / "
              f"{first['route']} (coverage {first['coverage']:.3f})"
              if first else "Break-even: no swept tier clears the gate")
    parts += [
        f'<text x="20" y="{height - 44}" font-size="12" font-family="sans-serif">'
        f'{footer}</text>',
        f'<text x="20" y="{height - 24}" font-size="11" font-family="sans-serif">'
        'Costs: cheap = measured amortized T4 GPU-seconds; strong = C4 cloud '
        'STUB estimate ($0 spent).</text>',
        '</svg>',
    ]
    return "\n".join(parts)


def render_table_md(tiers: list[dict], break_even: dict) -> str:
    lines = [
        "| model | route | coverage (95% CI) | C4-gap | McNemar p | "
        "GPU-s/query | cost ratio r | saving @ gate | verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for tier in sorted([t for t in tiers if t["params_billions"] is not None],
                       key=lambda t: t["params_billions"]):
        for route in CHEAP_ROUTES:
            if route not in tier["routes"]:
                continue
            e = tier["routes"][route]
            lines.append(
                f"| {tier['model_version'].split('/')[-1]} | {route} | "
                f"{e['coverage']:.3f} [{e['coverage_ci'][0]:.3f}, "
                f"{e['coverage_ci'][1]:.3f}] | {e['gap_c4_minus_tier']:.3f} | "
                f"{e['mcnemar_p']:.2e} | {e['gpu_seconds_per_query']:.2f} | "
                f"{e['cost_ratio_cheap_over_strong']:.4f} | "
                f"{e['saving_at_gate']:.4f} | {e['verdict']} |")
    lines.append("")
    lines.append(break_even["statement"])
    return "\n".join(lines)


def discover_runs(db_dir: Path) -> list[tuple[str, Path, Path]]:
    runs = []
    for sweep in sorted(db_dir.glob("sweep-*.db")):
        tag = sweep.stem[len("sweep-"):]
        repeats = db_dir / f"repeats-{tag}.db"
        if not repeats.exists():
            print(f"warning: no repeats DB for {sweep.name}; skipping",
                  file=sys.stderr)
            continue
        runs.append((tag, sweep, repeats))
    return runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tier break-even analysis")
    parser.add_argument("--db-dir", required=True,
                        help="directory with sweep-<tag>.db / repeats-<tag>.db")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    db_dir = Path(args.db_dir)
    runs = discover_runs(db_dir)
    if not runs:
        raise SystemExit(f"no sweep-<tag>.db / repeats-<tag>.db pairs in {db_dir}")

    tiers: list[dict] = []
    for tag, sweep_db, repeats_db in runs:
        base_rows = load_rows(str(sweep_db), "attempts")
        repeat_rows = load_rows(str(repeats_db), "repeat_attempts")
        stable_matrix, meta = build_stable_matrix(base_rows, repeat_rows)
        models = sorted({r["model_version"] for r in base_rows
                         if r.get("generator_mode") == "measured"
                         and r.get("route_id") in CHEAP_ROUTES})
        if not models:
            print(f"warning: {tag} has no measured cheap-tier rows; skipping",
                  file=sys.stderr)
            continue
        for model in models:
            tier = analyse_tier(model, base_rows, stable_matrix, repeat_rows,
                                args.resamples, args.seed)
            tier["tag"] = tag
            tier.update(meta)
            tiers.append(tier)
        print(f"analysed {tag}: models={[m.split('/')[-1] for m in models]} "
              f"({meta['n_repeated_pairs']} repeated pairs)", file=sys.stderr)

    break_even = break_even_statement(tiers)
    payload = {"tiers": tiers, "break_even": break_even}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tiers.json").write_text(json.dumps(payload, indent=2))
    (out_dir / "TIERS.md").write_text(render_table_md(tiers, break_even) + "\n")
    (out_dir / "tiers.svg").write_text(render_tiers_svg(tiers, break_even))
    print(json.dumps({
        "n_tiers": len(tiers),
        "first_go": break_even["first_go"],
        "statement": break_even["statement"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
