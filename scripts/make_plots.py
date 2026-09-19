#!/usr/bin/env python3
"""Report figures for costsmart-rag (stdlib only: hand-authored SVG).

Subcommand ``headroom`` (costheadroom-09): reads the pilot telemetry DB,
builds the cheapest-local vs strongest-cloud 2x2 headroom gate
(contingency + routable fraction + max saving at no quality loss, all with
bootstrap CIs and the McNemar paired test), writes ``headroom.json`` and
renders ``headroom.svg``.

Usage:
    python scripts/make_plots.py headroom --db results/costpilot-06/pilot.db \\
        --out-dir results/costheadroom-09
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from costsmart.eval.metrics import (  # noqa: E402
    CHEAP_ROUTE,
    STRONG_ROUTE,
    build_contingency_table,
    is_correct,
    max_savings_at_no_quality_loss,
    routable_fraction,
)
from costsmart.eval.stats import (  # noqa: E402
    HEADROOM_RESAMPLES,
    bootstrap_paired_diff_ci,
    bootstrap_proportion_ci,
    bootstrap_saving_ci,
    mcnemar,
)

#: Go/no-go gate from the plan: routable fraction below this means the
#: tiers sit too close together (no exploitable separation) -> NO-GO.
ROUTABLE_GATE = 0.10

#: Bootstrap budget for the gate (10k resamples per the spec).
N_RESAMPLES = HEADROOM_RESAMPLES


def load_attempts(db_path: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM attempts")]
    finally:
        conn.close()


def compute_headroom(attempts: list[dict], cheap_route: str = CHEAP_ROUTE,
                     strong_route: str = STRONG_ROUTE,
                     n_resamples: int = N_RESAMPLES,
                     seed: int = 0) -> dict:
    """Full headroom gate payload (JSON-serializable)."""
    table = build_contingency_table(attempts, cheap_route, strong_route)
    rf = routable_fraction(table)
    rf_ci = bootstrap_proportion_ci(rf["k"], rf["n"], n_resamples, seed=seed)
    savings = max_savings_at_no_quality_loss(attempts, strong_route)
    saving_ci = bootstrap_saving_ci(savings["per_query_strong"],
                                    savings["per_query_oracle"],
                                    n_resamples, seed=seed + 1)
    # Paired correctness vectors (cheap, strong) over the paired queries.
    by_query: dict[str, dict] = {}
    for attempt in attempts:
        if attempt.get("route_id") in (cheap_route, strong_route):
            by_query.setdefault(attempt.get("query_id", ""), {})[
                attempt["route_id"]] = attempt
    paired = [(r[cheap_route], r[strong_route]) for r in by_query.values()
              if cheap_route in r and strong_route in r]
    cheap_ok = [is_correct(r[0]) for r in paired]
    strong_ok = [is_correct(r[1]) for r in paired]
    b = sum(1 for c, s in zip(cheap_ok, strong_ok) if not c and s)
    c = sum(1 for c_, s in zip(cheap_ok, strong_ok) if c_ and not s)
    mc = mcnemar(b, c)
    diff_ci = bootstrap_paired_diff_ci(cheap_ok, strong_ok, n_resamples,
                                       seed=seed + 2)
    cheap_acc = sum(cheap_ok) / len(cheap_ok) if cheap_ok else 0.0
    strong_acc = sum(strong_ok) / len(strong_ok) if strong_ok else 0.0
    verdict = ("GO" if rf_ci["estimate"] >= ROUTABLE_GATE else "NO-GO")
    return {
        "cheap_route": cheap_route,
        "strong_route": strong_route,
        "n_paired_queries": table["n"],
        "n_skipped": table["skipped"],
        "contingency": {k: table[k] for k in (
            "both_correct", "cheap_only", "strong_only", "neither",
            "both_correct_rate", "cheap_only_rate",
            "strong_only_rate", "neither_rate")},
        "routable_fraction": {**rf, "ci_lo": rf_ci["lo"],
                               "ci_hi": rf_ci["hi"], "ci": rf_ci["ci"],
                               "n_resamples": n_resamples},
        "savings": {
            "strong_cost": savings["strong_cost"],
            "oracle_cost": savings["oracle_cost"],
            "saving_fraction": savings["saving_fraction"],
            "ci_lo": saving_ci["lo"], "ci_hi": saving_ci["hi"],
            "n_routed_off_strong": savings["n_routed_off_strong"],
            "n_no_correct_route": savings["n_no_correct_route"],
            "n_quality_loss_queries": savings["n_quality_loss_queries"],
            "n_resamples": n_resamples,
        },
        "paired": {
            "cheap_accuracy": cheap_acc,
            "strong_accuracy": strong_acc,
            "accuracy_gap": strong_acc - cheap_acc,
            "gap_ci_lo": diff_ci["lo"], "gap_ci_hi": diff_ci["hi"],
            "mcnemar_b": b, "mcnemar_c": c,
            "mcnemar_statistic": mc["statistic"],
            "mcnemar_p": mc["p_value"],
            "mcnemar_n_discordant": mc["n_discordant"],
        },
        "gate": {"threshold": ROUTABLE_GATE, "verdict": verdict},
        "n_resamples": n_resamples,
        "seed": seed,
    }


def _svg_bar(value: float, lo: float, hi: float, x: int, y: int, w: int,
             color: str, label: str) -> str:
    pos = lambda v: x + max(0.0, min(1.0, v)) * w
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="18" fill="#eee"/>',
        f'<rect x="{x}" y="{y}" width="{pos(value) - x:.1f}" height="18" fill="{color}"/>',
        f'<line x1="{pos(lo):.1f}" y1="{y - 3}" x2="{pos(lo):.1f}" y2="{y + 21}" stroke="#111" stroke-width="2"/>',
        f'<line x1="{pos(hi):.1f}" y1="{y - 3}" x2="{pos(hi):.1f}" y2="{y + 21}" stroke="#111" stroke-width="2"/>',
        f'<line x1="{pos(lo):.1f}" y1="{y + 9}" x2="{pos(hi):.1f}" y2="{y + 9}" stroke="#111" stroke-width="2"/>',
        f'<text x="{x}" y="{y - 6}" font-size="12" font-family="sans-serif">{label}</text>',
    ]
    return "\n".join(parts)


def render_headroom_svg(data: dict) -> str:
    """2x2 table + routable/saving bars with CI whiskers + verdict banner."""
    cont = data["contingency"]
    rf = data["routable_fraction"]
    sv = data["savings"]
    pr = data["paired"]
    cheap, strong = data["cheap_route"], data["strong_route"]
    verdict = data["gate"]["verdict"]
    banner = "#1a7f37" if verdict == "GO" else "#b42318"
    cells = [(f"{strong} correct", f"{strong} wrong"),
             (f"{cheap} correct", cont["both_correct"], cont["cheap_only"]),
             (f"{cheap} wrong", cont["strong_only"], cont["neither"])]
    n = data["n_paired_queries"]
    ty, tx0, tx1, tx2 = 120, 30, 250, 430
    rows = [(ty, "corner", [tx1, tx2], ["strong correct", "strong wrong"]),
            (ty + 44, f"{cheap} correct", [tx1, tx2],
             [cont["both_correct"], cont["cheap_only"]]),
            (ty + 88, f"{cheap} wrong", [tx1, tx2],
             [cont["strong_only"], cont["neither"]])]
    grid = [f'<text x="{tx0}" y="96" font-size="13" font-family="sans-serif">n={n} paired queries</text>']
    for y, label, xs, vals in rows:
        grid.append(f'<text x="{tx0}" y="{y + 28}" font-size="13" font-family="sans-serif">{label}</text>')
        for xx, vv in zip(xs, vals):
            grid.append(f'<rect x="{xx}" y="{y}" width="150" height="40" fill="#f6f8fa" stroke="#333"/>')
            grid.append(f'<text x="{xx + 75}" y="{y + 26}" font-size="16" text-anchor="middle" font-family="sans-serif">{vv}</text>')
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="640" height="520">
<rect x="0" y="0" width="640" height="520" fill="white"/>
<rect x="0" y="0" width="640" height="52" fill="{banner}"/>
<text x="20" y="33" font-size="20" font-weight="bold" fill="white" font-family="sans-serif">Headroom gate: {cheap} vs {strong} - {verdict} (pilot, n={n})</text>
<text x="30" y="80" font-size="13" font-family="sans-serif">Correctness = token_f1 &gt;= 0.5. Whiskers = 95% bootstrap CI ({data["n_resamples"]} resamples).</text>
{"".join(grid)}
{_svg_bar(rf["estimate"], rf["ci_lo"], rf["ci_hi"], 30, 280, 560, "#2f6fed", f'Routable fraction: {rf["estimate"]:.2f} [{rf["ci_lo"]:.2f}, {rf["ci_hi"]:.2f}] (gate &gt;= {data["gate"]["threshold"]:.2f})')}
{_svg_bar(sv["saving_fraction"], sv["ci_lo"], sv["ci_hi"], 30, 340, 560, "#8250df", f'Max saving at no quality loss: {sv["saving_fraction"]:.1%} [{sv["ci_lo"]:.1%}, {sv["ci_hi"]:.1%}]')}
<text x="30" y="410" font-size="12" font-family="sans-serif">Paired gap (strong − cheap): {pr["accuracy_gap"]:.2f} [{pr["gap_ci_lo"]:.2f}, {pr["gap_ci_hi"]:.2f}]; McNemar b={pr["mcnemar_b"]} c={pr["mcnemar_c"]} p={pr["mcnemar_p"]:.4f}.</text>
<text x="30" y="432" font-size="12" font-family="sans-serif">Oracle cost ${sv["oracle_cost"]:.6f} vs all-{strong} ${sv["strong_cost"]:.6f}; {sv["n_routed_off_strong"]}/{n} routed off {strong}; quality-loss queries: {sv["n_quality_loss_queries"]}.</text>
<text x="30" y="470" font-size="12" font-family="sans-serif">SMALL-n CAVEAT: n={n} pilot queries - CIs are wide; do not over-claim. Re-run on the full sweep matrix when it merges.</text>
<text x="30" y="492" font-size="12" font-family="sans-serif">Stub estimates only (no API spend); local tiers preliminary until Colab attaches.</text>
</svg>"""
    return svg


def cmd_headroom(args: argparse.Namespace) -> int:
    if args.attempts_json:
        # Stable-oracle recount (costfinal-10): compute the gate over a
        # JSON attempts matrix (e.g. results/costfinal-10/stable_attempts.json)
        # instead of a telemetry DB. Reproducible via:
        #   python scripts/make_plots.py headroom --attempts-json \\
        #       results/costfinal-10/stable_attempts.json --out-dir <dir>
        attempts = json.loads(Path(args.attempts_json).read_text())
    else:
        attempts = load_attempts(args.db)
    if not attempts:
        raise SystemExit(f"no attempts in {args.db or args.attempts_json}")
    data = compute_headroom(attempts, args.cheap, args.strong,
                            args.resamples, args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "headroom.json").write_text(json.dumps(data, indent=2))
    (out_dir / "headroom.svg").write_text(render_headroom_svg(data))
    print(json.dumps({"verdict": data["gate"]["verdict"],
                      "routable": round(data["routable_fraction"]["estimate"], 4),
                      "saving": round(data["savings"]["saving_fraction"], 4),
                      "n": data["n_paired_queries"]}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report figures (SVG, stdlib)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    head = sub.add_parser("headroom", help="headroom 2x2 gate figure")
    head.add_argument("--db", default=None,
                      help="telemetry SQLite DB (ignored with --attempts-json)")
    head.add_argument("--attempts-json", default=None,
                      help="JSON attempts matrix (stable-oracle recount path)")
    head.add_argument("--out-dir", required=True)
    head.add_argument("--cheap", default=CHEAP_ROUTE)
    head.add_argument("--strong", default=STRONG_ROUTE)
    head.add_argument("--resamples", type=int, default=N_RESAMPLES)
    head.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.cmd == "headroom":
        return cmd_headroom(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
