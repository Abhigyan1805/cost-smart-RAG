"""Trace viewer skeleton: render a saved loop trace as timeline + cost summary.

Usage: ``python demo/trace_viewer.py trace.json`` (prints to stdout), or
import :func:`summarize` / :func:`format_timeline` for embedding in the app.
"""

from __future__ import annotations

import json
import sys


def load_trace(path: str) -> dict:
    with open(path) as fh:
        data = json.load(fh)
    return data.get("loop", data)  # accept full report or bare loop dict


def summarize(loop: dict) -> dict:
    """Cost + action summary of a loop trace dict."""
    trace = loop.get("trace", [])
    by_action: dict[str, int] = {}
    for ev in trace:
        by_action[ev.get("action", "?")] = by_action.get(ev.get("action", "?"), 0) + 1
    return {
        "trace_id": loop.get("trace_id"),
        "query": loop.get("query"),
        "outcome": loop.get("outcome"),
        "iterations_used": loop.get("iterations_used"),
        "total_cost": loop.get("total_cost"),
        "low_confidence": loop.get("low_confidence"),
        "dedup_hits": loop.get("dedup_hits"),
        "actions": by_action,
    }


def format_timeline(loop: dict) -> str:
    """Human-readable timeline of trace events."""
    lines = [f"trace {loop.get('trace_id')} -- {loop.get('query')}"]
    for ev in loop.get("trace", []):
        lines.append(f"  [it={ev.get('iteration')}] {ev.get('action')}: "
                     f"{ev.get('detail')} (cost={ev.get('cost')})")
    for dec in loop.get("decisions", []):
        lines.append(f"  decision it={dec.get('iteration')}: "
                     f"{dec.get('action')} -- {dec.get('rationale')}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python demo/trace_viewer.py <trace.json>", file=sys.stderr)
        return 2
    loop = load_trace(argv[1])
    print(format_timeline(loop))
    print("--- summary ---")
    print(json.dumps(summarize(loop), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
