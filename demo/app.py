"""Demo app skeleton: query box + live route trace + cost counter vs always-strong.

Stdlib-first: ``python demo/app.py "query"`` runs a stubbed loop and prints
the route decision, trace, and running cost vs an always-strong baseline.
With streamlit installed, ``python demo/app.py --serve`` launches the
interactive query-box UI instead.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from costsmart.agent.loop import AgentLoop  # noqa: E402
from costsmart.routing.heuristic import COST, CLOUD, route as heuristic_route  # noqa: E402
from costsmart.verify import SignalResult  # noqa: E402

STRONG_COST = COST[CLOUD]


def run_query(query: str, budget: float = 50.0) -> dict:
    """Route *query* through a stubbed agent loop; return a JSON-able report."""
    decision = heuristic_route(query)
    loop = AgentLoop(budget=budget)

    def answer_fn(q: str, evidence: list[str]) -> str:
        if evidence:
            return f"stub-answer (local, {len(evidence)} docs): {q[:60]}"
        return f"stub-answer (local, no retrieval): {q[:60]}"

    def verify_fn(answer: str, evidence: list[str]) -> SignalResult:
        # Stub: retrieval-backed answers verify higher.
        score = 0.85 if evidence else 0.4
        return SignalResult(signal="stub", score=score, cost=0.0)

    def retrieve_fn(q: str) -> list[str]:
        return [f"stub-doc-{i} for {q[:40]}" for i in (1, 2)]

    result = loop.run(query, answer_fn, verify_fn, retrieve_fn)
    saved = STRONG_COST - decision.estimated_cost
    return {
        "query": query,
        "route": decision.route,
        "route_confidence": decision.confidence,
        "route_reasons": decision.reasons,
        "estimated_route_cost": decision.estimated_cost,
        "always_strong_cost": STRONG_COST,
        "saved_vs_always_strong": round(saved, 2),
        "loop": result.to_dict(),
    }


def print_report(report: dict) -> None:
    print(f"query:  {report['query']}")
    print(f"route:  {report['route']} "
          f"(conf={report['route_confidence']}, "
          f"cost={report['estimated_route_cost']})")
    print(f"vs always-strong: cost {report['always_strong_cost']} -> "
          f"saved {report['saved_vs_always_strong']}")
    loop = report["loop"]
    print(f"loop:   outcome={loop['outcome']} "
          f"iterations={loop['iterations_used']} "
          f"total_cost={loop['total_cost']} "
          f"low_confidence={loop['low_confidence']}")
    print("trace:")
    for ev in loop["trace"]:
        print(f"  [it={ev['iteration']}] {ev['action']}: {ev['detail']} "
              f"(cost={ev['cost']})")


def serve() -> None:
    """Streamlit UI (requires streamlit installed)."""
    import streamlit as st

    st.title("CostSmart RAG demo")
    st.caption("Query box + live route trace + cost counter vs always-strong")
    query = st.text_input("Ask a question", "Who wrote the song Hallelujah?")
    budget = st.slider("Agent budget", 1.0, 100.0, 50.0)
    if st.button("Run") and query:
        report = run_query(query, budget=budget)
        st.subheader(f"Route: {report['route']}")
        st.write({"confidence": report["route_confidence"],
                  "reasons": report["route_reasons"]})
        loop = report["loop"]
        st.subheader(f"Outcome: {loop['outcome']}")
        st.write({"iterations": loop["iterations_used"],
                  "total_cost": loop["total_cost"],
                  "low_confidence": loop["low_confidence"]})
        st.subheader("Live route trace")
        st.table(loop["trace"])
        st.subheader("Cost counter")
        st.write({"routed_cost": report["estimated_route_cost"],
                  "always_strong_cost": report["always_strong_cost"],
                  "saved": report["saved_vs_always_strong"]})
        st.download_button("Download trace JSON",
                           json.dumps(report, indent=2), "trace.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CostSmart RAG demo skeleton")
    parser.add_argument("query", nargs="?", default="Who wrote Hallelujah?",
                        help="query to route")
    parser.add_argument("--budget", type=float, default=50.0)
    parser.add_argument("--serve", action="store_true",
                        help="launch streamlit UI instead of CLI")
    parser.add_argument("--json", action="store_true", help="emit raw JSON")
    args = parser.parse_args(argv)
    if args.serve:
        serve()
        return 0
    report = run_query(args.query, budget=args.budget)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
