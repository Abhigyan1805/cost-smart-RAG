"""Typer entrypoint wiring each subcommand to its owning slice.

Runs on typer when installed (``make install``); falls back to an argparse
frontend with identical subcommands/options in stdlib-only environments so
the Makefile targets keep working without extra installs.
"""

from __future__ import annotations

from typing import Optional

try:
    import typer

    _HAS_TYPER = True
except ImportError:  # stdlib-only env (no pip): use the argparse frontend below
    typer = None  # type: ignore[assignment]
    _HAS_TYPER = False


def cmd_index(out: str = "data/index/pilot_index.json", source: str = "synthetic") -> int:
    """Build the retrieval index (corpus + retrieval slices)."""
    from costsmart.corpus.build_index import main as build_main

    return build_main(["--out", out, "--source", source])


def cmd_sweep(
    limit: int = 20,
    no_limit: bool = False,
    db: str = "telemetry.db",
    config: Optional[str] = None,
    export_csv: Optional[str] = None,
    export_parquet: Optional[str] = None,
    index: str = "data/index/pilot_index.json",
    no_retrieval: bool = False,
    live_local: bool = False,
) -> int:
    """Run the route sweep (exhaustive oracle) into the telemetry table."""
    from costsmart.eval.oracle_sweep import main as sweep_main

    argv = ["--limit", str(limit), "--db", db, "--index", index]
    if no_limit:
        argv.append("--no-limit")
    if config:
        argv += ["--config", config]
    if export_csv:
        argv += ["--export-csv", export_csv]
    if export_parquet:
        argv += ["--export-parquet", export_parquet]
    if no_retrieval:
        argv.append("--no-retrieval")
    if live_local:
        argv.append("--live-local")
    return sweep_main(argv)


def cmd_train_router(
    db: str = "telemetry.db", out: str = "router-v1.json", mode: str = "pre"
) -> int:
    """Train the router on sweep telemetry (routing slice)."""
    from costsmart.routing.train import main as train_main

    return train_main(["--db", db, "--out", out, "--mode", mode])


def cmd_eval(db: str = "telemetry.db", router: Optional[str] = None) -> int:
    """Evaluate router vs exhaustive oracle (eval slice)."""
    from costsmart.eval.evaluate import main as eval_main

    argv = ["--db", db]
    if router:
        argv += ["--router", router]
    return eval_main(argv)


def cmd_report(db: str = "telemetry.db", router: Optional[str] = None) -> int:
    """Print the pilot summary (eval slice aggregation; REPORT.md stays manual)."""
    import json

    from costsmart.eval.evaluate import evaluate_db

    print(json.dumps(evaluate_db(db, router), indent=2))
    return 0


def _argparse_main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="costsmart")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("index")
    p.add_argument("--out", default="data/index/pilot_index.json")
    p.add_argument("--source", default="synthetic")

    p = sub.add_parser("sweep")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--no-limit", action="store_true")
    p.add_argument("--db", default="telemetry.db")
    p.add_argument("--config", default=None)
    p.add_argument("--export-csv", default=None)
    p.add_argument("--export-parquet", default=None)
    p.add_argument("--index", default="data/index/pilot_index.json")
    p.add_argument("--no-retrieval", action="store_true")
    p.add_argument("--live-local", action="store_true")

    p = sub.add_parser("train-router")
    p.add_argument("--db", default="telemetry.db")
    p.add_argument("--out", default="router-v1.json")
    p.add_argument("--mode", default="pre")

    p = sub.add_parser("eval")
    p.add_argument("--db", default="telemetry.db")
    p.add_argument("--router", default=None)

    p = sub.add_parser("report")
    p.add_argument("--db", default="telemetry.db")
    p.add_argument("--router", default=None)

    args = parser.parse_args(argv)
    if args.cmd == "index":
        return cmd_index(args.out, args.source)
    if args.cmd == "sweep":
        return cmd_sweep(
            args.limit, args.no_limit, args.db, args.config,
            args.export_csv, args.export_parquet, args.index, args.no_retrieval,
            args.live_local,
        )
    if args.cmd == "train-router":
        return cmd_train_router(args.db, args.out, args.mode)
    if args.cmd == "eval":
        return cmd_eval(args.db, args.router)
    if args.cmd == "report":
        return cmd_report(args.db, args.router)
    parser.error(f"unknown command {args.cmd}")
    return 2


if _HAS_TYPER:
    app = typer.Typer(help="costsmart-rag: routed RAG over the joint cost-quality space.")

    @app.command()
    def index(
        out: str = typer.Option("data/index/pilot_index.json"),
        source: str = typer.Option("synthetic"),
    ) -> None:
        """Build the retrieval index (corpus + retrieval slices)."""
        raise SystemExit(cmd_index(out, source))

    @app.command()
    def sweep(
        limit: int = typer.Option(20),
        no_limit: bool = typer.Option(False),
        db: str = typer.Option("telemetry.db"),
        config: Optional[str] = typer.Option(None),
        export_csv: Optional[str] = typer.Option(None),
        export_parquet: Optional[str] = typer.Option(None),
        index: str = typer.Option("data/index/pilot_index.json"),
        no_retrieval: bool = typer.Option(False),
        live_local: bool = typer.Option(False),
    ) -> None:
        """Run the route sweep (exhaustive oracle) into the telemetry table."""
        raise SystemExit(
            cmd_sweep(limit, no_limit, db, config, export_csv, export_parquet,
                      index, no_retrieval, live_local)
        )

    @app.command("train-router")
    def train_router(
        db: str = typer.Option("telemetry.db"),
        out: str = typer.Option("router-v1.json"),
        mode: str = typer.Option("pre"),
    ) -> None:
        """Train the router on sweep telemetry (routing slice)."""
        raise SystemExit(cmd_train_router(db, out, mode))

    @app.command()
    def eval(
        db: str = typer.Option("telemetry.db"),
        router: Optional[str] = typer.Option(None),
    ) -> None:
        """Evaluate router vs exhaustive oracle (eval slice)."""
        raise SystemExit(cmd_eval(db, router))

    @app.command()
    def report(
        db: str = typer.Option("telemetry.db"),
        router: Optional[str] = typer.Option(None),
    ) -> None:
        """Print the pilot summary (eval slice aggregation; REPORT.md stays manual)."""
        raise SystemExit(cmd_report(db, router))
else:

    def main(argv: list[str] | None = None) -> int:  # noqa: D103
        return _argparse_main(argv)

    class _App:  # minimal stand-in so `python -m costsmart.cli <cmd>` works
        def __call__(self, argv: list[str] | None = None) -> None:
            raise SystemExit(_argparse_main(argv))

    app = _App()  # type: ignore[assignment]


if __name__ == "__main__":
    import sys

    if _HAS_TYPER:
        app()
    else:
        raise SystemExit(_argparse_main(sys.argv[1:]))
