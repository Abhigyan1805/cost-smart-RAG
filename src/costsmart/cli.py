"""Typer entrypoint. Stub subcommands - real logic lands in sibling slices."""

from __future__ import annotations

import typer

app = typer.Typer(help="costsmart-rag: routed RAG over the joint cost-quality space.")


@app.command()
def index() -> None:
    """Build the retrieval index. (Stub: implemented by the retrieval slice.)"""
    typer.echo("index: not implemented yet (retrieval slice owns this).")


@app.command()
def sweep() -> None:
    """Run the route sweep (exhaustive oracle). (Stub.)"""
    typer.echo("sweep: not implemented yet.")


@app.command("train-router")
def train_router() -> None:
    """Train the router on sweep telemetry. (Stub: routing slice owns this.)"""
    typer.echo("train-router: not implemented yet.")


@app.command()
def eval() -> None:
    """Evaluate router vs exhaustive oracle. (Stub: eval slice owns this.)"""
    typer.echo("eval: not implemented yet.")


@app.command()
def report() -> None:
    """Render REPORT.md figures. (Stub.)"""
    typer.echo("report: not implemented yet.")


if __name__ == "__main__":
    app()
