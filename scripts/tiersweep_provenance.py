#!/usr/bin/env python3
"""tiersweep-16 provenance: hash-pin the Kaggle run + per-model attestation.

Writes ``results/tiersweep-16/kaggle_provenance.json`` and
``KAGGLE_PROVENANCE.md`` from the ingested DBs: resolved model revision +
weight-sha256 (server-attested, read from the rows themselves), licences read
from the HuggingFace model cards, artifact sha256s, and the kernel log hash.
Re-run after any re-ingest so the recorded hashes always match the committed
DBs (the same contract as results/realdata-15/kaggle_provenance.json).

Usage:
    python scripts/tiersweep_provenance.py --dir results/tiersweep-16 \\
        --kernel-log /tmp/kout/tiersweep16-local-sweep.log
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

#: Licence facts read from each model's HuggingFace card (cardData.license),
#: with the resolved repo sha and read date. Recorded, never inferred.
MODEL_CARDS = {
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "license": "apache-2.0",
        "card_sha": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    },
    "Qwen/Qwen2.5-3B-Instruct": {
        "license": "other",
        "license_name": "qwen-research",
        "license_link": "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE",
        "card_sha": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "license": "apache-2.0",
        "card_sha": "a09a35458c702b33eeacc393d103063234e8bc28",
    },
}

ARTIFACTS = (
    "sweep-3b.db", "repeats-3b.db", "sweep-3b.csv",
    "repeats-3b_summary.json", "sweep-7b.db", "repeats-7b.db",
    "sweep-7b.csv", "repeats-7b_summary.json",
    "3b-server.log", "7b-server.log", "tiers.json", "TIERS.md", "tiers.svg",
    "attestation_audit.json", "MODEL_MANIFEST.md",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rows(db: Path, table: str) -> list[dict]:
    if not db.exists():
        return []
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        try:
            return [dict(r) for r in conn.execute(f"SELECT * FROM {table}")]
        except sqlite3.OperationalError:
            return []
    finally:
        conn.close()


def collect_models(directory: Path) -> dict:
    models: dict[str, dict] = {}
    for tag in ("3b", "7b"):
        for table, db_name in (("attempts", f"sweep-{tag}.db"),
                               ("repeat_attempts", f"repeats-{tag}.db")):
            for row in _load_rows(directory / db_name, table):
                if row.get("generator_mode") != "measured":
                    continue
                model = row.get("model_version") or ""
                entry = models.setdefault(model, {
                    "model_version": model,
                    "model_revision": row.get("model_revision") or "",
                    "weights_sha256": row.get("weights_sha256") or "",
                    "attestation": row.get("attestation") or "",
                    "measured_rows": 0,
                    "tags": set(),
                })
                entry["measured_rows"] += 1
                entry["tags"].add(tag)
    for entry in models.values():
        entry["tags"] = sorted(entry["tags"])
        entry.update(MODEL_CARDS.get(entry["model_version"], {}))
    return models


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="tiersweep-16 provenance")
    parser.add_argument("--dir", default="results/tiersweep-16")
    parser.add_argument("--kernel-log", default=None,
                        help="downloaded Kaggle kernel log (not committed)")
    parser.add_argument("--download-command", default=(
        "kaggle kernels output abhigyan1818/tiersweep16-local-sweep -p /tmp/kout"))
    args = parser.parse_args(argv)

    directory = Path(args.dir)
    artifacts = {}
    for name in ARTIFACTS:
        path = directory / name
        if path.exists():
            artifacts[name] = {"sha256": sha256_file(path),
                               "bytes": path.stat().st_size}
    models = collect_models(directory)
    logs = {}
    for name in ("3b-server.log", "7b-server.log"):
        if (directory / name).exists():
            logs[name] = {"sha256": sha256_file(directory / name)}
    if args.kernel_log and Path(args.kernel_log).exists():
        logs[Path(args.kernel_log).name] = {
            "sha256": sha256_file(Path(args.kernel_log)),
            "note": "Kaggle kernel stdout/stderr; not committed.",
        }

    provenance = {
        "description": (
            "Provenance for the tiersweep-16 Kaggle run: cheap-tier routes "
            "L0/L1/C0 on Qwen2.5-3B/7B over the committed real Tier-A corpus. "
            "The 1.5B tier is the committed realdata-15 run and is not "
            "re-measured; its rows predate server attestation and are flagged "
            "'unattested' by scripts/attestation_audit.py."),
        "kernel": {
            "id": "abhigyan1818/tiersweep16-local-sweep",
            "metadata": "kernels/tiersweep-16-local-sweep/kernel-metadata.json",
            "code": "kernels/tiersweep-16-local-sweep/kernel.py",
            "branch": "fm/costsmart-tiersweep-16",
            "git_sha": next(iter(
                {r.get("git_sha") for r in _load_rows(
                    directory / "sweep-3b.db", "attempts") if r.get("git_sha")}
            ), "unknown"),
            "models": ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct"],
        },
        "download": {"command": args.download_command,
                     "status": "complete"},
        "logs": logs,
        "models": models,
        "reused_1p5b": {
            "model_version": "Qwen/Qwen2.5-1.5B-Instruct",
            "sweep_db": "results/realdata-15/sweep.db",
            "repeats_db": "results/realdata-15/repeats.db",
            "note": ("Measured by the realdata-15 run; reused as the 1.5B tier "
                     "by cache key, not re-measured. No server revision was "
                     "recorded then, so those rows are flagged unattested."),
        },
        "artifacts": artifacts,
    }
    out = directory / "kaggle_provenance.json"
    out.write_text(json.dumps(provenance, indent=2, sort_keys=False))
    print(f"wrote {out}")

    lines = [
        "# tiersweep-16 Kaggle provenance",
        "",
        provenance["description"],
        "",
        f"- Kernel: `{provenance['kernel']['id']}` "
        f"(branch `{provenance['kernel']['branch']}`, "
        f"git `{provenance['kernel']['git_sha']}`).",
        f"- Download: `{provenance['download']['command']}`.",
        "",
        "## Server-attested models",
        "",
        "| model | resolved revision | weights sha256 | licence | measured rows |",
        "|---|---|---|---|---|",
    ]
    for model, entry in sorted(models.items()):
        lines.append(
            f"| {model} | `{entry.get('model_revision') or 'n/a'}` | "
            f"`{(entry.get('weights_sha256') or 'n/a')[:16]}...` | "
            f"{entry.get('license', 'n/a')}"
            f"{' (' + entry['license_name'] + ')' if entry.get('license_name') else ''} | "
            f"{entry.get('measured_rows')} |")
    lines += [
        "",
        "The 1.5B tier reuses `results/realdata-15/{sweep,repeats}.db`; those "
        "rows predate attestation and are flagged `unattested`, not falsified.",
        "",
        "## Artifact hashes",
        "",
    ]
    for name, meta in artifacts.items():
        lines.append(f"- `{name}`: `{meta['sha256']}` ({meta['bytes']} bytes)")
    if args.kernel_log:
        lines.append("")
        lines.append("The Kaggle kernel log is not committed; re-download and "
                     "verify its recorded sha256 above.")
    (directory / "KAGGLE_PROVENANCE.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {directory / 'KAGGLE_PROVENANCE.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
