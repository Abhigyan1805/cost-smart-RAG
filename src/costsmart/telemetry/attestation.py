"""Attestation audit for measured-model provenance (tiersweep-16).

The independent audit found that ``model_version`` was only what the *client*
sent: a row could be labelled ``Qwen/Qwen2.5-1.5B-Instruct`` while the
endpoint served a different checkpoint, and nothing detected it. The fix is
server-side attestation (``scripts/colab_local_tier.py serve`` reports the
resolved revision + weight hash, recorded in the ``model_revision`` /
``weights_sha256`` columns) plus this audit, which flags any *measured* row
that lacks server provenance.

Backward compatibility is deliberate: absent attestation is a **flag, not a
falsification**. Committed rows written before the columns existed keep their
original prediction/grade; the audit simply counts them as unattested.
"""

from __future__ import annotations

from .schema import (
    ATTESTATION_ATTESTED,
    ATTESTATION_LEGACY,
    ATTESTATION_STUB,
    ATTESTATION_UNATTESTED,
)


def has_server_provenance(row: dict) -> bool:
    """True when a row carries a server-reported revision or weight hash."""
    return bool(str(row.get("model_revision") or "").strip()
                or str(row.get("weights_sha256") or "").strip())


def attestation_status(row: dict) -> str:
    """Attestation status of one attempt/repeat row.

    A measured row is ``server-attested`` only when it carries server
    provenance; if the stored status already says so (defensive) it is
    trusted. Any other measured row - including legacy rows migrated to
    ``unflagged-legacy`` - is ``unattested`` and therefore **flagged**.
    Stub rows are ``stub``; non-measured legacy rows fall back to their
    stored status.
    """
    stored = str(row.get("attestation") or "").strip()
    if stored == ATTESTATION_ATTESTED:
        return ATTESTATION_ATTESTED
    if row.get("generator_mode") == "measured":
        return (ATTESTATION_ATTESTED if has_server_provenance(row)
                else ATTESTATION_UNATTESTED)
    if row.get("generator_mode") == "stub":
        return ATTESTATION_STUB
    if stored in (ATTESTATION_UNATTESTED, ATTESTATION_STUB):
        return stored
    return stored or ATTESTATION_LEGACY


def unattested_measured(rows: list[dict]) -> list[dict]:
    """Measured rows lacking server provenance (the audit finding set)."""
    return [r for r in rows if attestation_status(r) == ATTESTATION_UNATTESTED]


def audit(rows: list[dict], max_examples: int = 20) -> dict:
    """Count rows by attestation status and surface unattested examples.

    ``max_examples`` bounds the example list so the payload stays readable;
    the full counts are always exact.
    """
    counts = {
        ATTESTATION_ATTESTED: 0,
        ATTESTATION_UNATTESTED: 0,
        ATTESTATION_STUB: 0,
        ATTESTATION_LEGACY: 0,
    }
    measured = 0
    examples: list[dict] = []
    for row in rows:
        status = attestation_status(row)
        counts[status] = counts.get(status, 0) + 1
        if row.get("generator_mode") == "measured":
            measured += 1
        if status == ATTESTATION_UNATTESTED and len(examples) < max_examples:
            examples.append({
                "query_id": row.get("query_id"),
                "route_id": row.get("route_id"),
                "model_version": row.get("model_version"),
                "generator_mode": row.get("generator_mode"),
            })
    return {
        "n_rows": len(rows),
        "n_measured": measured,
        "counts": counts,
        "unattested_measured": len(unattested_measured(rows)),
        "unattested_examples": examples,
    }
