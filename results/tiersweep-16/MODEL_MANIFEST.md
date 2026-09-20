# tiersweep-16 cheap-tier model manifest

Every checkpoint swept as a cheap tier, with its resolved revision and
licence. Revisions + weight hashes for the 3B/7B tiers are **server-attested**
(the serving endpoint reported the loaded snapshot; recorded per row as
`model_revision` / `weights_sha256` / `attestation='server-attested'`). The
1.5B tier reuses the committed realdata-15 run and predates attestation, so
its revision is **not** attested - recorded here as unknown rather than
inferred, and flagged `unattested` by `scripts/attestation_audit.py`.

| tier | model id | routes | resolved revision | weights sha256 | licence |
|---|---|---|---|---|---|
| local-small | `Qwen/Qwen2.5-1.5B-Instruct` | L0, L1 | not attested (realdata-15 predates attestation; HF `main` card sha `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`) | not attested | Apache-2.0 |
| local-3b | `Qwen/Qwen2.5-3B-Instruct` | L0, L1, C0 | `aa8e72537993ba99e69dfaafa59ed015b17504d1` | `04a84318cdec5543e99a10e0db40df3f7843226f88a8c0e840da95bbfa52487c` | Qwen Research License (`license: other`, `qwen-research`) |
| local-medium | `Qwen/Qwen2.5-7B-Instruct` | L0, L1, C0 | `a09a35458c702b33eeacc393d103063234e8bc28` | `175d8c141750c8e6f8af96dda0d1870efdc15da736ff36cd25105a5d11241367` | Apache-2.0 |

## Licence source

Read from each model's HuggingFace card `cardData.license` on 2026-09-20
(`https://huggingface.co/api/models/<id>`); the 3B card names the Qwen
Research License (`https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE`).
Licences are recorded, never inferred from memory.

**Product note.** The first tier whose closed-book L0 coverage clears the
0.10 gate is the **3B**, which is under the **Qwen Research License** (not a
permissive licence); the 7B is Apache-2.0. A commercially deployable
break-even therefore depends on 7B, whose L0 coverage is statistically
indistinguishable from 3B's (see `TIERS.md`).

## Pinned config

`config/models.yaml` declares `local-small` (1.5B), `local-3b` (3B, added by
this task) and `local-medium` (7B). Each experiment row's `model_version` is
the exact HF id, and the server-attested revision is stored alongside it.
