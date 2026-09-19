# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Add durable project-specific notes here as they are discovered through real work.

- Tests are unittest-style, stdlib-only: `PYTHONPATH=src python3 -m unittest discover -s tests`
  (`make test` needs pytest, which the pilot env may lack).
- Grading split: Tier-A (NQ/TriviaQA/HotpotQA, gold answers) uses deterministic
  EM/lenient-EM/token-F1; Tier-B free-form (no gold) uses the LLM judge — see
  `src/costsmart/eval/graders.py` and `docs/judge_labelling_guide.md`.
- `judge_stub` in `src/costsmart/eval/graders.py` is a frozen telemetry-slice
  contract (signature + return keys); extend, never silently break.
- Sweep telemetry provenance: `generator_mode`/`retrieval_mode` per attempt
  row (`measured` vs `stub`, legacy rows read `unflagged-legacy`) — see
  `src/costsmart/telemetry/schema.py` and `results/costsweep-08/CALIBRATION.md`.
  Live local-tier sweep: `oracle_sweep --live-local` (temp 0, fixed seed);
  cloud routes are stub-only by construction (no live cloud path, $0 spent).
- Worker env is stdlib-only (no pip/httpx): keep `registry.py` imports lazy
  and HTTP calls urllib-capable — see `ColabClient._post_json`.
- Noisy small-model labels: single live draws paraphrase-flip at temp 0
  (from ~0.87 L0 string flip on the legacy mix down to ~0.22 on the multihop
  mix); stabilize with 3x repeats + majority vote (ties = incorrect) —
  see `src/costsmart/eval/stability.py`, `scripts/run_repeats.py`
  (repeat rows live in a separate DB/table, never in `attempts`), and
  `scripts/stable_oracle.py` for the stable-oracle recount.
- Kaggle GPU route when the Colab free tier is exhausted:
  `kernels/costsweep-13-local-sweep/` clones the public repo, serves
  Qwen2.5-1.5B via `scripts/colab_local_tier.py`, and resumes the committed
  telemetry DBs by cache key; `kernels push` does NOT upload sibling files,
  so the kernel fetches the repo itself. Exact push/status/output + ingest
  commands: `docs/kaggle-handoff.md`.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
