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

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
