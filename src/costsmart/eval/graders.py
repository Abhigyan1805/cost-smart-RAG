"""Answer graders: deterministic EM/lenient-EM/token-F1 + validated LLM judge.

Tier-A datasets (NQ, TriviaQA, HotpotQA) ship gold answers, so the oracle
sweep grades them deterministically: strict ``exact_match``, SQuAD-style
``lenient_exact_match`` (lowercase, strip punctuation/articles/whitespace),
and bag-of-tokens ``token_f1``. Tier-B answers are free-form (no gold), so
they are graded by the LLM judge below.

The judge is a measurement instrument with error bars, not ground truth:

* explicit rubric prompt (:data:`JUDGE_RUBRIC`, rendered by
  :func:`build_judge_prompt`),
* a strong model run at temperature 0 (:data:`JUDGE_MODEL`,
  :data:`JUDGE_TEMPERATURE`),
* agreement against hand labels via :func:`cohen_kappa` /
  :func:`kappa_from_file` with a kappa >= 0.7 gate (:data:`KAPPA_GATE`),
* a second-judge disagreement-rate check (:func:`judge_disagreement_rate`).

Numeric answers get deliberately minimal normalization: numbers are compared
as tokens (``"42"`` matches ``"42."`` after punctuation stripping, but
``"42"`` does not match ``"42.0"`` and no unit conversion is attempted).
Semantic numeric equivalence is judge territory, not deterministic-grader
territory.

Back-compatibility: :func:`judge_stub` keeps its exact signature and return
keys (it only gains a deprecation note). :func:`grade` keeps its default
output identical; it accepts an optional precomputed ``judge_result`` for
the backfill path.
"""

from __future__ import annotations

import csv
import json
import re
import string
from pathlib import Path

_WS_PUNCT = re.compile(rf"[{re.escape(string.punctuation)}]")
_WS = re.compile(r"\s+")
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize(text: str) -> str:
    """Lowercase, strip punctuation/articles/extra whitespace (SQuAD-style)."""
    text = text.lower()
    text = _WS_PUNCT.sub(" ", text)
    text = _ARTICLES.sub(" ", text)
    return _WS.sub(" ", text).strip()


def exact_match(prediction: str, reference: str) -> int:
    """Strict EM: 1 on exact string equality, else 0."""
    return int(prediction == reference)


def lenient_exact_match(prediction: str, reference: str) -> int:
    """Lenient EM: 1 when normalized strings match, else 0."""
    return int(normalize(prediction) == normalize(reference))


def token_f1(prediction: str, reference: str) -> float:
    """Bag-of-tokens F1 over normalized text (0.0 when either side empty)."""
    pred_tokens = normalize(prediction).split()
    ref_tokens = normalize(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = 0
    counts: dict[str, int] = {}
    for tok in ref_tokens:
        counts[tok] = counts.get(tok, 0) + 1
    for tok in pred_tokens:
        if counts.get(tok, 0) > 0:
            counts[tok] -= 1
            common += 1
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# LLM judge (Tier-B free-form answers)
# ---------------------------------------------------------------------------

#: C2 strong-route model (``config/models.yaml`` cloud-large). The judge must
#: NOT be this model: grading C2's own outputs with C2 inflates agreement via
#: self-preference bias.
C2_STRONG_MODEL = "gpt-4o-2024-08-06"

#: Primary judge: a strong model from a different family than the C2 strong
#: route, so judge-vs-C2 agreement cannot be explained by self-preference.
#: Re-pin to the current strong non-OpenAI model before any reported run and
#: record the pin in the run metadata.
JUDGE_MODEL = "claude-3-5-sonnet-20241022"

#: Second judge for the disagreement-rate check. Different from both the C2
#: strong route and the primary judge; already pinned in
#: ``config/models.yaml`` (cloud-small).
SECOND_JUDGE_MODEL = "gpt-4o-mini-2024-07-18"

#: The judge is a measurement instrument: temperature 0 for deterministic,
#: reproducible scores.
JUDGE_TEMPERATURE = 0

#: Explicit rubric prompt. Three-level scale keeps hand-labeller agreement
#: achievable (binary-correct vs partial vs incorrect) while remaining
#: fine-grained enough for frontier analysis.
JUDGE_RUBRIC = """You are grading a question-answering system. Score the PREDICTED ANSWER
against the QUESTION (and the REFERENCE ANSWER when one is provided).

Rubric (assign exactly one level):
- score 1.0 (correct): the predicted answer contains the reference answer's
  key fact(s), or — when no reference is given — directly and correctly
  answers the question with no material falsehood.
- score 0.5 (partial): the predicted answer is on-topic and partly correct
  but omits a key fact, adds an important falsehood, or answers a different
  question than asked.
- score 0.0 (incorrect): the predicted answer is wrong, irrelevant,
  a refusal, or empty.

Rules:
- Judge semantic equivalence, not wording: paraphrases, unit spellings
  ("meters per second" vs "m/s"), and numeric rephrasings of the same value
  count as correct.
- When a REFERENCE ANSWER is provided it is authoritative; do not overrule it
  with outside knowledge.
- When no reference is provided, grade correctness and completeness directly;
  penalize hallucinations.
- Be strict about numbers, names, and dates: a wrong value is incorrect,
  not partial.

Respond with a single JSON object only, no other text:
{"score": <0.0, 0.5, or 1.0>, "rationale": "<one sentence>"}
"""


def build_judge_prompt(question: str, prediction: str, reference: str | None = None) -> str:
    """Render the judge prompt for one (question, prediction) pair.

    ``reference`` is optional: Tier-B free-form pairs have no gold answer, in
    which case the rubric directs the judge to grade correctness directly.
    """
    lines = [
        JUDGE_RUBRIC.strip(),
        "",
        f"QUESTION: {question}",
        f"PREDICTED ANSWER: {prediction}",
    ]
    if reference:
        lines.append(f"REFERENCE ANSWER: {reference}")
    else:
        lines.append("REFERENCE ANSWER: (none provided — grade correctness directly)")
    return "\n".join(lines) + "\n"


def parse_judge_output(text: str) -> tuple[float, str]:
    """Parse raw judge output into a clamped ``(score, rationale)`` pair.

    Expects one JSON object with ``score`` and ``rationale`` keys (leading /
    trailing prose tolerated). Raises :class:`ValueError` when no usable
    score is found — callers must leave the score NULL, never fabricate one.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"judge output contains no JSON object: {text[:120]!r}")
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"judge output is not valid JSON: {text[:120]!r}") from exc
    try:
        score = float(payload["score"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"judge JSON has no numeric 'score': {payload!r}") from exc
    rationale = str(payload.get("rationale", ""))
    return max(0.0, min(1.0, score)), rationale


def judge_with_llm(
    question: str,
    prediction: str,
    reference: str | None = None,
    client=None,
    judge_model: str = JUDGE_MODEL,
    temperature: int = JUDGE_TEMPERATURE,
) -> dict:
    """Grade one free-form answer with the LLM judge.

    ``client`` is any object with ``generate(prompt, temperature=...)``
    returning an object with a ``.text`` attribute (compatible with
    :class:`costsmart.models.base.LLMClient`). With ``client=None`` no call
    is made: the prompt, model, and temperature are recorded for backfill and
    ``judge_score`` stays NULL (same contract as :func:`judge_stub`).
    Unparseable judge output likewise leaves the score NULL with an error
    note instead of a fabricated score.
    """
    prompt = build_judge_prompt(question, prediction, reference)
    if client is None:
        return {
            "judge_score": None,
            "judge_model": judge_model,
            "judge_temperature": temperature,
            "prompt": prompt,
            "note": "judge not wired; score left NULL for backfill",
            "graded": False,
        }
    raw = client.generate(prompt, temperature=temperature)
    try:
        score, rationale = parse_judge_output(raw.text)
    except ValueError as exc:
        return {
            "judge_score": None,
            "judge_model": judge_model,
            "judge_temperature": temperature,
            "prompt": prompt,
            "note": f"judge output unparseable, score left NULL: {exc}",
            "graded": False,
        }
    return {
        "judge_score": score,
        "judge_model": judge_model,
        "judge_temperature": temperature,
        "rationale": rationale,
        "graded": True,
    }


def judge_disagreement_rate(
    scores_a: list[float], scores_b: list[float], tol: float = 0.25
) -> dict:
    """Disagreement rate between two judges over parallel score lists.

    Two scores disagree when ``abs(a - b) > tol`` (default straddles the
    0.5 rubric step, so adjacent-level splits count but rounding noise does
    not). The live second judge (``SECOND_JUDGE_MODEL``) is not wired yet —
    pass its scores in once it is; this rate computation itself is final and
    is what gates judge-vs-judge trust. Raises :class:`ValueError` on
    length mismatch.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"parallel score lists required, got {len(scores_a)} vs {len(scores_b)}"
        )
    n = len(scores_a)
    disagreements = sum(1 for a, b in zip(scores_a, scores_b) if abs(a - b) > tol)
    return {
        "n": n,
        "disagreements": disagreements,
        "disagreement_rate": (disagreements / n) if n else 0.0,
        "tol": tol,
        "judge_a": JUDGE_MODEL,
        "judge_b": SECOND_JUDGE_MODEL,
    }


def judge_stub(prediction: str, reference: str, judge_model: str = "") -> dict:
    """LLM-judge placeholder: returns no score until a judge is wired.

    DEPRECATED for new code — prefer :func:`judge_with_llm`, which carries
    the explicit rubric, temperature-0 call, and backfill prompt. This stub
    is kept with its exact signature and return keys so telemetry-slice
    callers never break silently.

    Pilot only - the sweep leaves ``judge_score`` NULL and records which
    (absent) judge would have scored it, so later slices can backfill.
    """
    return {
        "judge_score": None,
        "judge_model": judge_model or "judge-stub/unwired",
        "note": "judge not wired in pilot; score left NULL for backfill",
        "graded": False,
    }


# ---------------------------------------------------------------------------
# Human agreement (Cohen's kappa) + acceptance gate
# ---------------------------------------------------------------------------

#: Minimum judge-vs-human Cohen's kappa before ``judge_score`` may be trusted
#: in sweep analysis. Below this the judge is recalibrated or replaced.
KAPPA_GATE = 0.7


def cohen_kappa(labels_a: list, labels_b: list) -> float:
    """Cohen's kappa hook for judge-vs-human agreement (-1..1).

    Takes two parallel discrete label lists (binarize scores first).
    Returns 0.0 on degenerate inputs (no agreement possible to measure).
    """
    if len(labels_a) != len(labels_b) or not labels_a:
        return 0.0
    classes = sorted(set(labels_a) | set(labels_b))
    if len(classes) < 2:
        return 0.0
    n = len(labels_a)
    observed = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    expected = sum(
        (sum(1 for a in labels_a if a == c) / n) * (sum(1 for b in labels_b if b == c) / n)
        for c in classes
    )
    if expected >= 1.0:
        return 0.0
    return (observed - expected) / (1 - expected)


def passes_kappa_gate(kappa: float, gate: float = KAPPA_GATE) -> bool:
    """True when judge-vs-human agreement clears the acceptance gate."""
    return kappa >= gate


_JUDGE_COLUMNS = ("judge", "judge_label", "judge_score", "llm", "llm_label")
_HUMAN_COLUMNS = ("human", "human_label", "captain", "captain_label", "gold_human")


def _pick_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {name.strip().lower(): name for name in fieldnames}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def _coerce_label(value: object) -> object:
    """Normalize one label cell: strip/lowercase text, keep numbers numeric."""
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text.lower()


def load_labels_file(path: str | Path) -> tuple[list, list]:
    """Load ``(judge_labels, human_labels)`` from the captain's labels file.

    Accepts CSV (header must name a judge column and a human column, see
    :data:`docs/judge_labelling_guide.md`) or JSONL (one object per line
    with judge/human keys). Rows with a blank on either side are skipped and
    reported via the ``skipped`` count in :func:`kappa_from_file` metadata.

    The captain supplies this file separately; nothing here fabricates
    labels — a missing file or missing columns raises, never synthesizes.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"labels file not found (captain supplies it): {path}")
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        judge_key = next(
            (k for k in _JUDGE_COLUMNS if rows and k in {kk.lower() for kk in rows[0]}),
            None,
        )
        human_key = next(
            (k for k in _HUMAN_COLUMNS if rows and k in {kk.lower() for kk in rows[0]}),
            None,
        )
        if judge_key is None or human_key is None:
            raise ValueError(
                f"{path} rows need judge + human keys "
                f"(judge-like: {_JUDGE_COLUMNS}, human-like: {_HUMAN_COLUMNS})"
            )
        keymap = {k.lower(): k for k in rows[0]}
        pairs = [
            (_coerce_label(r.get(keymap[judge_key])), _coerce_label(r.get(keymap[human_key])))
            for r in rows
        ]
    elif path.suffix.lower() == ".csv":
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames:
                raise ValueError(f"{path} has no CSV header row")
            judge_col = _pick_column(reader.fieldnames, _JUDGE_COLUMNS)
            human_col = _pick_column(reader.fieldnames, _HUMAN_COLUMNS)
            if judge_col is None or human_col is None:
                raise ValueError(
                    f"{path} header needs judge + human columns "
                    f"(judge-like: {_JUDGE_COLUMNS}, human-like: {_HUMAN_COLUMNS})"
                )
            pairs = [(_coerce_label(r[judge_col]), _coerce_label(r[human_col])) for r in reader]
    else:
        raise ValueError(f"unsupported labels format {path.suffix!r}; use .csv or .jsonl")
    kept = [(j, h) for j, h in pairs if j != "" and h != ""]
    skipped = len(pairs) - len(kept)
    judge_labels = [j for j, _ in kept]
    human_labels = [h for _, h in kept]
    load_labels_file.skipped = skipped  # type: ignore[attr-defined]
    return judge_labels, human_labels


def _binarize(values: list, threshold: float) -> list[int]:
    out = []
    for v in values:
        try:
            out.append(int(float(v) >= threshold))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"cannot binarize non-numeric label {v!r} at threshold {threshold}; "
                "use discrete labels or numeric scores"
            ) from exc
    return out


def kappa_from_file(
    path: str | Path, threshold: float | None = None, gate: float = KAPPA_GATE
) -> dict:
    """Cohen's kappa between judge and human labels from a labels file.

    ``threshold`` binarizes numeric scores (``value >= threshold`` -> 1)
    for continuous judge outputs; omit it for already-discrete labels.
    Returns kappa, n, skipped blanks, and the gate verdict.
    """
    judge_labels, human_labels = load_labels_file(path)
    skipped = getattr(load_labels_file, "skipped", 0)
    if threshold is not None:
        judge_labels = _binarize(judge_labels, threshold)
        human_labels = _binarize(human_labels, threshold)
    kappa = cohen_kappa(judge_labels, human_labels)
    return {
        "kappa": kappa,
        "n": len(judge_labels),
        "skipped_blank": skipped,
        "gate": gate,
        "passes_gate": passes_kappa_gate(kappa, gate),
    }


def grade(prediction: str, reference: str, judge_result: dict | None = None) -> dict:
    """All cheap graders for one attempt (judge stays a stub by default).

    ``judge_result`` accepts a precomputed :func:`judge_with_llm` output for
    the backfill path; when omitted the sweep contract is unchanged
    (``judge_score`` NULL, stub model recorded).
    """
    if judge_result is None:
        judged = judge_stub(prediction, reference)
        judge_score, judge_model = judged["judge_score"], judged["judge_model"]
    else:
        judge_score = judge_result.get("judge_score")
        judge_model = judge_result.get("judge_model", JUDGE_MODEL)
    return {
        "exact_match": exact_match(prediction, reference),
        "lenient_em": lenient_exact_match(prediction, reference),
        "token_f1": token_f1(prediction, reference),
        "judge_score": judge_score,
        "judge_model": judge_model,
    }
