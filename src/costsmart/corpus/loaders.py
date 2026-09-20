"""Pilot corpus loaders for the Week-1 costsmart-rag pilot.

Covers five Tier-A datasets: Natural Questions (NQ), TriviaQA (single-hop
lookup), and HotpotQA / 2WikiMultihopQA / MuSiQue (multi-hop).

Two mixes (both n=200, offline synthetic default):
  * pilot (legacy): 80 NQ + 70 TriviaQA + 50 HotpotQA (75% single-hop).
  * multihop (costmultihop-12): 30 NQ + 20 TriviaQA + 60 HotpotQA +
    50 2WikiMultihopQA + 40 MuSiQue (75% multi-hop) - reweights toward
    questions that need 2+ passages joined, where closed-book local tiers
    collapse and retrieval+reasoning tiers separate.

Three modes:
  * ``source="hf"``  -- load the real datasets via HuggingFace ``datasets``
    (requires the ``datasets`` package + network). Used to *fetch* the real
    pilot; :func:`normalize_hf_row` converts each dataset's raw schema into
    the repo record shape (question/answers/gold passages).
  * ``source="real"`` -- load the committed, normalized real Tier-A corpus
    (``results/realdata-15/corpus.json``) with no network and no ``datasets``
    dependency, so every offline recount runs on the exact fetched rows.
  * ``source="synthetic"`` (default) -- build a deterministic, seeded offline
    pilot with the same query-id scheme and sizes, so the index build and
    smoke retrieval run anywhere without network access.

Query-id scheme (stable across modes): ``{dataset}:{index:04d}``,
e.g. ``nq:0007``, ``triviaqa:0042``, ``hotpotqa:0013``, ``2wiki:0001``,
``musique:0002``.

Each query record: ``query_id``, ``source`` (nq|triviaqa|hotpotqa|2wiki|
musique), ``question``, ``answers`` (list[str]), ``gold_passage_ids``
(list[str]).
Each passage record: ``passage_id``, ``doc_id``, ``text``.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

# Pilot composition (legacy, frozen): 200 queries total across three Tier-A
# datasets. Do not change - results/costsweep-08/splits_freeze.json pins it.
PILOT_SIZES = {
    "nq": 80,
    "triviaqa": 70,
    "hotpotqa": 50,
}

PILOT_TOTAL = sum(PILOT_SIZES.values())

# Multihop composition (costmultihop-12): 200 queries, 75% multi-hop.
# Difficulty ladder: nq/triviaqa (1 hop, lookup) < hotpotqa (2-hop
# bridge/comparison) < 2wiki (2-5 hop compositional/inference chains) <
# musique (2-4 hop + distractor passages that must be filtered).
MULTIHOP_SIZES = {
    "nq": 30,
    "triviaqa": 20,
    "hotpotqa": 60,
    "2wiki": 50,
    "musique": 40,
}

MULTIHOP_TOTAL = sum(MULTIHOP_SIZES.values())

MULTIHOP_ORDER = ("nq", "triviaqa", "hotpotqa", "2wiki", "musique")

# Every dataset loadable by name (legacy pilot sizes stay the default n for
# the original three; the multihop pair defaults to MULTIHOP_SIZES).
KNOWN_DATASETS = tuple(sorted(set(PILOT_SIZES) | set(MULTIHOP_SIZES)))

# Real Tier-A fetch specs (realdata-15). Pinned HuggingFace repo ids, the
# config + split each loader reads, and the answer-field shape. The licence
# is NOT pinned here - scripts/fetch_tier_a.py reads it from the HF dataset
# card at fetch time and records it in the committed manifest, so no licence
# is ever inferred from memory.
HF_DATASET_SPECS = {
    "nq": {
        "repo": "google-research-datasets/nq_open",
        "config": None,
        "split": "validation",
        "kind": "nq_open",
    },
    "triviaqa": {
        "repo": "mandarjoshi/trivia_qa",
        "config": "rc",
        "split": "validation",
        "kind": "trivia_qa",
    },
    "hotpotqa": {
        "repo": "hotpotqa/hotpot_qa",
        "config": "distractor",
        "split": "validation",
        "kind": "hotpot_qa",
    },
    "2wiki": {
        "repo": "voidful/2wiki",
        "config": None,
        "split": "validation",
        "kind": "raw",
    },
    "musique": {
        "repo": "dgslibisey/MuSiQue",
        "config": "default",
        "split": "validation",
        "kind": "musique",
    },
}

# Back-compat alias: the dataset id each corpus name fetches from.
HF_DATASET_NAMES = {name: spec["repo"] for name, spec in HF_DATASET_SPECS.items()}

#: Committed normalized real Tier-A corpus (realdata-15). Written by
#: ``scripts/fetch_tier_a.py`` on a network-capable host, then committed so
#: every sweep/recount downstream is offline + deterministic.
REAL_CORPUS_PATH = "results/realdata-15/corpus.json"

#: Default real-data mix (realdata-15): 200 queries, 75% multi-hop, mirroring
#: the synthetic ``costmultihop-12`` composition so the real-vs-synthetic
#: recount is a like-for-like comparison. NQ has no gold passage (nq_open
#: ships question+answer only), so its L1 retrieval pool is the shared
#: multi-hop pool; the gate does not read L1.
REAL_SIZES = {
    "nq": 50,
    "hotpotqa": 80,
    "musique": 70,
}

REAL_TOTAL = sum(REAL_SIZES.values())

REAL_ORDER = ("nq", "hotpotqa", "musique")

# Deterministic offline seed so every checkout builds the same pilot.
SYNTHETIC_SEED = 20260919

# Small topical pools used to synthesize realistic offline pilot content.
_TOPICS = [
    ("photosynthesis", "Photosynthesis converts light energy into chemical energy in plant chloroplasts."),
    ("Eiffel Tower", "The Eiffel Tower in Paris was completed in 1889 for the World's Fair."),
    ("mitochondria", "Mitochondria generate ATP through oxidative phosphorylation in eukaryotic cells."),
    ("Great Wall of China", "The Great Wall of China stretches over 21,000 km across northern China."),
    ("World War II", "World War II ended in 1945 with the surrender of Germany and Japan."),
    ("quantum mechanics", "Quantum mechanics describes matter and energy at atomic and subatomic scales."),
    ("Marie Curie", "Marie Curie won Nobel Prizes in Physics (1903) and Chemistry (1911)."),
    ("Amazon River", "The Amazon River carries more water than any other river in the world."),
    ("Renaissance", "The Renaissance was a period of cultural rebirth in Europe from the 14th to 17th century."),
    ("black holes", "Black holes are regions of spacetime where gravity prevents anything from escaping."),
    ("Shakespeare", "William Shakespeare wrote 37 plays including Hamlet and Macbeth."),
    ("Pacific Ocean", "The Pacific Ocean is the largest and deepest ocean basin on Earth."),
    ("DNA", "DNA carries genetic instructions in a double-helix structure discovered in 1953."),
    ("Roman Empire", "The Roman Empire at its height spanned three continents around the Mediterranean."),
    ("gravity", "Newton's law of universal gravitation describes attraction between masses."),
    ("Mount Everest", "Mount Everest rises 8,849 metres on the Nepal-China border."),
    ("Industrial Revolution", "The Industrial Revolution began in Britain in the late 18th century."),
    ("neural networks", "Neural networks learn representations through layered weighted connections."),
    ("Mona Lisa", "The Mona Lisa by Leonardo da Vinci hangs in the Louvre in Paris."),
    ("Sahara Desert", "The Sahara is the largest hot desert, covering much of North Africa."),
]

_QUESTION_TEMPLATES = {
    "nq": [
        "what is {topic} known for",
        "where is {topic} located",
        "when did {topic} happen",
        "who discovered {topic}",
    ],
    "triviaqa": [
        "Which {topic} fact is described as: {fact}",
        "Name the {topic} associated with this description: {fact}",
    ],
    "hotpotqa": [
        "What connects {topic_a} and {topic_b} historically",
        "Compare {topic_a} with {topic_b}: which came first and why",
    ],
    # 2WikiMultihopQA: compositional / inference chains over a bridge entity.
    # The answer (topic_b) is only reachable via the bridge passage, so a
    # closed-book direct answer must guess the join.
    "2wiki": [
        "Which {topic_b} is linked to {topic_a} through {bridge}",
        "The {bridge} connects {topic_a} to which {topic_b}, and why",
    ],
    # MuSiQue: nested multi-hop with distractor passages that must be
    # filtered (two distractors per query vs one elsewhere).
    "musique": [
        "Following the link from {topic_a} via {bridge}, which {topic_b} is described, excluding unrelated {distractor}",
        "Which {topic_b} connects {topic_a} and {bridge}, distinguishing it from {distractor}",
    ],
}


def _rng_for(dataset: str) -> random.Random:
    digest = hashlib.sha256(f"{SYNTHETIC_SEED}/{dataset}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def _synthetic_queries(dataset: str, n: int) -> tuple[list[dict], list[dict]]:
    """Build deterministic synthetic queries + passages for one dataset."""
    rng = _rng_for(dataset)
    queries: list[dict] = []
    passages: list[dict] = []
    for i in range(n):
        qid = f"{dataset}:{i:04d}"
        n_distractors = 0
        if dataset == "hotpotqa":
            (ta, fa), (tb, fb) = rng.sample(_TOPICS, 2)
            tmpl = rng.choice(_QUESTION_TEMPLATES[dataset])
            question = tmpl.format(topic_a=ta, topic_b=tb)
            answer = ta
            gold_texts = [
                f"{fa} Further context on {ta} relates it to {tb}.",
                f"{fb} Scholars link {tb} back to developments around {ta}.",
            ]
            n_distractors = 1
        elif dataset in ("2wiki", "musique"):
            # Compositional chain: topic_a -> bridge -> topic_b; the answer
            # (topic_b) requires joining both gold passages.
            (ta, fa), (tb, fb), (br, _) = rng.sample(_TOPICS, 3)
            tmpl = rng.choice(_QUESTION_TEMPLATES[dataset])
            if dataset == "musique":
                dt = rng.choice(_TOPICS)[0]
                question = tmpl.format(topic_a=ta, topic_b=tb, bridge=br, distractor=dt)
                n_distractors = 2
            else:
                question = tmpl.format(topic_a=ta, topic_b=tb, bridge=br)
                n_distractors = 1
            answer = tb
            gold_texts = [
                f"{fa} The record for {ta} points onward to {br}.",
                f"{fb} Via {br}, the chain resolves to {tb}.",
            ]
        else:
            topic, fact = rng.choice(_TOPICS)
            tmpl = rng.choice(_QUESTION_TEMPLATES[dataset])
            question = tmpl.format(topic=topic, fact=fact)
            answer = topic
            gold_texts = [f"{fact} Additional detail: {topic} is widely documented."]
            n_distractors = 1
        gold_ids = []
        for g, text in enumerate(gold_texts):
            pid = f"{qid}#p{g}"
            gold_ids.append(pid)
            passages.append({"passage_id": pid, "doc_id": qid, "text": text})
        # Distractor passages so retrieval is non-trivial (MuSiQue gets two,
        # reflecting its distractor-filtering difficulty).
        for d in range(n_distractors):
            d_topic, d_fact = rng.choice(_TOPICS)
            passages.append(
                {
                    "passage_id": f"{qid}#d{d}",
                    "doc_id": f"{qid}-distractor",
                    "text": f"Unrelated background on {d_topic}: {d_fact}",
                }
            )
        queries.append(
            {
                "query_id": qid,
                "source": dataset,
                "question": question,
                "answers": [answer],
                "gold_passage_ids": gold_ids,
            }
        )
    return queries, passages


def normalize_hf_row(dataset: str, row: dict) -> dict:
    """Normalize one raw HuggingFace row into the repo record shape.

    Pure + stdlib-only so it is unit-testable without ``datasets`` installed.
    Returns ``{"question", "answers", "passages"}`` where ``passages`` is a
    list of ``{"title", "text", "is_gold"}``. The caller assigns stable
    ``{qid}#p{j}`` passage ids from the list order. Never fabricates a
    passage: a dataset with no context (``nq_open``) yields ``[]``.
    """
    if dataset not in HF_DATASET_SPECS:
        raise ValueError(f"no HuggingFace spec for {dataset!r}")
    kind = HF_DATASET_SPECS[dataset]["kind"]
    question = str(row.get("question", "")).strip()
    passages: list[dict] = []

    if kind == "nq_open":
        answers = [str(a) for a in (row.get("answer") or []) if str(a).strip()]
    elif kind == "trivia_qa":
        ans = row.get("answer") or {}
        answers = [str(ans.get("value"))] if ans.get("value") else []
        pages = row.get("entity_pages") or {}
        for title, text in zip(pages.get("title") or [],
                               pages.get("wiki_context") or []):
            if str(text).strip():
                passages.append({"title": str(title), "text": str(text),
                                 "is_gold": True})
        results = row.get("search_results") or {}
        for title, text in zip(results.get("title") or [],
                               results.get("search_context") or []):
            if str(text).strip():
                passages.append({"title": str(title), "text": str(text),
                                 "is_gold": False})
    elif kind == "hotpot_qa":
        answers = [str(row["answer"])] if row.get("answer") else []
        ctx = row.get("context") or {}
        titles = list(ctx.get("title") or [])
        sentences = list(ctx.get("sentences") or [])
        support = set((row.get("supporting_facts") or {}).get("title") or [])
        for title, sent in zip(titles, sentences):
            text = " ".join(str(s) for s in sent).strip()
            if not text:
                continue
            passages.append({"title": str(title), "text": text,
                             "is_gold": str(title) in support})
    elif kind == "musique":
        answers = [str(row["answer"])] if row.get("answer") else []
        for alias in row.get("answer_aliases") or []:
            if str(alias).strip() and str(alias) not in answers:
                answers.append(str(alias))
        for para in row.get("paragraphs") or []:
            text = str(para.get("paragraph_text", "")).strip()
            if not text:
                continue
            passages.append({"title": str(para.get("title", "")), "text": text,
                             "is_gold": bool(para.get("is_supporting"))})
    else:  # raw: unknown schema; only the answer string is trustworthy
        answers = [str(row["answer"])] if row.get("answer") else []
    if not question:
        raise ValueError(f"{dataset} row has no question: {str(row)[:120]!r}")
    return {"question": question, "answers": answers, "passages": passages}


def _hf_queries(dataset: str, n: int):
    """Load real queries via HuggingFace ``datasets`` (requires network)."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "HuggingFace 'datasets' package not installed; "
            "use source='synthetic' or source='real' for offline runs."
        ) from exc
    spec = HF_DATASET_SPECS[dataset]
    ds = load_dataset(spec["repo"], spec["config"], split=spec["split"])
    queries: list[dict] = []
    passages: list[dict] = []
    taken = 0
    for row in ds:
        if taken >= n:
            break
        if dataset == "musique" and row.get("answerable") is False:
            continue
        record = normalize_hf_row(dataset, dict(row))
        if not record["answers"]:
            continue
        qid = f"{dataset}:{taken:04d}"
        gold_ids: list[str] = []
        for j, passage in enumerate(record["passages"]):
            pid = f"{qid}#p{j}"
            passages.append({"passage_id": pid, "doc_id": passage["title"] or qid,
                             "text": passage["text"]})
            if passage["is_gold"]:
                gold_ids.append(pid)
        queries.append(
            {
                "query_id": qid,
                "source": dataset,
                "question": record["question"],
                "answers": record["answers"],
                "gold_passage_ids": gold_ids,
            }
        )
        taken += 1
    return queries, passages


# Cache the parsed real corpus so repeated per-dataset slicing reads one file.
_REAL_CORPUS_CACHE: dict[str, tuple[str, dict]] = {}


def load_real_corpus(path: str | None = None) -> dict:
    """Load + cache the committed normalized real Tier-A corpus JSON."""
    resolved = str(Path(path or REAL_CORPUS_PATH))
    cached = _REAL_CORPUS_CACHE.get(resolved)
    if cached is not None:
        return cached[1]
    corpus_path = Path(resolved)
    if not corpus_path.exists():
        raise RuntimeError(
            f"real corpus {resolved} not found; fetch it with "
            "scripts/fetch_tier_a.py on a network host, or use "
            "source='synthetic'/'hf'."
        )
    corpus = json.loads(corpus_path.read_text())
    if not corpus.get("queries"):
        raise RuntimeError(f"real corpus {resolved} has no queries")
    _REAL_CORPUS_CACHE[resolved] = (resolved, corpus)
    return corpus


def real_queries_for(dataset: str, n: int | None = None,
                     path: str | None = None) -> tuple[list[dict], list[dict]]:
    """Slice one dataset out of the committed real corpus (offline)."""
    corpus = load_real_corpus(path)
    wanted = [q for q in corpus["queries"] if q.get("source") == dataset]
    if n is not None:
        wanted = wanted[:n]
    qids = {q["query_id"] for q in wanted}
    passages = [p for p in corpus.get("passages", [])
                if p.get("passage_id", "").rsplit("#", 1)[0] in qids]
    return [dict(q) for q in wanted], [dict(p) for p in passages]


def load_real_subset(path: str | None = None) -> tuple[list[dict], list[dict]]:
    """Load the committed real Tier-A corpus (all datasets, frozen order)."""
    corpus = load_real_corpus(path)
    queries = [dict(q) for q in corpus["queries"]]
    passages = [dict(p) for p in corpus.get("passages", [])]
    return queries, passages


def load_real_dataset_queries(dataset: str, n: int | None = None,
                              path: str | None = None) -> tuple[list[dict], list[dict]]:
    """Back-compat alias used by :func:`load_dataset_queries`."""
    return real_queries_for(dataset, n, path)


def load_dataset_queries(
    dataset: str, n: int | None = None, source: str = "synthetic"
) -> tuple[list[dict], list[dict]]:
    """Load (queries, passages) for one dataset.

    Returns a ``(queries, passages)`` tuple. ``n`` defaults to the legacy
    pilot size for the original three datasets (:data:`PILOT_SIZES`) or to
    the multihop size for ``2wiki``/``musique`` (:data:`MULTIHOP_SIZES`).
    """
    if dataset not in KNOWN_DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {sorted(KNOWN_DATASETS)}")
    if n is None:
        n = PILOT_SIZES.get(dataset, MULTIHOP_SIZES.get(dataset, 0))
    if source == "hf":
        if dataset not in HF_DATASET_SPECS:
            raise ValueError(f"no HuggingFace mapping for {dataset!r}")
        return _hf_queries(dataset, n)
    if source == "real":
        return load_real_dataset_queries(dataset, n)
    if source == "synthetic":
        return _synthetic_queries(dataset, n)
    raise ValueError(
        f"unknown source {source!r}; expected 'hf', 'real', or 'synthetic'")


def load_pilot_subset(source: str = "synthetic") -> tuple[list[dict], list[dict]]:
    """Load the full 200-query legacy pilot across NQ + TriviaQA + HotpotQA.

    Frozen: results/costsweep-08/splits_freeze.json pins this exact mix.
    """
    all_queries: list[dict] = []
    all_passages: list[dict] = []
    for dataset in ("nq", "triviaqa", "hotpotqa"):
        queries, passages = load_dataset_queries(dataset, source=source)
        all_queries.extend(queries)
        all_passages.extend(passages)
    assert len(all_queries) == PILOT_TOTAL, f"pilot must be {PILOT_TOTAL} queries"
    return all_queries, all_passages


def load_multihop_subset(
    source: str = "synthetic", counts: dict | None = None
) -> tuple[list[dict], list[dict]]:
    """Load the 200-query multihop mix (costmultihop-12).

    Default counts are :data:`MULTIHOP_SIZES` (75% multi-hop); ``counts``
    may override per-dataset sizes with the same ``{dataset: n}`` shape.
    """
    counts = dict(counts or MULTIHOP_SIZES)
    unknown = set(counts) - set(KNOWN_DATASETS)
    if unknown:
        raise ValueError(f"unknown datasets {sorted(unknown)}; expected subset of {sorted(KNOWN_DATASETS)}")
    all_queries: list[dict] = []
    all_passages: list[dict] = []
    for dataset in MULTIHOP_ORDER:
        if dataset not in counts:
            continue
        queries, passages = load_dataset_queries(dataset, n=counts[dataset], source=source)
        all_queries.extend(queries)
        all_passages.extend(passages)
    expected = sum(counts.values())
    assert len(all_queries) == expected, f"multihop mix must be {expected} queries"
    return all_queries, all_passages
