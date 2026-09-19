"""Pilot corpus loaders for the Week-1 costsmart-rag pilot.

Covers three Tier-A datasets: Natural Questions (NQ), TriviaQA, and HotpotQA.
Pilot size is 200 queries: 80 NQ + 70 TriviaQA + 50 HotpotQA.

Two modes:
  * ``source="hf"``  -- load the real datasets via HuggingFace ``datasets``
    (requires the ``datasets`` package + network). Used for the real pilot.
  * ``source="synthetic"`` (default) -- build a deterministic, seeded offline
    pilot with the same query-id scheme and sizes, so the index build and
    smoke retrieval run anywhere without network access.

Query-id scheme (stable across modes): ``{dataset}:{index:04d}``,
e.g. ``nq:0007``, ``triviaqa:0042``, ``hotpotqa:0013``.

Each query record: ``query_id``, ``source`` (nq|triviaqa|hotpotqa),
``question``, ``answers`` (list[str]), ``gold_passage_ids`` (list[str]).
Each passage record: ``passage_id``, ``doc_id``, ``text``.
"""

from __future__ import annotations

import hashlib
import random

# Pilot composition: 200 queries total across the three Tier-A datasets.
PILOT_SIZES = {
    "nq": 80,
    "triviaqa": 70,
    "hotpotqa": 50,
}

PILOT_TOTAL = sum(PILOT_SIZES.values())

HF_DATASET_NAMES = {
    "nq": "nq_open",
    "triviaqa": "trivia_qa",
    "hotpotqa": "hotpot_qa",
}

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
        if dataset == "hotpotqa":
            (ta, fa), (tb, fb) = rng.sample(_TOPICS, 2)
            tmpl = rng.choice(_QUESTION_TEMPLATES[dataset])
            question = tmpl.format(topic_a=ta, topic_b=tb)
            answer = ta
            gold_texts = [
                f"{fa} Further context on {ta} relates it to {tb}.",
                f"{fb} Scholars link {tb} back to developments around {ta}.",
            ]
        else:
            topic, fact = rng.choice(_TOPICS)
            tmpl = rng.choice(_QUESTION_TEMPLATES[dataset])
            question = tmpl.format(topic=topic, fact=fact)
            answer = topic
            gold_texts = [f"{fact} Additional detail: {topic} is widely documented."]
        gold_ids = []
        for g, text in enumerate(gold_texts):
            pid = f"{qid}#p{g}"
            gold_ids.append(pid)
            passages.append({"passage_id": pid, "doc_id": qid, "text": text})
        # One distractor passage per query so retrieval is non-trivial.
        d_topic, d_fact = rng.choice(_TOPICS)
        passages.append(
            {
                "passage_id": f"{qid}#d0",
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


def _hf_queries(dataset: str, n: int):
    """Load real queries via HuggingFace ``datasets`` (requires network)."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "HuggingFace 'datasets' package not installed; "
            "use source='synthetic' for the offline pilot."
        ) from exc
    name = HF_DATASET_NAMES[dataset]
    ds = load_dataset(name, split="validation")
    queries: list[dict] = []
    passages: list[dict] = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        qid = f"{dataset}:{i:04d}"
        if dataset == "nq":
            question = row["question"]
            answers = list(row.get("answer", []))
        elif dataset == "triviaqa":
            question = row["question"]
            answers = [row["answer"]["value"]] if row.get("answer") else []
        else:  # hotpotqa
            question = row["question"]
            answers = [row["answer"]] if row.get("answer") else []
        gold_ids = [f"{qid}#p0"]
        ctx = str(row.get("context", row.get("question", "")))[:2000]
        passages.append({"passage_id": gold_ids[0], "doc_id": qid, "text": ctx})
        queries.append(
            {
                "query_id": qid,
                "source": dataset,
                "question": question,
                "answers": answers,
                "gold_passage_ids": gold_ids,
            }
        )
    return queries, passages


def load_dataset_queries(
    dataset: str, n: int | None = None, source: str = "synthetic"
) -> tuple[list[dict], list[dict]]:
    """Load (queries, passages) for one dataset.

    Returns a ``(queries, passages)`` tuple. ``n`` defaults to the pilot
    size for that dataset from :data:`PILOT_SIZES`.
    """
    if dataset not in PILOT_SIZES:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {sorted(PILOT_SIZES)}")
    n = PILOT_SIZES[dataset] if n is None else n
    if source == "hf":
        return _hf_queries(dataset, n)
    if source == "synthetic":
        return _synthetic_queries(dataset, n)
    raise ValueError(f"unknown source {source!r}; expected 'hf' or 'synthetic'")


def load_pilot_subset(source: str = "synthetic") -> tuple[list[dict], list[dict]]:
    """Load the full 200-query pilot across NQ + TriviaQA + HotpotQA."""
    all_queries: list[dict] = []
    all_passages: list[dict] = []
    for dataset in ("nq", "triviaqa", "hotpotqa"):
        queries, passages = load_dataset_queries(dataset, source=source)
        all_queries.extend(queries)
        all_passages.extend(passages)
    assert len(all_queries) == PILOT_TOTAL, f"pilot must be {PILOT_TOTAL} queries"
    return all_queries, all_passages
