"""Small deterministic fixtures for tests and smoke runs.

Synthetic port records follow `natlang.port-record/1` (S1 §2.1). Each source is a short
project card; questions ask for one attribute, so a correct block helps and a shuffled
one (another card) does not. Text fixtures come from a local raw-text file when one is
available, else from generated sentences.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .records import VERSION

NAMES = ["Alder", "Birch", "Cedar", "Dogwood", "Elm", "Fir", "Ginkgo", "Hazel", "Ironwood", "Juniper",
         "Kapok", "Larch", "Maple", "Nutmeg", "Olive", "Pine", "Quince", "Rowan", "Spruce", "Teak"]
LEADS = ["Amara", "Bruno", "Chen", "Dalia", "Emeka", "Farah", "Goran", "Hana", "Ivo", "Jun"]
CITIES = ["Lisbon", "Osaka", "Quito", "Tallinn", "Accra", "Hobart", "Bergen", "Cusco", "Leeds", "Perth"]
COLOURS = ["amber", "cobalt", "crimson", "jade", "ochre", "violet", "silver", "teal"]
ATTRIBUTES = {
    "lead": "Who leads project {name}?",
    "city": "In which city is project {name} based?",
    "colour": "What is the badge colour of project {name}?",
}

DEFAULT_TEXT = Path.home() / "bgkit-data-nvme/bgkit2/filler/wikitext.jsonl"


def synthetic_records(count: int = 64, seed: int = 0, heldout_every: int = 4) -> list[dict]:
    rng = random.Random(seed)
    records = []
    for i in range(count):
        name = f"{NAMES[i % len(NAMES)]}-{i // len(NAMES) + 1}"
        card = {"lead": rng.choice(LEADS), "city": rng.choice(CITIES), "colour": rng.choice(COLOURS)}
        source = (f"Project {name} is led by {card['lead']}. Its team is based in {card['city']}. "
                  f"The project's badge colour is {card['colour']}.")
        attribute = list(ATTRIBUTES)[i % len(ATTRIBUTES)]
        split = "heldout" if i % heldout_every == heldout_every - 1 else "train"
        records.append({
            "version": VERSION,
            "id": f"fixture:projects:{i}",
            "family": "fixture_projects",
            "task": "consume",
            "sources": [{"role": "document", "text": source, "exact_refs": [name]}],
            "writer": {
                "instructions": "Read the project card so that later questions about the project can be answered.",
                "result_type": "Neuralese<ProjectCard>",
                "context": [],
            },
            "consumer": {
                "context": [{"role": "user", "content": ATTRIBUTES[attribute].format(name=name)}],
                "withheld": ["sources"],
            },
            "target": {"kind": "text", "value": card[attribute], "alternatives": []},
            "contrasts": {"purpose_pairs": [], "distractors": []},
            "outcome": {"label": "gold", "checked": "constructed"},
            "lineage": {"project": "fixture", "store": "projects", "row": i},
            "license": {"spdx": "CC0-1.0", "noncommercial": False, "notes": "synthetic"},
            "split": split,
            "split_groups": [f"fixture:project:{name}"],
        })
    return records


def write_jsonl(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def fixture_texts(limit: int = 200, path: str | Path | None = None, min_chars: int = 400) -> list[str]:
    path = Path(path) if path else DEFAULT_TEXT
    texts = []
    if path.exists():
        with open(path) as handle:
            for line in handle:
                text = json.loads(line).get("text", "")
                if len(text) >= min_chars:
                    texts.append(text)
                if len(texts) >= limit:
                    break
    if texts:
        return texts
    rng = random.Random(0)
    return [" ".join(f"The {rng.choice(COLOURS)} survey from {rng.choice(CITIES)} was reviewed by {rng.choice(LEADS)}."
                     for _ in range(12)) for _ in range(limit)]
