"""Port records for the `view`/`ask` summarizer family (plans/neuralese/VIEW_CORPUS.md; DECISIONS 2026-10-09).

Every artifact type the harness hands to `view` (prose, code, HTML, JSON/YAML, tables, logs and tool outputs) gets
records with static targets only:

- `reconstruct`: `view(x)` without instructions, read back; the target is the source itself.
- `consume` extraction: a code-computed target (JSONPath values, keys and lengths; CSS-selector text; table cells,
  row counts and arg-max rows; function parameters and called names; exit codes, failing tests, exceptions).
- `consume` dataset QA and purpose-conditioned summaries: questions, answers and summaries taken from public
  datasets, never generated. `writer.instructions` is the dataset's query or instruction (generic for plain
  summaries); `writer.instructions_general` is the purpose-free form.
- `compare`: when one source carries several purposes (questions, extraction requests), each of its records becomes
  `compare` and lists the others in `contrasts.purpose_pairs`.

Conventions specific to this corpus:
- `family` is `view_<artifact>_<kind>`, kind in reconstruct / summary / qa / extract; `sources[].meta.artifact`
  repeats the artifact type; `lineage.notes` records the artifact, the `view` call shape and the target origin.
- `writer.result_type` is `Neuralese<string>`: `view` is representation-generic over a string result.
- `exact_refs` mark identifiers, paths and URLs that a view must keep verbatim. A record's own extraction target is
  never listed as an exact ref (a renderer shows exact refs beside the block, which would hand over the answer);
  reconstruct records keep the full list, which is the identifier-retention check over the reconstruction.
- Splits: upstream splits are kept (test and validation never move to train); sources with only a train split get a
  deterministic 5% test / 5% validation carve by split group. Groups are then closed transitively together with
  the protected bgkit benchmark index (splits.close), so a protected hit pulls its component into test.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import io
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from . import view_extract as vx
from .common import Reject
from .records import VERSION, exact_refs_from, group_key, leakage, license_, seal, source, text_hash, validate_with_schema
from . import cross_corpus, cross_corpus_registry
from .splits import check_closed

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training" / "neuralese"))
from natlang_neuralese.common.paths import resolve  # noqa: E402
from natlang_neuralese.data.records import RecordError, parse_record  # noqa: E402

CONVERTER = "scripts/neuralese_data/view_corpus.py@2"
RESULT_TYPE = "Neuralese<string>"
MAX_SOURCE_CHARS = 32_000
MIN_SOURCE_CHARS = 200
PROTECTED_INDEX = resolve("data_hdd", "natlang-development-data/data/neuralese/protected/bgkit-benchmarks.protected.json")
HF = Path.home() / ".cache/huggingface/hub"
NEBIUS = resolve("repo", "data/neuralese/corpora/nebius-swe-rebench-openhands-trajectories-20261009-v1/trajectories.parquet")
XLAM = resolve("archive", "raw/agentic-20260927/xlam-function-calling-60k/xlam_function_calling_60k.json")
REPOS = resolve("data_hdd", "bgkit-data/repos")
AGENTIC = resolve("archive", "raw/agentic-20260927")
SPIDER = AGENTIC / "spider/official/spider_data"
BIRD = {"train": (AGENTIC / "bird/train/train.json", resolve("data_nvme", "bird-sqlite/train/train_databases")),
        "validation": (AGENTIC / "bird/dev_20240627/dev.json", AGENTIC / "bird/dev_20240627/dev_databases")}
TOOLACE = AGENTIC / "toolace/data.json"
S1 = resolve("repo", "data/neuralese/corpora/s1-full-final-20261003")
CROSS_INDEXES = [resolve("data_nvme", "natlang-corpora/cross-corpus-index/s1-full-final-20261003"),
                 resolve("data_nvme", "natlang-corpora/cross-corpus-index/harness-bench-swe-rebench-openhands-pi-records-20261010-v3")]

# Documents (or functions, pages, tables, files, tool outputs) per source in the default balanced slice.
DEFAULT_CAPS = {
    "cnn_dailymail": 500, "squad": 220, "quality": 40, "qmsum": 160,
    "codesearchnet": 900,
    "websrc": 700,
    "xlam": 700, "repo_configs": 700,
    "wtq": 450, "fetaqa": 700,
    "swe_tool_outputs": 1100,
}
# Sources added in v2 (absent from DEFAULT_CAPS means off, so a v1 rebuild stays a v1 rebuild).
V2_CAPS = {
    # SQuAD and QuALITY: their questions are already in S1 (exact examples there are dropped, SQuAD is S1 test).
    "cnn_dailymail": 4000, "squad": 400, "quality": 300, "qmsum": 400,
    "codesearchnet": 5000, "codesearchnet_go": 1500, "codesearchnet_java": 1500, "codesearchnet_javascript": 1500,
    "codesearchnet_php": 1500, "codesearchnet_ruby": 1500,
    "websrc": 2000,
    "xlam": 3000, "repo_configs": 1500, "toolace": 2000,
    "wtq": 2000, "fetaqa": 2500, "spider_bird": 2500, "tabfact": 2000,
    "swe_tool_outputs": 3500, "s1_tool_outputs": 3000,
}
QUESTIONS_PER_SOURCE = 4
EXTRACTS_PER_SOURCE = 3
SPLIT_MAP = {"train": "train", "validation": "validation", "valid": "validation", "val": "validation", "dev": "validation",
             "test": "test"}
UPSTREAM_SHARE = {"train": 0.9, "validation": 0.05, "test": 0.05}

NOUNS = {"prose": "document", "code": "function", "html": "web page", "data": "file", "table": "table", "log": "tool output"}
ROLES = {"prose": "document", "code": "file", "html": "document", "data": "file", "table": "document", "log": "tool_output"}


# ---------------------------------------------------------------------------------------------- plumbing

def _h(*parts) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()


def carve(group: str, salt: str = "view-corpus") -> str:
    """Deterministic 5% test / 5% validation carve for sources with only a train split."""
    b = int(_h(salt, group)[:8], 16) % 100
    return "test" if b < 5 else "validation" if b < 10 else "train"


def _safe(key) -> str:
    return re.sub(r"[^A-Za-z0-9_.:#@/+-]+", "_", str(key))


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def provenance(dataset_spdx: str, dataset_source: str, *, content_spdx: str | None = None,
               content_source: str | None = None, holder: str | None = None, content_unverified: bool = False,
               concerns=(), license_class: str | None = None) -> dict:
    """Per-record licence provenance (`lineage.notes.license_provenance`), so a licence review can filter records:
    the dataset's licence and where it was read, the licence of the underlying content (repository, website, news
    publisher) when it differs or is known, a review class (the most restrictive of the known licences, `unverified`
    when the content's licence is unknown and substantial), and short concerns."""
    # A dataset whose licence is "per repository" has no class of its own: its content's licence decides.
    per_content = dataset_spdx == "LicenseRef-repository-content"
    classes = [] if per_content else [vx.license_class(dataset_spdx)]
    if content_spdx:
        classes.append(vx.license_class(content_spdx))
    if (content_unverified or per_content) and not content_spdx:
        classes.append("unverified")
    cls = license_class or max(classes, key=vx.CLASS_ORDER.index)
    return {"dataset": {"spdx": dataset_spdx, "source": dataset_source},
            "content": {"spdx": content_spdx, "source": content_source, "holder": holder},
            "class": cls, "concerns": list(concerns)}


class Ctx:
    def __init__(self, raw: Path, caps: dict, max_chars: int, min_chars: int, seed: int):
        self.raw, self.caps, self.max_chars, self.min_chars, self.seed = raw, caps, max_chars, min_chars, seed
        self.rejected: Counter = Counter()
        self.reject_examples: dict = {}
        self.inputs: dict[str, dict] = {}
        self.info: dict = {}

    def rng(self, *key) -> random.Random:
        return random.Random(_h(self.seed, *key))

    def reject(self, dataset: str, row, reason: str):
        key = f"{dataset}:{reason.split(':')[0]}"
        self.rejected[key] += 1
        self.reject_examples.setdefault(key, f"{dataset}#{row}: {reason}")

    def use(self, path: Path, **info):
        """Record an input file (hashed once in the manifest)."""
        self.inputs.setdefault(str(path), info)

    def fits(self, text: str) -> bool:
        return self.min_chars <= len(text) <= self.max_chars


class Doc:
    """One source artifact and the records written over it."""

    def __init__(self, ctx: Ctx, *, dataset: str, key, artifact: str, text: str, title=None, meta=None, refs=None,
                 upstream: str, upstream_id=None, revision=None, store_version=None, row=None, lic: dict, split: str,
                 groups: list[str], source_role: str | None = None, notes=None, prov: dict | None = None):
        self.ctx, self.dataset, self.key, self.artifact, self.text = ctx, dataset, _safe(key), artifact, text
        self.title, self.meta, self.refs = title, dict(meta or {}), list(refs or [])
        self.upstream, self.upstream_id, self.revision, self.store_version = upstream, upstream_id, revision, store_version
        self.row, self.lic, self.split, self.groups = row if row is not None else str(key), lic, split, groups
        self.role = source_role or ROLES[artifact]
        self.notes = dict(notes or {})
        self.notes["license_provenance"] = prov or provenance(lic["spdx"], lic.get("notes", ""))
        self.records: list[dict] = []

    def _source(self, drop: str | None = None) -> dict:
        refs = [r for r in self.refs if not (drop and r["text"] in drop)]
        return source(self.role, self.text, title=self.title, meta={"artifact": self.artifact, **self.meta},
                      exact_refs=refs)

    def _record(self, kind: str, task: str, n: int, *, instructions: str, general: str | None, request: str, target,
                target_kind: str, alternatives, checked: str, origin: str, view_call: str, extra_notes=None) -> dict:
        target_text = target if isinstance(target, str) else json.dumps(target, ensure_ascii=False)
        writer = {"instructions": instructions, "result_type": RESULT_TYPE, "context": []}
        if general:
            writer["instructions_general"] = general
        record = {
            "version": VERSION,
            "id": f"view:{self.dataset}:{self.key}:{kind}:{n}",
            "family": f"view_{self.artifact}_{kind}",
            "task": task,
            "sources": [self._source(drop=None if kind == "reconstruct" else target_text)],
            "writer": writer,
            "consumer": {"context": [{"role": "user", "content": request}], "withheld": ["sources"]},
            "target": {"kind": target_kind, "value": target, "alternatives": list(alternatives)},
            "contrasts": {"purpose_pairs": [], "distractors": []},
            "outcome": {"label": "gold", "checked": checked,
                        "details": {"checker_evidence": {"origin": origin, "local_execution": origin == "code-computed"}}},
            "lineage": {"project": "upstream", "store": self.dataset, "store_version": self.store_version, "row": self.row,
                        "upstream": self.upstream, "upstream_id": self.upstream_id, "upstream_revision": self.revision,
                        "teacher": None, "converter": CONVERTER, "sha256": "",
                        "notes": {"artifact": self.artifact, "view_call": view_call, "target_origin": origin,
                                  **self.notes, **(extra_notes or {})}},
            "license": self.lic,
            "split": self.split,
            "split_groups": list(self.groups),
        }
        self.records.append(record)
        return record

    def reconstruct(self):
        noun = NOUNS[self.artifact]
        return self._record("reconstruct", "reconstruct", 0,
                            instructions=f"Keep the whole {noun}, so that it can be reproduced exactly.", general=None,
                            request=f"Reproduce the {noun} exactly.", target=self.text, target_kind="text",
                            alternatives=(), checked="identity", origin="identity", view_call="view(x)")

    def summary(self, instruction: str, target: str, *, origin: str = "dataset-summary", n: int = 0, alternatives=(),
                general: str | None = None):
        return self._record("summary", "consume", n, instructions=instruction, general=general, request=instruction,
                            target=target, target_kind="text", alternatives=alternatives, checked="reference-summary",
                            origin=origin, view_call="view(x, instructions)")

    def qa(self, question: str, answer, n: int, *, request: str | None = None, target_kind: str = "text",
           alternatives=(), checked: str = "reference-answer", origin: str = "dataset-qa", extra_notes=None):
        noun = NOUNS[self.artifact]
        return self._record("qa", "consume", n, instructions=f"Keep what is needed to answer this question: {question}",
                            general=f"Keep what later questions about the {noun} will need.", request=request or question,
                            target=answer, target_kind=target_kind, alternatives=alternatives, checked=checked,
                            origin=origin, view_call="ask(view(x), question)",
                            extra_notes={"question": question, **(extra_notes or {})})

    def extract(self, request: str, answer: str, n: int, *, method: str):
        noun = NOUNS[self.artifact]
        return self._record("extract", "consume", n, instructions=f"Keep what is needed for this request: {request}",
                            general=f"Keep the {noun}'s exact values, names and structure.", request=request,
                            target=answer, target_kind="text", alternatives=(), checked=f"code-computed:{method}",
                            origin="code-computed", view_call="ask(view(x), request)", extra_notes={"extractor": method})


def _upstream_sample(ctx: Ctx, dataset: str, sizes: dict[str, int], cap: int) -> dict[str, list[int]]:
    """Row indices per upstream split: the cap is divided train/validation/test 90/5/5, seeded."""
    out = {}
    for split, n in sizes.items():
        want = max(1, round(cap * UPSTREAM_SHARE[SPLIT_MAP[split]])) if n else 0
        idx = list(range(n))
        ctx.rng(dataset, split).shuffle(idx)
        out[split] = idx[: min(n, want * 3)]  # over-draw: some rows are rejected (length, format)
    return out


def _read_rows(path: Path, indices: list[int], columns=None) -> list[dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(str(path), columns=columns)
    return [table.slice(i, 1).to_pylist()[0] | {"__row": i} for i in indices]


# ---------------------------------------------------------------------------------------------- prose

def cnn_dailymail(ctx: Ctx):
    snap = sorted(glob.glob(str(HF / "datasets--cnn_dailymail/snapshots/*/3.0.0")))
    if not snap:
        raise FileNotFoundError("cnn_dailymail 3.0.0 not in the Hugging Face cache")
    root = Path(snap[-1])
    revision = root.parent.name
    files = {"train": root / "train-00000-of-00003.parquet", "validation": root / "validation-00000-of-00001.parquet",
             "test": root / "test-00000-of-00001.parquet"}
    import pyarrow.parquet as pq

    sizes = {s: pq.ParquetFile(str(p)).metadata.num_rows for s, p in files.items()}
    picks = _upstream_sample(ctx, "cnn_dailymail", sizes, ctx.caps["cnn_dailymail"])
    lic = license_("Apache-2.0", False, "abisee/cnn_dailymail card licence; article text from CNN and the Daily Mail.")
    prov = provenance("Apache-2.0", "hf:abisee/cnn_dailymail dataset card", content_source="news articles and highlights",
                      holder="CNN / Daily Mail", content_unverified=True,
                      concerns=["article and highlight text is the publishers' copyright; the card's Apache-2.0 covers the dataset release"])
    for split, path in files.items():
        ctx.use(path, dataset="cnn_dailymail", upstream="hf:abisee/cnn_dailymail", revision=revision)
        want = max(1, round(ctx.caps["cnn_dailymail"] * UPSTREAM_SHARE[split]))
        made = 0
        for row in _read_rows(path, picks[split]):
            if made >= want:
                break
            article = row["article"].strip()
            highlights = re.sub(r"\s+\.(?=\s|$)", ".", row["highlights"]).strip()
            if not ctx.fits(article):
                ctx.reject("cnn_dailymail", row["id"], "length")
                continue
            doc = Doc(ctx, dataset="cnn_dailymail", key=row["id"], artifact="prose", text=article, upstream="hf:abisee/cnn_dailymail",
                      upstream_id=row["id"], revision=revision, store_version="3.0.0", row=row["__row"], lic=lic, prov=prov,
                      split=SPLIT_MAP[split], groups=[group_key("cnn_dailymail", row["id"])], meta={"format": "news article"})
            doc.summary("Write the story highlights of this news article: a few short sentences with its main facts.",
                        highlights, general="Keep what a short summary of the article needs.")
            doc.reconstruct()
            made += 1
            yield doc


def squad(ctx: Ctx):
    """SQuAD v1.1 train paragraphs (validation is a protected benchmark); several questions per paragraph."""
    snap = sorted(glob.glob(str(HF / "datasets--rajpurkar--squad/snapshots/*/plain_text/train-00000-of-00001.parquet")))
    path = Path(snap[-1])
    revision = path.parent.parent.name
    ctx.use(path, dataset="squad", upstream="hf:rajpurkar/squad", revision=revision)
    import pyarrow.parquet as pq

    table = pq.read_table(str(path), columns=["id", "title", "context", "question", "answers"]).to_pylist()
    by_context: dict[str, list] = defaultdict(list)
    for i, r in enumerate(table):
        by_context[r["context"]].append((i, r))
    contexts = sorted(by_context, key=lambda c: _h("squad", c))
    lic = license_("CC-BY-SA-4.0", False, "SQuAD v1.1 (rajpurkar/squad); Wikipedia text.")
    prov = provenance("CC-BY-SA-4.0", "hf:rajpurkar/squad dataset card", content_spdx="CC-BY-SA-4.0",
                      content_source="Wikipedia", holder="Wikipedia contributors")
    for context in contexts[: ctx.caps["squad"]]:
        rows = by_context[context]
        first = rows[0][1]
        title = first["title"]
        doc = Doc(ctx, dataset="squad", key=first["id"], artifact="prose", text=context, title=title.replace("_", " "),
                  upstream="hf:rajpurkar/squad", upstream_id=first["id"], revision=revision, row=rows[0][0], lic=lic, prov=prov,
                  split=carve(group_key("wiki", title)), groups=[group_key("wiki", title)], meta={"format": "encyclopedia paragraph"})
        if len(context) < ctx.min_chars:
            ctx.reject("squad", first["id"], "length")
            continue
        chosen = sorted(rows, key=lambda x: _h(ctx.seed, x[1]["id"]))[:QUESTIONS_PER_SOURCE]
        for n, (i, r) in enumerate(chosen):
            answers = list(dict.fromkeys(r["answers"]["text"]))
            doc.qa(r["question"].strip(), answers[0], n, alternatives=answers[1:], extra_notes={"upstream_question_id": r["id"]})
        doc.reconstruct()
        yield doc


def quality(ctx: Ctx):
    """QuALITY multiple-choice questions over long articles (train and validation as published)."""
    snaps = sorted(glob.glob(str(HF / "datasets--tasksource--quality/snapshots/*/data")))
    root = Path(snaps[-1])
    revision = root.parent.name
    import pyarrow.parquet as pq

    lic = license_("LicenseRef-QuALITY", False, "QuALITY annotations (nyu-mll/quality via tasksource/quality, licence to "
                   "be confirmed); article licence per row (mostly Project Gutenberg).")
    for split in ("train", "validation"):
        path = next(root.glob(f"{split}-*.parquet"))
        ctx.use(path, dataset="quality", upstream="hf:tasksource/quality", revision=revision)
        rows = pq.read_table(str(path)).to_pylist()
        by_article: dict[str, list] = defaultdict(list)
        for i, r in enumerate(rows):
            by_article[r["article_id"]].append((i, r))
        want = max(1, round(ctx.caps["quality"] * UPSTREAM_SHARE[split]))
        made = 0
        for aid in sorted(by_article, key=lambda a: _h("quality", a)):
            if made >= want:
                break
            qs = by_article[aid]
            article = qs[0][1]["article"].strip()
            if not ctx.fits(article):
                ctx.reject("quality", aid, "length")
                continue
            doc = Doc(ctx, dataset="quality", key=aid, artifact="prose", text=article, title=qs[0][1]["title"],
                      upstream="hf:tasksource/quality", upstream_id=str(aid), revision=revision, row=qs[0][0], lic=lic,
                      split=SPLIT_MAP[split], groups=[group_key("quality-doc", aid)], meta={"format": "long article"},
                      notes={"article_license": (qs[0][1].get("license") or "")[:200]},
                      prov=provenance("LicenseRef-QuALITY", "no licence on nyu-mll/quality or the tasksource/quality card",
                                      content_source="article licence field: " + (qs[0][1].get("license") or "none")[:160],
                                      holder="Project Gutenberg / source publication",
                                      concerns=["QuALITY annotation licence not stated"]))
            seen = set()
            n = 0
            for i, r in sorted(qs, key=lambda x: _h(ctx.seed, x[1]["question_unique_id"])):
                if r["question"] in seen or n >= QUESTIONS_PER_SOURCE:
                    continue
                seen.add(r["question"])
                options = r["options"] if isinstance(r["options"], list) else _literal_list(r["options"])
                gold = int(r["gold_label"]) - 1
                if not options or not 0 <= gold < len(options):
                    ctx.reject("quality", r["question_unique_id"], "bad-options")
                    continue
                letters = "ABCDEFGH"
                request = r["question"].strip() + "\nOptions:\n" + "\n".join(f"({letters[k]}) {o}" for k, o in enumerate(options)) \
                    + "\nAnswer with the letter of the correct option."
                doc.qa(r["question"].strip(), letters[gold], n, request=request, target_kind="choice",
                       extra_notes={"upstream_question_id": r["question_unique_id"]})
                n += 1
            doc.reconstruct()
            made += 1
            yield doc


def _literal_list(text: str) -> list:
    import ast
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    return [str(v) for v in value] if isinstance(value, (list, tuple)) else []


def _turns(meeting: dict) -> list[str]:
    return [f"{t.get('speaker', '').strip()}: {re.sub(r'\s+', ' ', t.get('content', '')).strip()}" for t in meeting["meeting_transcripts"]]


def _span(spans) -> tuple[int, int] | None:
    pairs = [(int(a), int(b)) for a, b in spans if str(a).isdigit() and str(b).isdigit()]
    if not pairs:
        return None
    return min(a for a, _ in pairs), max(b for _, b in pairs)


def qmsum_windows(turns: list[str], spans: list[tuple[int, int]], cap: int) -> list[tuple[int, int, list[int]]]:
    """Excerpts of a long meeting for its specific queries: queries are packed in order of their relevant turns into
    windows that fit `cap` characters (several queries over one window become purposes over one source), then each
    window is padded with neighbouring turns, alternately before and after, to about three times its relevant text
    (at least 6,000 characters), so that an excerpt is never just the answer's span."""
    def size(a, b):
        return sum(len(t) + 1 for t in turns[a: b + 1])

    order = sorted(range(len(spans)), key=lambda i: spans[i])
    windows: list[list] = []
    for i in order:
        a, b = spans[i]
        if b >= len(turns) or size(a, b) > cap:
            continue
        if windows and size(min(windows[-1][0], a), max(windows[-1][1], b)) <= cap * 0.6:
            w = windows[-1]
            w[0], w[1] = min(w[0], a), max(w[1], b)
            w[2].append(i)
        else:
            windows.append([a, b, [i]])
    out = []
    for a, b, members in windows:
        goal = min(cap, max(6000, 3 * size(a, b)))
        left = True
        while size(a, b) < goal and (a > 0 or b < len(turns) - 1):
            if (left and a > 0) or b >= len(turns) - 1:
                if size(a - 1, b) > cap:
                    break
                a -= 1
            else:
                if size(a, b + 1) > cap:
                    break
                b += 1
            left = not left
        same = next((w for w in out if w[:2] == (a, b)), None)
        if same:  # padding made two windows the same excerpt: one source, more purposes
            same[2].extend(members)
        else:
            out.append((a, b, list(members)))
    return out


def qmsum(ctx: Ctx):
    """QMSum query-based meeting summaries. General queries use the whole transcript when it fits; specific queries
    use excerpts around their relevant turns (`qmsum_windows`)."""
    spec = "qmsum"
    meta = _raw_meta(ctx, spec)
    lic = license_("MIT", False, "QMSum (Yale-LILY/QMSum, MIT); transcripts from AMI and ICSI (CC-BY-4.0) and "
                   "Welsh/Canadian parliament records (open government licences).")
    prov = provenance("MIT", "github:Yale-LILY/QMSum LICENSE", content_spdx="CC-BY-4.0",
                      content_source="AMI and ICSI corpora (CC-BY-4.0); Welsh Parliament and Parliament of Canada records "
                                     "(open government licences)", holder="AMI/ICSI; parliaments",
                      concerns=["committee transcripts are under open government licences, not CC"])
    for split_file, split in (("train", "train"), ("val", "validation"), ("test", "test")):
        path = ctx.raw / "qmsum/data/ALL/jsonl" / f"{split_file}.jsonl"
        ctx.use(path, dataset="qmsum", upstream="github:Yale-LILY/QMSum", revision=meta["revision"])
        meetings = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        want = max(1, round(ctx.caps["qmsum"] * UPSTREAM_SHARE[split]))
        order = sorted(range(len(meetings)), key=lambda i: _h("qmsum", split, i))[:want]
        for mi in order:
            m = meetings[mi]
            turns = _turns(m)
            mkey = f"{split_file}-{mi}"
            group = group_key("qmsum-meeting", _h(turns[0] if turns else mi, len(turns))[:16])
            base = dict(upstream="github:Yale-LILY/QMSum", revision=meta["revision"], lic=lic, split=split, groups=[group],
                        store_version="ALL", prov=prov)
            whole = "\n".join(turns)
            if ctx.fits(whole) and m.get("general_query_list"):
                doc = Doc(ctx, dataset="qmsum", key=f"{mkey}:all", artifact="prose", text=whole, title="meeting transcript",
                          upstream_id=mkey, row=mi, meta={"format": "meeting transcript"}, **base)
                for n, q in enumerate(m["general_query_list"][:QUESTIONS_PER_SOURCE]):
                    doc.summary(q["query"].strip(), q["answer"].strip(), n=n, general="Keep what a summary of the meeting needs.")
                yield doc
            elif m.get("general_query_list"):
                ctx.reject("qmsum", mkey, "length: general query needs the whole transcript")
            specific = [q for q in m.get("specific_query_list") or [] if _span(q.get("relevant_text_span") or [])]
            spans = [_span(q["relevant_text_span"]) for q in specific]
            placed = set()
            for a, b, members in qmsum_windows(turns, spans, ctx.max_chars):
                text = "\n".join(turns[a: b + 1])
                if not ctx.fits(text):
                    continue
                placed.update(members[:QUESTIONS_PER_SOURCE])
                doc = Doc(ctx, dataset="qmsum", key=f"{mkey}:turns{a}-{b}", artifact="prose", text=text,
                          title="meeting transcript excerpt", upstream_id=mkey, row=mi,
                          meta={"format": "meeting transcript excerpt", "turns": [a, b]}, **base)
                for n, i in enumerate(members[:QUESTIONS_PER_SOURCE]):
                    doc.summary(specific[i]["query"].strip(), specific[i]["answer"].strip(), n=n,
                                general="Keep what a summary of this part of the meeting needs.")
                yield doc
            for i in range(len(specific)):
                if i not in placed:
                    ctx.reject("qmsum", f"{mkey}:q{i}", "unplaced: relevant turns exceed the source cap, or the excerpt already has QUESTIONS_PER_SOURCE queries")


# ---------------------------------------------------------------------------------------------- code

CSN_LANGS = ("python", "go", "java", "javascript", "php", "ruby")
_DOC_COMMENT = {
    "java": re.compile(r"/\*\*.*?\*/\s*", re.S), "javascript": re.compile(r"/\*\*.*?\*/\s*", re.S),
    "php": re.compile(r"/\*\*.*?\*/\s*", re.S), "go": re.compile(r"(?:^[ \t]*//[^\n]*\n)+", re.M),
    "ruby": re.compile(r"(?:^[ \t]*#[^\n]*\n)+", re.M),
}


def _csn_cap(ctx: Ctx, lang: str) -> int:
    return ctx.caps.get("codesearchnet" if lang == "python" else f"codesearchnet_{lang}", 0)


def _csn_licenses(ctx: Ctx, lang: str) -> dict:
    path = ctx.raw / "codesearchnet" / f"{lang}-licenses.json"
    if not path.exists():
        return {}
    ctx.use(path, dataset="codesearchnet-licenses", upstream="hf:code-search-net/code_search_net", revision="fdc6a9e")
    return json.loads(path.read_text())


def _strip_doc_comment(lang: str, code: str, doc: str) -> str | None:
    """The function without its documentation comment; None if the docstring's text would remain."""
    pattern = _DOC_COMMENT.get(lang)
    probe = " ".join(doc.split()[:8])
    out = code
    if pattern and probe and probe in " ".join(code.split()):
        out = pattern.sub("", code, count=1)
    return None if probe and probe in " ".join(out.split()) else out


def codesearchnet(ctx: Ctx):
    """CodeSearchNet functions. Python: docstring summary (docstring removed from the source), parameters, called
    names, and reconstruction of the whole function. Other languages (v2): docstring summary (documentation comment
    removed) and reconstruction. Licences per repository from the original release's licence files, when fetched."""
    meta = _raw_meta(ctx, "codesearchnet")
    import pyarrow.parquet as pq

    for lang in CSN_LANGS:
        cap = _csn_cap(ctx, lang)
        if not cap or not (ctx.raw / "codesearchnet" / f"{lang}-train.parquet").exists():
            continue
        licenses = _csn_licenses(ctx, lang)
        sizes, paths = {}, {}
        for split in ("train", "validation", "test"):
            p = ctx.raw / "codesearchnet" / f"{lang}-{split}.parquet"
            ctx.use(p, dataset="codesearchnet", upstream="hf:code-search-net/code_search_net", revision=meta["revision"])
            paths[split], sizes[split] = p, pq.ParquetFile(str(p)).metadata.num_rows
        picks = _upstream_sample(ctx, "codesearchnet" if lang == "python" else f"codesearchnet-{lang}", sizes, cap)
        cols = ["repository_name", "func_path_in_repository", "func_name", "whole_func_string", "func_code_url"]
        if lang != "python":
            cols.append("func_documentation_string")
        dataset = "codesearchnet" if lang == "python" else f"codesearchnet_{lang}"
        seen_keys: set = set()
        for split, path in paths.items():
            want = max(1, round(cap * UPSTREAM_SHARE[split]))
            made = 0
            for row in _read_rows(path, picks[split], cols):
                if made >= want:
                    break
                repo = row["repository_name"]
                key = f"{repo}/{row['func_path_in_repository']}:{row['func_name']}"
                if key in seen_keys:  # same-named methods in one file (Go receivers, overloads): the line anchor
                    anchor = (row.get("func_code_url") or "").rpartition("#")[2]
                    key = f"{key}@{anchor or row['__row']}"
                    if key in seen_keys:
                        key = f"{key}@{split}{row['__row']}"
                seen_keys.add(key)
                whole = row["whole_func_string"]
                if lang == "python":
                    facts = vx.python_facts(whole)
                    if facts is None or not facts["docstring"]:
                        ctx.reject(dataset, key, "no-docstring-or-parse")
                        continue
                    doc_text = facts["docstring"]
                else:
                    facts, doc_text = None, (row.get("func_documentation_string") or "").strip()
                    if not doc_text:
                        ctx.reject(dataset, key, "no-docstring")
                        continue
                summary = vx.summary_sentence(doc_text)
                if summary is None:
                    ctx.reject(dataset, key, "docstring-not-prose")
                    continue
                code = vx.without_docstring(facts) if facts else _strip_doc_comment(lang, whole, doc_text)
                if code is None or summary in code:
                    ctx.reject(dataset, key, "docstring-in-code")
                    continue
                if not ctx.fits(code):
                    ctx.reject(dataset, key, "length")
                    continue
                found = licenses.get(repo) or {}
                spdx = found.get("spdx")
                lic = license_(spdx or "LicenseRef-repository-content", False,
                               f"Function from github.com/{repo} under that repository's licence"
                               + (f" ({spdx}, detected from {', '.join(found.get('files', [])[:2])} in CodeSearchNet's "
                                  "licence files)" if spdx else
                                  " (CodeSearchNet kept only repositories whose licence permits redistribution; "
                                  "not detected per repository).")
                               )
                prov = provenance("LicenseRef-repository-content", "hf:code-search-net/code_search_net card (per repository)",
                                  content_spdx=spdx, holder=f"github:{repo}", content_unverified=not spdx,
                                  content_source=("CodeSearchNet original release licence files: " + ", ".join(found.get("files", [])[:3]))
                                  if found else "licence not found in CodeSearchNet's licence files",
                                  concerns=[] if spdx else ["repository licence not detected"])
                path_ref = row["func_path_in_repository"]
                refs = [{"text": path_ref, "kind": "path"}] + \
                    ([{"text": row["func_name"], "kind": "identifier"}] if row["func_name"] else [])  # anonymous JS
                common = dict(upstream="hf:code-search-net/code_search_net", upstream_id=row.get("func_code_url"),
                              revision=meta["revision"], store_version=lang, row=row["__row"], lic=lic, prov=prov,
                              split=SPLIT_MAP[split], groups=[group_key("repo", repo)])
                fmt = f"{lang} function"
                doc = Doc(ctx, dataset=dataset, key=key, artifact="code", text=code, title=f"{path_ref}:{row['func_name']}",
                          refs=refs, meta={"format": fmt, "language": lang, "repository": repo, "docstring_removed": True},
                          **common)
                name = facts["name"] if facts else row["func_name"].split(".")[-1]
                doc.summary(f"Describe in one sentence what the function `{name}` does." if name else
                            "Describe in one sentence what this function does.", summary, origin="upstream-docstring",
                            general="Keep what a description of the function needs.")
                n = 0
                if facts and facts["params"]:
                    doc.extract(f"List the parameters of `{name}` in order, one per line.", "\n".join(facts["params"]), n,
                                method="python-ast-params")
                    n += 1
                if facts and 1 <= len(facts["calls"]) <= 15:
                    doc.extract(f"List every function or method that `{name}` calls, once each, in order of first appearance, "
                                "one per line.", "\n".join(facts["calls"]), n, method="python-ast-calls")
                yield doc
                if ctx.fits(whole):
                    full = Doc(ctx, dataset=dataset, key=key + "#whole", artifact="code", text=whole,
                               title=f"{path_ref}:{row['func_name']}", refs=refs,
                               meta={"format": fmt, "language": lang, "repository": repo}, **common)
                    full.reconstruct()
                    yield full
                made += 1


# ---------------------------------------------------------------------------------------------- HTML

def websrc(ctx: Ctx):
    """WebSRC pages (dataset tag ids and scripts removed): the dataset's questions, and CSS-selector extraction."""
    meta = _raw_meta(ctx, "websrc")
    lic = license_("CC-BY-4.0", False, "WebSRC v1.0 (X-LANCE/WebSRC_v1.0); page content from the source websites.")
    prov = provenance("CC-BY-4.0", "hf:X-LANCE/WebSRC_v1.0 card", content_source="third-party web pages (cars, books, "
                      "jobs, sports, ... sites)", holder="the source websites", content_unverified=True,
                      concerns=["page text and markup belong to the source websites"])
    csvs = sorted((ctx.raw / "websrc").rglob("dataset.csv"))
    pages = []
    for csv_path in csvs:
        site = csv_path.parent
        ctx.use(csv_path, dataset="websrc", upstream="hf:X-LANCE/WebSRC_v1.0", revision=meta["revision"])
        with open(csv_path, encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        by_page: dict[str, list] = defaultdict(list)
        for r in rows:
            by_page[r["id"][2:9]].append(r)
        for page_id in sorted(by_page):
            html_path = site / "processed_data" / f"{page_id}.html"
            if html_path.exists():
                pages.append((site, page_id, html_path, by_page[page_id]))
    pages.sort(key=lambda p: _h("websrc", p[1], str(p[0])))
    made = 0
    for site, page_id, html_path, qs in pages:
        if made >= ctx.caps["websrc"]:
            break
        site_key = "/".join(site.relative_to(ctx.raw / "websrc").parts[-2:])
        html = vx.clean_html(html_path.read_text(encoding="utf-8", errors="replace"), drop_attrs=("tid",))
        if not ctx.fits(html):
            ctx.reject("websrc", page_id, "length")
            continue
        ctx.use(html_path, dataset="websrc", upstream="hf:X-LANCE/WebSRC_v1.0", revision=meta["revision"])
        dom = vx.HtmlDoc(html)
        group = group_key("websrc-site", site_key)
        doc = Doc(ctx, dataset="websrc", key=f"{site_key}/{page_id}", artifact="html", text=html, title=dom.title() or None,
                  upstream="hf:X-LANCE/WebSRC_v1.0", upstream_id=page_id, revision=meta["revision"], row=page_id, lic=lic,
                  split=carve(group), groups=[group], refs=exact_refs_from(html, limit=16), prov=prov,
                  meta={"format": "html", "normalized": "scripts, styles, svg, comments and WebSRC tid attributes removed"})
        n = 0
        for r in sorted(qs, key=lambda r: _h(ctx.seed, r["id"])):
            if n >= QUESTIONS_PER_SOURCE:
                break
            answer = (r.get("answer") or "").strip()
            if not answer:
                continue
            doc.qa(r["question"].strip(), answer, n, extra_notes={"upstream_question_id": r["id"]})
            n += 1
        for m, (request, answer, method) in enumerate(_html_extracts(dom, ctx.rng("websrc", page_id))):
            doc.extract(request, answer, m, method=method)
        doc.reconstruct()
        made += 1
        yield doc


def _html_extracts(dom: vx.HtmlDoc, rng: random.Random) -> list[tuple[str, str, str]]:
    cands = []
    title = dom.title()
    if title:
        cands.append(("What is the text of the page's `title` element?", title, "css:title"))
    for tag in ("h1", "h2", "h3", "th", "li"):
        texts = [n.text() for n in dom.select(tag)]
        texts = [t for t in texts if t]
        if 2 <= len(texts) <= 12 and len("\n".join(texts)) <= 600:
            cands.append((f"List the text of every `{tag}` element in document order, one per line.", "\n".join(texts), f"css:{tag}"))
    links = [n.attrs.get("href", "").strip() for n in dom.select("a[href]")]
    links = [l for l in links if l and not l.startswith(("javascript:", "#"))]
    if 1 <= len(links) <= 12:
        cands.append(("List the `href` of every `a` element with an href, in document order, one per line.", "\n".join(links), "css:a[href]"))
    ids = [n for n in dom.root.elements() if n.attrs.get("id") and re.match(r"^[A-Za-z][\w-]*$", n.attrs["id"])]
    id_texts = [(n.attrs["id"], n.text()) for n in ids if 2 <= len(n.text()) <= 200]
    seen_ids = Counter(n.attrs["id"] for n in ids)
    id_texts = [(i, t) for i, t in id_texts if seen_ids[i] == 1]
    if id_texts:
        i, t = rng.choice(id_texts)
        cands.append((f"What is the text content of the element matching `#{i}`?", t, "css:#id"))
    tables = dom.select("table")
    if tables:
        rows = dom.select("table tr")
        if 2 <= len(rows):
            cands.append(("How many `tr` elements are inside `table` elements on the page?", str(len(rows)), "css:table tr"))
    rng.shuffle(cands)
    return cands[:EXTRACTS_PER_SOURCE]


# ---------------------------------------------------------------------------------------------- JSON / YAML

def _json_extracts(value, rng: random.Random, limit: int = EXTRACTS_PER_SOURCE) -> list[tuple[str, str, str]]:
    nodes = list(vx.walk(value))
    scalars = [(p, v) for p, v in nodes if p != "$" and not isinstance(v, (dict, list)) and v is not None
               and not (isinstance(v, str) and (not v.strip() or len(v) > 200))]
    counts = Counter(json.dumps(v, sort_keys=True) for _, v in scalars)
    cands = []
    deep = [s for s in scalars if vx.depth(s[0]) >= 2] or scalars
    if deep:
        p, v = rng.choice(deep)
        cands.append((f"What is the value at the JSONPath `{p}`? Give the value only (strings without quotes).",
                      vx.scalar_text(v), "jsonpath-value"))
    objects = [(p, v) for p, v in nodes if isinstance(v, dict) and 2 <= len(v) <= 30]
    if objects:
        p, v = rng.choice(objects)
        cands.append((f"List the keys of the object at `{p}` in order, one per line.", "\n".join(map(str, v)), "jsonpath-keys"))
    arrays = [(p, v) for p, v in nodes if isinstance(v, list) and len(v) >= 2]
    if arrays:
        p, v = rng.choice(arrays)
        cands.append((f"How many items does the array at `{p}` have?", str(len(v)), "jsonpath-length"))
    unique = [(p, v) for p, v in scalars if isinstance(v, str) and 4 <= len(v) <= 80 and counts[json.dumps(v, sort_keys=True)] == 1]
    if unique:
        p, v = rng.choice(unique)
        cands.append((f"At which JSONPath is the string value {json.dumps(v, ensure_ascii=False)}? Answer with the path.",
                      p, "jsonpath-locate"))
    rng.shuffle(cands)
    return cands[:limit]


def xlam(ctx: Ctx):
    """xLAM function-calling: the tool schema list (JSON) is the artifact; the query and gold calls are the dataset QA."""
    rows = json.loads(XLAM.read_text())
    ctx.use(XLAM, dataset="xlam", upstream="hf:Salesforce/xlam-function-calling-60k", revision="26d14eb")
    lic = license_("CC-BY-4.0", False, "Salesforce/xlam-function-calling-60k.")
    order = sorted(range(len(rows)), key=lambda i: _h("xlam", i))
    made = 0
    for i in order:
        if made >= ctx.caps["xlam"]:
            break
        r = rows[i]
        tools_text = r["tools"]
        try:
            tools = json.loads(tools_text)
            answers = json.loads(r["answers"])
        except json.JSONDecodeError:
            ctx.reject("xlam", r["id"], "json")
            continue
        if not ctx.fits(tools_text) or not isinstance(tools, list) or not answers:
            ctx.reject("xlam", r["id"], "length-or-empty")
            continue
        names = sorted(t.get("name", "") for t in tools if isinstance(t, dict))
        group = group_key("xlam-tools", _h(*names)[:16])
        schema_group = group_key("tool-schema", "+".join(sorted(set(n.strip() for n in names if n))))  # S1's convention
        doc = Doc(ctx, dataset="xlam", key=r["id"], artifact="data", text=tools_text, title="tool schemas (JSON)",
                  upstream="hf:Salesforce/xlam-function-calling-60k", upstream_id=str(r["id"]), revision="26d14eb", row=i,
                  lic=lic, split=carve(group), groups=[group, schema_group], source_role="schema",
                  prov=provenance("CC-BY-4.0", "hf:Salesforce/xlam-function-calling-60k card", content_spdx="CC-BY-4.0",
                                  content_source="generated queries and calls over API descriptions (APIGen)",
                                  concerns=["queries and calls were generated by models in the dataset's pipeline"]),
                  refs=[{"text": n, "kind": "identifier"} for n in names if n][:16], meta={"format": "json"})
        doc.qa(r["query"].strip(), answers, 0, target_kind="calls",
               request=r["query"].strip() + "\nAnswer with the function calls as a JSON list of {\"name\", \"arguments\"} objects.")
        for m, (request, answer, method) in enumerate(_json_extracts(tools, ctx.rng("xlam", r["id"]), limit=2)):
            doc.extract(request, answer, m, method=method)
        doc.reconstruct()
        made += 1
        yield doc


PERMISSIVE = [
    ("MIT", re.compile(r"Permission is hereby granted, free of charge", re.I)),
    ("Apache-2.0", re.compile(r"Apache License,?\s+Version 2\.0", re.I)),
    ("BSD-3-Clause", re.compile(r"Redistribution and use in source and binary forms[\s\S]*Neither the name", re.I)),
    ("BSD-2-Clause", re.compile(r"Redistribution and use in source and binary forms", re.I)),
    ("ISC", re.compile(r"Permission to use, copy, modify, and/or distribute this software for any purpose", re.I)),
    ("Unlicense", re.compile(r"This is free and unencumbered software released into the public domain", re.I)),
]
LICENSE_FILES = re.compile(r"^(LICEN[SC]E|COPYING)(\.(md|txt|rst))?$", re.I)
CONFIG_FILE = re.compile(r"\.(json|ya?ml)$", re.I)
SKIP_CONFIG = re.compile(r"(^|/)(package-lock\.json|yarn\.lock|pnpm-lock\.yaml|composer\.lock|.*\.min\.json|.*\.map\.json"
                         r"|node_modules/.*|vendor/.*|dist/.*|build/.*)$", re.I)


def _git(repo: Path, *args) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=60,
                          check=True).stdout


def classify_license(text: str) -> str | None:
    for spdx, pattern in PERMISSIVE:
        if pattern.search(text):
            return spdx
    return None


def repo_configs(ctx: Ctx):
    """JSON and YAML files from permissively licensed bare repositories (bgkit's mirror); JSONPath targets."""
    import yaml

    # Lazy, seeded walk: owners in hash order, then their repositories in hash order. Listing every repository of the
    # mirror first costs a metadata walk over ~12k directories on the shared external drive.
    owners = sorted((p.name for p in os.scandir(REPOS) if p.is_dir()), key=lambda o: _h("repo_configs", o))

    def repos_in_order():
        for owner in owners:
            names = sorted((e.name for e in os.scandir(REPOS / owner) if e.is_dir()),
                           key=lambda n: _h("repo_configs", owner, n))
            for n in names:
                yield REPOS / owner / n

    made = 0
    for repo in repos_in_order():
        if made >= ctx.caps["repo_configs"]:
            break
        name = str(repo.relative_to(REPOS))
        try:
            head = _git(repo, "rev-parse", "HEAD").strip()
            files = _git(repo, "ls-tree", "-r", "-l", "HEAD").splitlines()
        except (subprocess.SubprocessError, OSError):
            ctx.reject("repo_configs", name, "git")
            continue
        entries = []
        for line in files:
            meta_part, _, path = line.partition("\t")
            parts = meta_part.split()
            if len(parts) == 4 and parts[1] == "blob" and parts[3].isdigit():
                entries.append((path, int(parts[3])))
        lic_paths = [p for p, _ in entries if "/" not in p and LICENSE_FILES.match(p)]
        spdx = None
        for lp in lic_paths:
            spdx = classify_license(_git(repo, "cat-file", "-p", f"HEAD:{lp}"))
            if spdx:
                break
        if not spdx:
            ctx.reject("repo_configs", name, "license-not-permissive-or-unknown")
            continue
        cands = [(p, s) for p, s in entries if CONFIG_FILE.search(p) and not SKIP_CONFIG.search(p)
                 and ctx.min_chars <= s <= ctx.max_chars]
        cands.sort(key=lambda c: _h("repo_configs", name, c[0]))
        taken = 0
        for path, _size in cands:
            if taken >= 2 or made >= ctx.caps["repo_configs"]:
                break
            text = _git(repo, "cat-file", "-p", f"HEAD:{path}")
            fmt = "json" if path.lower().endswith(".json") else "yaml"
            try:
                value = json.loads(text) if fmt == "json" else yaml.safe_load(text)
            except (json.JSONDecodeError, yaml.YAMLError, ValueError):
                ctx.reject("repo_configs", f"{name}:{path}", f"parse-{fmt}")
                continue
            if not isinstance(value, (dict, list)) or not ctx.fits(text):
                ctx.reject("repo_configs", f"{name}:{path}", "shape-or-length")
                continue
            try:
                json.dumps(value, allow_nan=False)
            except (TypeError, ValueError):  # YAML dates, sets or other non-JSON scalars: JSONPath answers would be ambiguous
                ctx.reject("repo_configs", f"{name}:{path}", "non-json-scalars")
                continue
            extracts = _json_extracts(value, ctx.rng("repo_configs", name, path))
            if not extracts:
                ctx.reject("repo_configs", f"{name}:{path}", "no-extract")
                continue
            lic = license_(spdx, False, f"File from {name} (licence detected from its {', '.join(lic_paths)} at HEAD).")
            prov = provenance(spdx, f"github:{name} {', '.join(lic_paths)} at HEAD (regex detection)", content_spdx=spdx,
                              content_source="repository file", holder=f"github:{name}",
                              concerns=["licence detected by a regular expression; spot-check"])
            doc = Doc(ctx, dataset="repo_configs", key=f"{name}:{path}", artifact="data", text=text, title=path,
                      upstream=f"github:{name}", upstream_id=path, revision=head, row=f"{name}:{path}", lic=lic,
                      split=carve(group_key("repo", name)), groups=[group_key("repo", name)], prov=prov,
                      refs=[{"text": path, "kind": "path"}], meta={"format": fmt, "repository": name,
                                                                   "note": "JSONPath over the parsed document"})
            for m, (request, answer, method) in enumerate(extracts):
                if fmt == "yaml":
                    request = request.replace("JSONPath", "JSONPath (over the parsed YAML)", 1) if "JSONPath" in request \
                        else request + " (Paths are JSONPath over the parsed YAML.)"
                doc.extract(request, answer, m, method=method)
            doc.reconstruct()
            taken += 1
            made += 1
            yield doc
        if taken:
            ctx.use(repo, dataset="repo_configs", upstream=f"github:{name}", revision=head, kind="git-repository")


# ---------------------------------------------------------------------------------------------- tables

def _table_extracts(rows: list[list[str]], rng: random.Random, limit: int = 2) -> list[tuple[str, str, str]]:
    header, body = rows[0], rows[1:]
    if len(body) < 2 or not header:
        return []
    cands = [("How many data rows does the table have (not counting the header row)?", str(len(body)), "table-row-count")]
    width = len(header)
    keys = [c for c in range(width) if header[c].strip() and len({r[c] for r in body if c < len(r)}) == len(body)
            and all(c < len(r) and r[c].strip() for r in body)]
    if keys:
        k = keys[0]
        others = [c for c in range(width) if c != k and header[c].strip()]
        if others:
            r = rng.choice(body)
            c = rng.choice(others)
            if c < len(r) and r[c].strip():
                cands.append((f"In the row whose \"{header[k]}\" is \"{r[k]}\", what is the value of \"{header[c]}\"?",
                              r[c].strip(), "table-cell"))
        for c in others:
            nums = [vx.number(r[c]) if c < len(r) else None for r in body]
            if all(n is not None for n in nums):
                best = max(nums)
                if nums.count(best) == 1:
                    cands.append((f"Which \"{header[k]}\" has the largest \"{header[c]}\"?", body[nums.index(best)][k],
                                  "table-argmax"))
                    break
    rng.shuffle(cands)
    return cands[:limit]


def _wtq_unescape(s: str) -> str:
    return s.replace("\\n", "\n").replace("\\p", "|").replace("\\\\", "\\")


def wtq(ctx: Ctx):
    """WikiTableQuestions: tables as CSV, the dataset's questions (training set → train; pristine unseen tables →
    test), and code-computed cell, count and arg-max targets."""
    meta = _raw_meta(ctx, "wtq")
    root = ctx.raw / "wtq/repo"
    lic = license_("CC-BY-SA-4.0", False, "WikiTableQuestions (ppasupat/WikiTableQuestions); Wikipedia tables.")
    prov = provenance("CC-BY-SA-4.0", "github:ppasupat/WikiTableQuestions README", content_spdx="CC-BY-SA-4.0",
                      content_source="Wikipedia tables", holder="Wikipedia contributors")
    by_table: dict[tuple[str, str], list] = defaultdict(list)
    for fname, split in (("training.tsv", "train"), ("pristine-unseen-tables.tsv", "test")):
        path = root / "data" / fname
        ctx.use(path, dataset="wtq", upstream="github:ppasupat/WikiTableQuestions", revision=meta["revision"])
        with open(path, encoding="utf-8") as stream:
            header = stream.readline().rstrip("\n").split("\t")
            for line in stream:
                row = dict(zip(header, line.rstrip("\n").split("\t")))
                by_table[(split, row["context"])].append(row)
    want = {"train": round(ctx.caps["wtq"] * 0.9), "test": ctx.caps["wtq"] - round(ctx.caps["wtq"] * 0.9)}
    made = Counter()
    for split, table_path in sorted(by_table, key=lambda k: _h("wtq", *k)):
        if made[split] >= want[split]:
            continue
        csv_path = root / table_path
        try:
            with open(csv_path, encoding="utf-8", newline="") as stream:
                rows = [[_wtq_unescape(c) for c in r] for r in csv.reader(stream)]
        except (OSError, csv.Error):
            ctx.reject("wtq", table_path, "table-read")
            continue
        text = vx.table_csv(rows)
        if not ctx.fits(text) or len(rows) < 3:
            ctx.reject("wtq", table_path, "length")
            continue
        group = group_key("wtq-table", table_path)
        doc = Doc(ctx, dataset="wtq", key=table_path, artifact="table", text=text, title=None,
                  upstream="github:ppasupat/WikiTableQuestions", upstream_id=table_path, revision=meta["revision"],
                  row=table_path, lic=lic, split=split if split == "test" else carve(group), groups=[group], prov=prov,
                  meta={"format": "csv"})
        qs = sorted(by_table[(split, table_path)], key=lambda r: _h(ctx.seed, r["id"]))[:QUESTIONS_PER_SOURCE]
        for n, q in enumerate(qs):
            values = [_wtq_unescape(v) for v in q["targetValue"].split("|")]
            doc.qa(_wtq_unescape(q["utterance"]).strip(), ", ".join(values), n,
                   extra_notes={"upstream_question_id": q["id"], "answer_values": values})
        for m, (request, answer, method) in enumerate(_table_extracts(rows, ctx.rng("wtq", table_path))):
            doc.extract(request, answer, m, method=method)
        doc.reconstruct()
        ctx.use(csv_path, dataset="wtq", upstream="github:ppasupat/WikiTableQuestions", revision=meta["revision"])
        made[split] += 1
        yield doc


def fetaqa(ctx: Ctx):
    """FeTaQA: Wikipedia tables (as CSV) with free-form answers to the dataset's questions."""
    meta = _raw_meta(ctx, "fetaqa")
    lic = license_("CC-BY-SA-4.0", False, "FeTaQA (Yale-LILY/FeTaQA); Wikipedia tables.")
    prov = provenance("CC-BY-SA-4.0", "github:Yale-LILY/FeTaQA LICENSE", content_spdx="CC-BY-SA-4.0",
                      content_source="Wikipedia tables", holder="Wikipedia contributors")
    for fname, split in (("fetaQA-v1_train.jsonl", "train"), ("fetaQA-v1_dev.jsonl", "validation"), ("fetaQA-v1_test.jsonl", "test")):
        path = ctx.raw / "fetaqa/data" / fname
        ctx.use(path, dataset="fetaqa", upstream="github:Yale-LILY/FeTaQA", revision=meta["revision"])
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        want = max(1, round(ctx.caps["fetaqa"] * UPSTREAM_SHARE[split]))
        made = 0
        for i in sorted(range(len(rows)), key=lambda i: _h("fetaqa", split, i)):
            if made >= want:
                break
            r = rows[i]
            table = [[str(c) for c in row] for row in r["table_array"]]
            text = vx.table_csv(table)
            if not ctx.fits(text):
                ctx.reject("fetaqa", r["feta_id"], "length")
                continue
            title = f"{r.get('table_page_title', '')} — {r.get('table_section_title', '')}".strip(" —")
            group = group_key("wiki", r.get("table_page_title") or str(r["feta_id"]))
            doc = Doc(ctx, dataset="fetaqa", key=r["feta_id"], artifact="table", text=text, title=title,
                      upstream="github:Yale-LILY/FeTaQA", upstream_id=str(r["feta_id"]), revision=meta["revision"], row=i,
                      lic=lic, split=split, groups=[group], prov=prov,
                      refs=[{"text": r["page_wikipedia_url"], "kind": "url"}] if r.get("page_wikipedia_url") else [],
                      meta={"format": "csv"})
            doc.qa(r["question"].strip(), r["answer"].strip(), 0, extra_notes={"upstream_question_id": r["feta_id"]})
            for m, (request, answer, method) in enumerate(_table_extracts(table, ctx.rng("fetaqa", r["feta_id"]), limit=1)):
                doc.extract(request, answer, m, method=method)
            made += 1
            yield doc


# ---------------------------------------------------------------------------------------------- logs and tool outputs

def _log_extracts(text: str, rng: random.Random, command: str = "") -> list[tuple[str, str, str]]:
    """Facts of one tool output, informative ones first: the exit code (a trailer at the end of every OpenHands
    output) only fills a slot that nothing else takes. Search facts only for grep/rg commands."""
    cands = []
    failed = vx.failing_tests(text)
    if 1 <= len(failed) <= 20:
        cands.append(("List the ids of the failing or erroring tests, once each, in order, one per line.", "\n".join(failed),
                      "log-failing-tests"))
    summary = vx.pytest_summary(text)
    if summary:
        cands.append(("What does pytest's final summary line report (the counts before \"in …s\")?", summary, "log-pytest-summary"))
    exc = vx.final_exception(text)
    if exc:
        cands.append(("What exception ended the last traceback? Answer as `Type: message`.",
                      f"{exc['type']}: {exc['message']}" if exc["message"] else exc["type"], "log-exception"))
        cands.append(("Where was the last traceback's innermost frame? Answer as `path:line`.", f"{exc['path']}:{exc['line']}",
                      "log-innermost-frame"))
    hits = vx.grep_hits(text) if re.search(r"\b(grep|rg|ag|git grep)\b", command) else []
    if 2 <= len(hits) <= 200:
        files = list(dict.fromkeys(p for p, _ in hits))
        if len(files) <= 15:
            cands.append(("Which files have matches in this search output? List each once, in order, one per line.",
                          "\n".join(files), "log-grep-files"))
        cands.append(("How many matching lines does this search output list?", str(len(hits)), "log-grep-count"))
    rng.shuffle(cands)
    code = vx.exit_code(text)
    if code is not None:
        cands.append(("What exit code did the command finish with? Answer with the number.", str(code), "log-exit-code"))
    return cands[:EXTRACTS_PER_SOURCE]


def swe_tool_outputs(ctx: Ctx):
    """Bash outputs from SWE-rebench OpenHands trajectories (workspace renamed as in the harness bench), with
    code-computed facts; the agent's command is the purpose context. Splits follow the harness bench (repository)."""
    import pyarrow.parquet as pq

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training" / "neuralese"))
    from natlang_neuralese.harness_bench.openhands import normalize
    from natlang_neuralese.harness_bench.records import split_of

    ctx.use(NEBIUS, dataset="swe_tool_outputs", upstream="hf:nebius/SWE-rebench-openhands-trajectories", revision=None)
    lic = license_("CC-BY-4.0", False, "nebius/SWE-rebench-openhands-trajectories (CC-BY-4.0); tool output over "
                   "public repositories under their own licences.")
    pf = pq.ParquetFile(str(NEBIUS))
    groups = list(range(pf.metadata.num_row_groups))
    ctx.rng("swe_tool_outputs", "groups").shuffle(groups)
    made = 0
    for g in groups:
        if made >= ctx.caps["swe_tool_outputs"]:
            break
        base = sum(pf.metadata.row_group(k).num_rows for k in range(g))
        for batch in pf.iter_batches(batch_size=32, row_groups=[g], columns=["trajectory_id", "instance_id", "repo", "trajectory"]):
            for local, row in enumerate(batch.to_pylist()):
                if made >= ctx.caps["swe_tool_outputs"]:
                    break
                index = base + local
                if int(_h("swe-pick", ctx.seed, index)[:4], 16) % 8:  # an eighth of the trajectories, spread out
                    continue
                try:
                    t = normalize(row, "pi", workspace="/workspace/project")
                except (KeyError, TypeError, ValueError):
                    ctx.reject("swe_tool_outputs", index, "normalize")
                    continue
                calls = {}
                outputs = []
                for m in t.messages:
                    if m["role"] == "assistant":
                        for part in m["content"]:
                            if part.get("type") == "toolCall":
                                calls[part["id"]] = part
                    elif m["role"] == "toolResult" and m.get("toolName") == "bash":
                        text = "\n".join(p.get("text", "") for p in m["content"]).strip()
                        call = calls.get(m["toolCallId"])
                        if call and ctx.fits(text) and len(text) >= 800:
                            outputs.append((m["toolCallId"], call, text))
                rng = ctx.rng("swe_tool_outputs", t.id)
                rng.shuffle(outputs)
                # Outputs with more facts than the exit code first (stable within equal counts: seeded order).
                scored = []
                for call_id, call, text in outputs:
                    command = str(call["arguments"].get("command", "")).strip()
                    extracts = _log_extracts(text, ctx.rng("swe_tool_outputs", t.id, call_id), command)
                    informative = sum(1 for _, _, m in extracts if m != "log-exit-code")
                    scored.append((-informative, len(scored), call_id, command, text, extracts))
                scored.sort()
                taken = 0
                for _, _, call_id, command, text, extracts in scored:
                    if taken >= 2:
                        break
                    if not extracts:
                        continue
                    # The harness bench's groups, plus S1's (repo:, swe-instance:) for cross-corpus closure.
                    groups_ = [f"swe-rebench-repo:{t.repo}", f"swe-rebench-instance:{t.instance_id}",
                               group_key("repo", t.repo), group_key("swe-instance", t.instance_id)]
                    doc = Doc(ctx, dataset="swe_tool_outputs", key=f"{t.id}:{call_id}", artifact="log", text=text,
                              title=f"bash: {command[:200]}", upstream="hf:nebius/SWE-rebench-openhands-trajectories",
                              upstream_id=t.id, revision=None, row=index, lic=lic, split=split_of(t.repo, 5), groups=groups_,
                              refs=exact_refs_from(text, limit=24),
                              meta={"format": "bash output", "tool": "bash", "command": command[:2000]},
                              notes={"instance_id": t.instance_id},
                              prov=provenance("CC-BY-4.0", "hf:nebius/SWE-rebench-openhands-trajectories card",
                                              content_source="command output over a public repository",
                                              holder=f"github:{t.repo}",
                                              concerns=["outputs can quote repository files under the repository's licence"]))
                    for m, (request, answer, method) in enumerate(extracts):
                        doc.extract(request, answer, m, method=method)
                    doc.reconstruct()
                    taken += 1
                    made += 1
                    yield doc


# ---------------------------------------------------------------------------------------------- v2 sources

SQL_MAX_ROWS = 400
SQL_ANSWER_ROWS = 20


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return format(value, ".10g")
    if isinstance(value, bytes):
        return value.hex()
    return str(value)


def _sqlite(path: Path):
    import sqlite3

    return sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)


def _execute(conn, sql: str, seconds: float = 5.0):
    """Rows of `sql` (at most SQL_ANSWER_ROWS + 1), aborted after `seconds`."""
    import time as _time

    deadline = _time.monotonic() + seconds
    conn.set_progress_handler(lambda: 1 if _time.monotonic() > deadline else 0, 10_000)
    try:
        return conn.execute(sql).fetchmany(SQL_ANSWER_ROWS + 1)
    finally:
        conn.set_progress_handler(None, 0)


def sql_answer(rows: list[tuple]) -> str | None:
    """The executed gold query's result as text: one row per line, columns joined by ' | '. None for no rows, too
    many rows, or only NULLs."""
    if not rows or len(rows) > SQL_ANSWER_ROWS:
        return None
    lines = [" | ".join(_cell(v) for v in row) for row in rows]
    return None if not any(l.strip(" |") for l in lines) else "\n".join(lines)


def _sql_items(ctx: Ctx):
    """(dataset, split, question dicts, database root) for Spider (train, train_others; dev as validation; the
    test set is left out) and BIRD (train; dev as validation)."""
    if SPIDER.exists():
        train = []
        for name in ("train_spider.json", "train_others.json"):
            ctx.use(SPIDER / name, dataset="spider", upstream="official:spider_data.zip", revision="00636695")
            train += json.loads((SPIDER / name).read_text())
        ctx.use(SPIDER / "dev.json", dataset="spider", upstream="official:spider_data.zip", revision="00636695")
        yield "spider", "train", train, SPIDER / "database"
        yield "spider", "validation", json.loads((SPIDER / "dev.json").read_text()), SPIDER / "database"
    for split, (questions, root) in BIRD.items():
        if questions.exists() and root.exists():
            ctx.use(questions, dataset="bird", upstream="bird-bench.github.io", revision="train.zip 66e9e311 / dev.zip cdd6d19f")
            yield "bird", split, json.loads(questions.read_text()), root


def spider_bird(ctx: Ctx):
    """Spider and BIRD: the tables a gold query reads (each at most SQL_MAX_ROWS rows), rendered as CSV, with the
    dataset's question as the purpose and the answer computed by executing the gold SQL on the database."""
    cap = ctx.caps.get("spider_bird", 0)
    if not cap:
        return
    made = 0
    per_split = {"train": round(cap * 0.92), "validation": cap - round(cap * 0.92)}
    for dataset, split, questions, root in _sql_items(ctx):
        made_split = 0
        by_db: dict[str, list] = defaultdict(list)
        for i, q in enumerate(questions):
            by_db[q["db_id"]].append((i, q))
        lic = license_("CC-BY-SA-4.0", False, f"{dataset} (CC-BY-SA-4.0); databases from the dataset release.")
        prov = provenance("CC-BY-SA-4.0", f"{dataset} release / HF card", content_spdx="CC-BY-SA-4.0",
                          content_source=f"{dataset} SQLite databases", holder=f"{dataset} authors")
        for db in sorted(by_db, key=lambda d: _h(dataset, d)):
            if made_split >= per_split.get(split, 0) or made >= cap:
                break
            path = root / db / f"{db}.sqlite"
            if not path.exists():
                ctx.reject(dataset, db, "database-missing")
                continue
            conn = _sqlite(path)
            try:
                tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
                small = {}
                for t in tables:
                    try:
                        n = conn.execute(f'SELECT count(*) FROM (SELECT 1 FROM "{t}" LIMIT {SQL_MAX_ROWS + 1})').fetchone()[0]
                    except Exception:  # noqa: BLE001 - malformed table in an upstream database
                        continue
                    if 2 <= n <= SQL_MAX_ROWS:
                        small[t] = n
                groups_of: dict[tuple, list] = defaultdict(list)
                for i, q in by_db[db]:
                    sql = q.get("query") or q.get("SQL") or ""
                    used = [t for t in tables if re.search(r'(?<![\w"`])["`]?' + re.escape(t) + r'["`]?(?![\w"`])', sql, re.I)]
                    if not used or any(t not in small for t in used):
                        ctx.reject(dataset, f"{db}#{i}", "table-size-or-unresolved")
                        continue
                    groups_of[tuple(sorted(used))].append((i, q, sql))
                ctx.use(path, dataset=dataset, upstream=dataset, revision=None, kind="sqlite", hash=False)
                for used, qs in sorted(groups_of.items(), key=lambda kv: _h(dataset, db, *kv[0])):
                    if made_split >= per_split.get(split, 0) or made >= cap:
                        break
                    parts = []
                    for t in used:
                        cur = conn.execute(f'SELECT * FROM "{t}"')
                        header = [d[0] for d in cur.description]
                        parts.append(f"# table: {t}\n" + vx.table_csv([header] + [[_cell(v) for v in row] for row in cur.fetchall()]))
                    text = "\n\n".join(parts)
                    if not ctx.fits(text):
                        ctx.reject(dataset, f"{db}:{'+'.join(used)}", "length")
                        continue
                    group = group_key("sql-db", f"{dataset}:{db}")  # S1's convention
                    doc = Doc(ctx, dataset=dataset, key=f"{db}:{'+'.join(used)}", artifact="table", text=text,
                              title=f"{db}: {', '.join(used)}", upstream=dataset, upstream_id=db, revision=None,
                              row=f"{db}:{'+'.join(used)}", lic=lic, prov=prov, split=split, groups=[group],
                              meta={"format": "csv tables", "database": db, "tables": list(used)})
                    n = 0
                    for i, q, sql in sorted(qs, key=lambda x: _h(ctx.seed, dataset, x[0])):
                        if n >= QUESTIONS_PER_SOURCE:
                            break
                        try:
                            answer = sql_answer(_execute(conn, sql))
                        except Exception as error:  # noqa: BLE001 - gold queries that fail on the shipped database
                            ctx.reject(dataset, f"{db}#{i}", f"execution: {type(error).__name__}")
                            continue
                        if answer is None:
                            ctx.reject(dataset, f"{db}#{i}", "answer-empty-or-too-long")
                            continue
                        question = q["question"].strip()
                        hint = (q.get("evidence") or "").strip()
                        request = question + (f"\nHint: {hint}" if hint else "") + \
                            "\nAnswer with the result values only: one row per line, columns separated by ' | '."
                        doc.qa(question, answer, n, request=request, checked="code-computed:sqlite-gold-sql",
                               origin="code-computed", extra_notes={"gold_sql": sql, "db_id": db, "upstream_question_index": i,
                                                                    "evidence": hint or None})
                        n += 1
                    if not n:
                        continue
                    if len(used) == 1:
                        rows = list(csv.reader(io.StringIO(parts[0].split("\n", 1)[1])))
                        for m, (request, answer, method) in enumerate(_table_extracts(rows, ctx.rng(dataset, db, *used), limit=1)):
                            doc.extract(request, answer, m, method=method)
                    doc.reconstruct()
                    made += 1
                    made_split += 1
                    yield doc
            finally:
                conn.close()


def tabfact(ctx: Ctx):
    """TabFact: Wikipedia tables with crowd-written statements labelled entailed/refuted (yes/no `ask`)."""
    cap = ctx.caps.get("tabfact", 0)
    root = ctx.raw / "tabfact"
    if not cap or not root.exists():
        return
    meta = _raw_meta(ctx, "tabfact")
    statements = {}
    for f in ("collected_data/r1_training_all.json", "collected_data/r2_training_all.json"):
        ctx.use(root / f, dataset="tabfact", upstream="github:wenhuchen/Table-Fact-Checking", revision=meta["revision"])
        for t, value in json.loads((root / f).read_text()).items():
            entry = statements.setdefault(t, [[], [], value[2] if len(value) > 2 else ""])
            entry[0] += value[0]
            entry[1] += value[1]
    pages = json.loads((root / "data/table_to_page.json").read_text())
    lic = license_("CC-BY-4.0", False, "TabFact (wenhu/tab_fact card CC-BY-4.0; repository code MIT); Wikipedia tables (CC-BY-SA).")
    prov = provenance("CC-BY-4.0", "hf:wenhu/tab_fact card; github:wenhuchen/Table-Fact-Checking LICENSE (MIT, code)",
                      content_spdx="CC-BY-SA-4.0", content_source="Wikipedia tables (WikiTables)", holder="Wikipedia contributors")
    made = 0
    for split, name in (("train", "train_id.json"), ("validation", "val_id.json"), ("test", "test_id.json")):
        ids = [t for t in json.loads((root / "data" / name).read_text()) if t in statements]
        ids.sort(key=lambda t: _h("tabfact", t))
        want = max(1, round(cap * UPSTREAM_SHARE[split]))
        made_split = 0
        for t in ids:
            if made_split >= want:
                break
            path = root / "data/all_csv" / t
            if not path.exists():
                continue
            rows = [line.split("#") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            text = vx.table_csv(rows)
            if len(rows) < 3 or not ctx.fits(text):
                ctx.reject("tabfact", t, "length")
                continue
            ctx.use(path, dataset="tabfact", upstream="github:wenhuchen/Table-Fact-Checking", revision=meta["revision"])
            stmts, labels, caption = statements[t]
            page = pages.get(t) or []
            groups = [group_key("tabfact-table", t)] + ([group_key("wiki", page[0])] if page and page[0] else [])
            doc = Doc(ctx, dataset="tabfact", key=t, artifact="table", text=text, title=caption or None,
                      upstream="github:wenhuchen/Table-Fact-Checking", upstream_id=t, revision=meta["revision"], row=t,
                      lic=lic, prov=prov, split=split, groups=groups, meta={"format": "csv"},
                      refs=[{"text": page[1], "kind": "url"}] if len(page) > 1 and page[1] else [])
            pairs = sorted(zip(stmts, labels), key=lambda x: _h(ctx.seed, t, x[0]))
            yes = [p for p in pairs if p[1] == 1][:QUESTIONS_PER_SOURCE // 2]
            no = [p for p in pairs if p[1] == 0][:QUESTIONS_PER_SOURCE - len(yes)]
            for n, (stmt, label) in enumerate(sorted(yes + no, key=lambda x: _h(ctx.seed, "order", x[0]))):
                doc.qa(f"Is this statement supported by the table: {stmt.strip()}", "yes" if label == 1 else "no", n,
                       request=f"Statement: {stmt.strip()}\nIs this statement supported by the table? Answer yes or no.",
                       extra_notes={"statement": stmt.strip(), "label": int(label)})
            for m, (request, answer, method) in enumerate(_table_extracts(rows, ctx.rng("tabfact", t), limit=1)):
                doc.extract(request, answer, m, method=method)
            doc.reconstruct()
            made += 1
            made_split += 1
            yield doc


TOOLACE_MARK = "Here is a list of functions in JSON format that you can invoke:"


def toolace(ctx: Ctx):
    """ToolACE: tool-schema lists and tool results (JSON) from its synthetic dialogues. Only code-computed JSONPath
    targets and reconstruction: the dialogues' assistant turns are model-generated and never used as targets."""
    cap = ctx.caps.get("toolace", 0)
    if not cap or not TOOLACE.exists():
        return
    ctx.use(TOOLACE, dataset="toolace", upstream="hf:Team-ACE/ToolACE", revision="6bda777")
    rows = json.loads(TOOLACE.read_text())
    lic = license_("Apache-2.0", False, "Team-ACE/ToolACE (Apache-2.0); synthetic tool schemas and results.")
    prov = provenance("Apache-2.0", "hf:Team-ACE/ToolACE card", content_spdx="Apache-2.0",
                      content_source="tool schemas and results synthesized by models in the ToolACE pipeline",
                      concerns=["model-generated source content (used as sources only; targets are code-computed)"])
    made = 0
    for i in sorted(range(len(rows)), key=lambda i: _h("toolace", i)):
        if made >= cap:
            break
        r = rows[i]
        system = r.get("system") or ""
        at = system.find(TOOLACE_MARK)
        if at < 0:
            ctx.reject("toolace", i, "no-schema")
            continue
        try:
            tools, _end = json.JSONDecoder().raw_decode(system[at + len(TOOLACE_MARK):].lstrip())
        except json.JSONDecodeError:
            ctx.reject("toolace", i, "json")
            continue
        if not isinstance(tools, list):
            ctx.reject("toolace", i, "shape")
            continue
        names = sorted({str(t.get("name", "")).strip() for t in tools if isinstance(t, dict)} - {""})
        group = group_key("tool-schema", "+".join(names))
        split = carve(group)
        text = json.dumps(tools, ensure_ascii=False, indent=1)
        if ctx.fits(text):
            extracts = _json_extracts(tools, ctx.rng("toolace", i))
            doc = Doc(ctx, dataset="toolace", key=f"{i}:schema", artifact="data", text=text, title="tool schemas (JSON)",
                      upstream="hf:Team-ACE/ToolACE", upstream_id=str(i), revision="6bda777", row=i, lic=lic, prov=prov,
                      split=split, groups=[group], source_role="schema", meta={"format": "json"},
                      refs=[{"text": n, "kind": "identifier"} for n in names][:16])
            for m, (request, answer, method) in enumerate(extracts):
                doc.extract(request, answer, m, method=method)
            doc.reconstruct()
            made += 1
            yield doc
        for k, turn in enumerate(r.get("conversations") or []):
            if made >= cap or turn.get("from") != "tool":
                continue
            try:
                value = json.loads(turn.get("value") or "")
            except json.JSONDecodeError:
                continue
            out_text = json.dumps(value, ensure_ascii=False, indent=1)
            if not isinstance(value, (list, dict)) or not ctx.fits(out_text):
                continue
            extracts = _json_extracts(value, ctx.rng("toolace", i, k))
            if not extracts:
                continue
            doc = Doc(ctx, dataset="toolace", key=f"{i}:tool{k}", artifact="data", text=out_text, title="tool result (JSON)",
                      upstream="hf:Team-ACE/ToolACE", upstream_id=str(i), revision="6bda777", row=f"{i}:{k}", lic=lic,
                      prov=prov, split=split, groups=[group], source_role="tool_output", meta={"format": "json"})
            for m, (request, answer, method) in enumerate(extracts):
                doc.extract(request, answer, m, method=method)
            doc.reconstruct()
            made += 1
            yield doc
            break  # one result per dialogue


_S1_TOOL_FILES = (("trajectory_continuation_swe", 2 / 3), ("trajectory_continuation_terminal", 1 / 3))
_REPO_LICENSE_NOTE = re.compile(r"repository \S+ under ([A-Za-z0-9+-]+(?:\.[0-9]+)*)")


def _tool_pairs(messages: list[dict]):
    """(command, tool name, output text) per tool result, pairing each assistant's tool calls with the tool messages
    that follow it in order (S1 trajectories keep tool calls as a Python-literal string)."""
    import ast

    calls: list = []
    for m in messages:
        if m["role"] == "assistant":
            raw = m.get("tool_calls")
            calls = []
            if raw:
                try:
                    calls = ast.literal_eval(raw) if isinstance(raw, str) else list(raw)
                except (ValueError, SyntaxError):
                    calls = []
        elif m["role"] == "tool":
            call = calls.pop(0) if calls else {}
            fn = (call or {}).get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except (json.JSONDecodeError, TypeError):
                args = {}
            command = str(args.get("command") or args.get("cmd") or args.get("pattern") or "") if isinstance(args, dict) else ""
            yield command, fn.get("name") or m.get("name") or "tool", m.get("content") or ""


def _terminal_outputs(messages: list[dict]):
    for m in messages[1:]:
        content = m.get("content") or ""
        if m["role"] == "user" and "Terminal Output" in content[:200]:
            prompt = re.search(r"^\S+@\S+:[^#\n]*# (.*)$", content, re.M)
            yield (prompt.group(1).strip() if prompt else ""), "terminal", content


def s1_tool_outputs(ctx: Ctx):
    """More tool-output diversity (v2): tool results from S1's agent trajectories other than the SWE-rebench
    OpenHands set (Open-SWE-Traces and Nemotron SWE in `trajectory_continuation_swe`, AgentTrove terminal sessions
    in `trajectory_continuation_terminal`), sampled by seeded seeks. Each output keeps its S1 record's split and
    groups, so it is closed with S1 by construction; targets are the same code-computed facts as for bash outputs."""
    cap = ctx.caps.get("s1_tool_outputs", 0)
    if not cap:
        return
    seen: set = set()
    for name, share in _S1_TOOL_FILES:
        path = S1 / f"{name}.port-records.jsonl"
        if not path.exists():
            continue
        ctx.use(path, dataset="s1_tool_outputs", upstream="corpus:s1-full-final-20261003", revision=None, hash=False)
        want = round(cap * share)
        size = path.stat().st_size
        rng = ctx.rng("s1_tool_outputs", name)
        made = 0
        with open(path, "rb") as stream:
            for attempt in range(want * 12):
                if made >= want:
                    break
                stream.seek(rng.randrange(size))
                stream.readline()
                line = stream.readline()
                if not line:
                    continue
                r = json.loads(line)
                upstream = r["lineage"].get("upstream") or ""
                if upstream.startswith("nebius/"):
                    continue  # already covered by swe_tool_outputs from the registered corpus
                messages = r["sources"][0].get("messages") or []
                pairs = list(_tool_pairs(messages) if name.endswith("swe") else _terminal_outputs(messages))
                taken = 0
                for k, (command, tool, text) in enumerate(pairs):
                    text = text.strip()
                    if taken >= 2 or len(text) < 800 or not ctx.fits(text):
                        continue
                    digest = text_hash(text)
                    if digest in seen:
                        continue
                    extracts = _log_extracts(text, ctx.rng("s1_tool_outputs", r["id"], k), command)
                    if not extracts:
                        continue
                    seen.add(digest)
                    lic = dict(r["license"])
                    content = _REPO_LICENSE_NOTE.search(lic.get("notes") or "")
                    doc = Doc(ctx, dataset="s1_tool_outputs", key=f"{r['id']}:{k}", artifact="log", text=text,
                              title=f"{tool}: {command[:200]}", upstream=upstream, upstream_id=r["lineage"].get("upstream_id"),
                              revision=r["lineage"].get("upstream_revision"), row=r["id"], lic=lic, split=r["split"],
                              groups=list(r["split_groups"]), refs=exact_refs_from(text, limit=24),
                              meta={"format": f"{tool} output", "tool": tool, "command": command[:2000]},
                              notes={"s1_record": r["id"]},
                              prov=provenance(lic["spdx"], f"S1 record licence ({upstream.split(':')[0]} card)",
                                              content_spdx=content.group(1) if content else None,
                                              content_source="tool output in an agent trajectory",
                                              concerns=[] if content else ["outputs can quote files under their own licences"]))
                    for m, (request, answer, method) in enumerate(extracts):
                        doc.extract(request, answer, m, method=method)
                    doc.reconstruct()
                    taken += 1
                    made += 1
                    yield doc
        ctx.info[f"s1_tool_outputs:{name}"] = {"outputs": made, "wanted": want}


# ---------------------------------------------------------------------------------------------- build

ADAPTERS = {
    "cnn_dailymail": cnn_dailymail, "squad": squad, "quality": quality, "qmsum": qmsum,
    "codesearchnet": codesearchnet,
    "websrc": websrc,
    "xlam": xlam, "repo_configs": repo_configs,
    "wtq": wtq, "fetaqa": fetaqa,
    "swe_tool_outputs": swe_tool_outputs,
    "spider_bird": spider_bird, "tabfact": tabfact, "toolace": toolace, "s1_tool_outputs": s1_tool_outputs,
}


def _raw_meta(ctx: Ctx, dataset: str) -> dict:
    sources = json.loads((ctx.raw / "sources.json").read_text())
    for f in sources["files"]:
        if f.get("dataset") == dataset:
            return f
    raise FileNotFoundError(f"{dataset} is not in {ctx.raw / 'sources.json'}; run fetch first")


def mark_compare(docs: list[Doc]) -> int:
    """A source with several purposes: its consume records become `compare` and list each other."""
    changed = 0
    for doc in docs:
        purposeful = [r for r in doc.records if r["task"] == "consume"]
        if len({r["writer"]["instructions"] for r in purposeful}) < 2:
            continue
        for r in purposeful:
            r["task"] = "compare"
            r["contrasts"]["purpose_pairs"] = [o["id"] for o in purposeful if o is not r][:16]
            changed += 1
    return changed


def check(record: dict) -> list[str]:
    errors = validate_with_schema(record)
    errors += leakage(record)
    try:
        parse_record(record)
    except RecordError as error:
        errors.append(f"trainer-loader: {error}")
    return errors


def build(raw: Path, out: Path, *, corpus_id: str, caps: dict, max_source_chars: int = MAX_SOURCE_CHARS,
          min_source_chars: int = MIN_SOURCE_CHARS, seed: int = 0, protected_path: Path | None = PROTECTED_INDEX,
          only: list[str] | None = None, adapters: dict | None = None, cross_indexes=(), log=print) -> dict:
    """Build the corpus. `cross_indexes` are published-corpus indexes (cross_corpus.index_corpus) to close splits
    and deduplicate against; the build fails if a kept record still shares a group, source or question with a
    published record of another split."""
    ctx = Ctx(raw, caps, max_source_chars, min_source_chars, seed)
    resolved = [cross_corpus_registry.resolve(p) for p in cross_indexes]
    indexes = [cross_corpus.Index(root) for root, _ in resolved]
    adapters = adapters or ADAPTERS
    docs: list[Doc] = []
    for name, adapter in adapters.items():
        if only and name not in only:
            continue
        before = len(docs)
        for doc in adapter(ctx):
            if doc.records:
                docs.append(doc)
        log(f"{name}: {len(docs) - before} sources")
    # 1. Exact duplicates (same family, source, request and target) keep the first copy.
    seen = set()
    for d in docs:
        keep = []
        for r in d.records:
            key = (r["family"], text_hash(r["sources"][0]["text"]), r["consumer"]["context"][0]["content"],
                   json.dumps(r["target"]["value"], sort_keys=True, ensure_ascii=False))
            if key in seen:
                ctx.reject(r["lineage"]["store"], r["id"], "duplicate")
                continue
            seen.add(key)
            keep.append(r)
        d.records = keep
    # 2. Contract, leakage and trainer-loader checks; invalid records are rejected before any pairing.
    for d in docs:
        keep = []
        for r in d.records:
            errors = check(seal(json.loads(json.dumps(r))))
            if errors:
                ctx.reject(r["lineage"]["store"], r["id"], "invalid: " + "; ".join(errors)[:300])
                continue
            keep.append(r)
        d.records = keep
    records = [r for d in docs for r in d.records]
    ids = Counter(r["id"] for r in records)
    dup_ids = [i for i, c in ids.items() if c > 1]
    if dup_ids:
        raise ValueError(f"duplicate record ids: {dup_ids[:5]}")
    # 3. Split closure: groups, internal dedup links and protected hits, then against the published corpora
    #    (cross_corpus.apply drops components that cannot be placed and exact examples already published).
    protected = json.loads(Path(protected_path).read_text()) if protected_path and Path(protected_path).exists() else None
    kept_records, closure = cross_corpus.apply(records, indexes, protected=protected, log=log)
    kept_ids = {r["id"] for r in kept_records}
    for d in docs:
        d.records = [r for r in d.records if r["id"] in kept_ids]
    conflicts = check_closed(kept_records)
    if conflicts:
        raise ValueError(f"split closure failed: {conflicts[:5]}")
    violations = cross_corpus.check(kept_records, indexes)
    if violations:
        raise ValueError(f"cross-corpus closure failed: {violations[:5]}")
    closure["cross_corpus_violations"] = 0
    # 4. Purposes over one source pair up only among surviving records; then sealing (the content hash excludes
    #    split and groups).
    mark_compare(docs)
    records = [r for d in docs for r in d.records]
    kept: list[dict] = []
    for r in records:
        seal(r)
        errors = check(r)
        if errors:  # cannot happen after step 2 unless pairing broke the contract
            raise ValueError(f"{r['id']}: {errors}")
        kept.append(r)
    out.mkdir(parents=True, exist_ok=True)
    by_family: dict[str, list] = defaultdict(list)
    for r in kept:
        by_family[r["family"]].append(r)
    files = []
    for family in sorted(by_family):
        path = out / f"{family}.port-records.jsonl"
        tmp = path.with_name(path.name + ".pending")
        with open(tmp, "w", encoding="utf-8") as stream:
            for r in by_family[family]:
                stream.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(path)
        files.append({"path": path.name, "records": len(by_family[family]), "bytes": path.stat().st_size,
                      "sha256": _file_sha256(path)})

    def count(key):
        c = defaultdict(Counter)
        for r in kept:
            c[key(r)][r["split"]] += 1
        return {k: dict(v) for k, v in sorted(c.items())}

    inputs = []
    for path, info in sorted(ctx.inputs.items()):
        p = Path(path)
        hashed = info.pop("hash", True)
        entry = {"path": path, **info}
        if p.is_file():
            entry |= {"bytes": p.stat().st_size}
            if hashed:
                entry["sha256"] = _file_sha256(p)
            else:  # large inputs (published corpora, SQLite databases): identity by size and mtime
                entry["mtime_ns"] = p.stat().st_mtime_ns
        inputs.append(entry)
    try:
        commit = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        commit = None
    manifest = {
        "schema": "natlang.view-corpus-manifest/1",
        "id": corpus_id,
        "record_version": VERSION,
        "converter": CONVERTER,
        "code_commit": commit,
        "options": {"caps": caps, "max_source_chars": max_source_chars, "min_source_chars": min_source_chars, "seed": seed,
                    "questions_per_source": QUESTIONS_PER_SOURCE, "extracts_per_source": EXTRACTS_PER_SOURCE,
                    "protected_index": str(protected_path) if protected else None,
                    "protected_index_sha256": _file_sha256(Path(protected_path)) if protected else None},
        "records": len(kept),
        "sources": sum(1 for d in docs if d.records),
        "compare_records": sum(1 for r in kept if r["task"] == "compare"),
        "by_artifact": count(lambda r: r["lineage"]["notes"]["artifact"]),
        "by_task": count(lambda r: r["task"]),
        "by_family": count(lambda r: r["family"]),
        "by_dataset": count(lambda r: r["lineage"]["store"]),
        "by_origin": count(lambda r: r["lineage"]["notes"]["target_origin"]),
        "by_split": dict(Counter(r["split"] for r in kept)),
        "licenses": dict(Counter(f"{r['lineage']['store']}:{r['license']['spdx']}" for r in kept)),
        "closure": closure,
        "cross_corpus_indexes": [{"corpus_id": ix.corpus_id, **({"registry_id": rid} if rid else {"path": str(ix.root)}),
                                  "records": ix.meta["records"]} for ix, (_, rid) in zip(indexes, resolved)],
        "license_review": {
            "by_class": dict(Counter(r["lineage"]["notes"]["license_provenance"]["class"] for r in kept)),
            "by_dataset_class": dict(Counter(f"{r['lineage']['store']}:{r['lineage']['notes']['license_provenance']['class']}"
                                             for r in kept)),
        },
        "info": ctx.info,
        "rejected": dict(sorted(ctx.rejected.items())),
        "reject_examples": ctx.reject_examples,
        "files": files,
        "inputs": inputs,
        "admission": {"training_admission": False,
                      "reason": "pending view-stage qualification (TRAINING_RECIPE.md) and license review"},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    return manifest


def filter_by_license(src: Path, out: Path, allow: set[str]) -> dict:
    """Copy a built corpus keeping only records whose `license_provenance.class` is in `allow`. Purpose pairs that
    point at dropped records are removed; a compare record left without pairs becomes consume. Splits are unchanged
    (a subset of a closed corpus stays closed). Writes a manifest with the filter and counts."""
    unknown = allow - set(vx.CLASS_ORDER)
    if unknown:
        raise ValueError(f"unknown licence classes: {sorted(unknown)}")
    records = [json.loads(l) for p in sorted(Path(src).glob("*.port-records.jsonl")) for l in p.open(encoding="utf-8") if l.strip()]
    keep = [r for r in records if r["lineage"]["notes"]["license_provenance"]["class"] in allow]
    ids = {r["id"] for r in keep}
    for r in keep:
        pairs = [i for i in r["contrasts"]["purpose_pairs"] if i in ids]
        r["contrasts"]["purpose_pairs"] = pairs
        if r["task"] == "compare" and not pairs:
            r["task"] = "consume"
        seal(r)
    out.mkdir(parents=True, exist_ok=True)
    by_family: dict[str, list] = defaultdict(list)
    for r in keep:
        by_family[r["family"]].append(r)
    files = []
    for family in sorted(by_family):
        path = out / f"{family}.port-records.jsonl"
        with open(path, "w", encoding="utf-8") as stream:
            for r in by_family[family]:
                stream.write(json.dumps(r, ensure_ascii=False) + "\n")
        files.append({"path": path.name, "records": len(by_family[family]), "sha256": _file_sha256(path)})
    manifest = {"schema": "natlang.view-corpus-license-filter/1", "source": str(src), "allow": sorted(allow),
                "records_in": len(records), "records": len(keep),
                "dropped_by_class": dict(Counter(r["lineage"]["notes"]["license_provenance"]["class"] for r in records
                                                 if r["id"] not in ids)), "files": files,
                "admission": {"training_admission": False, "reason": "a licence-filtered copy inherits the source's hold"}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest
