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
from .splits import check_closed, close

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training" / "neuralese"))
from natlang_neuralese.data.records import RecordError, parse_record  # noqa: E402

CONVERTER = "scripts/neuralese_data/view_corpus.py@1"
RESULT_TYPE = "Neuralese<string>"
MAX_SOURCE_CHARS = 32_000
MIN_SOURCE_CHARS = 200
PROTECTED_INDEX = Path("/mnt/external/natlang-development-data/data/neuralese/protected/bgkit-benchmarks.protected.json")
HF = Path.home() / ".cache/huggingface/hub"
NEBIUS = Path("/home/werg/natlang/data/neuralese/corpora/nebius-swe-rebench-openhands-trajectories-20261009-v1/trajectories.parquet")
XLAM = Path("/mnt/external/sdkb-archive/raw/agentic-20260927/xlam-function-calling-60k/xlam_function_calling_60k.json")
REPOS = Path("/mnt/external/bgkit-data/repos")

# Documents (or functions, pages, tables, files, tool outputs) per source in the default balanced slice.
DEFAULT_CAPS = {
    "cnn_dailymail": 500, "squad": 220, "quality": 40, "qmsum": 160,
    "codesearchnet": 900,
    "websrc": 700,
    "xlam": 700, "repo_configs": 700,
    "wtq": 450, "fetaqa": 700,
    "swe_tool_outputs": 1100,
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


class Ctx:
    def __init__(self, raw: Path, caps: dict, max_chars: int, min_chars: int, seed: int):
        self.raw, self.caps, self.max_chars, self.min_chars, self.seed = raw, caps, max_chars, min_chars, seed
        self.rejected: Counter = Counter()
        self.reject_examples: dict = {}
        self.inputs: dict[str, dict] = {}

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
                 groups: list[str], source_role: str | None = None, notes=None):
        self.ctx, self.dataset, self.key, self.artifact, self.text = ctx, dataset, _safe(key), artifact, text
        self.title, self.meta, self.refs = title, dict(meta or {}), list(refs or [])
        self.upstream, self.upstream_id, self.revision, self.store_version = upstream, upstream_id, revision, store_version
        self.row, self.lic, self.split, self.groups = row if row is not None else str(key), lic, split, groups
        self.role = source_role or ROLES[artifact]
        self.notes = dict(notes or {})
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
                      upstream_id=row["id"], revision=revision, store_version="3.0.0", row=row["__row"], lic=lic,
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
    for context in contexts[: ctx.caps["squad"]]:
        rows = by_context[context]
        first = rows[0][1]
        title = first["title"]
        doc = Doc(ctx, dataset="squad", key=first["id"], artifact="prose", text=context, title=title.replace("_", " "),
                  upstream="hf:rajpurkar/squad", upstream_id=first["id"], revision=revision, row=rows[0][0], lic=lic,
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
                      notes={"article_license": (qs[0][1].get("license") or "")[:200]})
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
                        store_version="ALL")
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

def codesearchnet(ctx: Ctx):
    """Python functions: docstring summary (docstring removed from the source), parameters, called names, and
    reconstruction of the whole function."""
    meta = _raw_meta(ctx, "codesearchnet")
    import pyarrow.parquet as pq

    sizes, paths = {}, {}
    for split in ("train", "validation", "test"):
        p = ctx.raw / "codesearchnet" / f"python-{split}.parquet"
        ctx.use(p, dataset="codesearchnet", upstream="hf:code-search-net/code_search_net", revision=meta["revision"])
        paths[split], sizes[split] = p, pq.ParquetFile(str(p)).metadata.num_rows
    picks = _upstream_sample(ctx, "codesearchnet", sizes, ctx.caps["codesearchnet"])
    cols = ["repository_name", "func_path_in_repository", "func_name", "whole_func_string", "func_code_url"]
    for split, path in paths.items():
        want = max(1, round(ctx.caps["codesearchnet"] * UPSTREAM_SHARE[split]))
        made = 0
        for row in _read_rows(path, picks[split], cols):
            if made >= want:
                break
            repo = row["repository_name"]
            key = f"{repo}/{row['func_path_in_repository']}:{row['func_name']}"
            facts = vx.python_facts(row["whole_func_string"])
            if facts is None or not facts["docstring"]:
                ctx.reject("codesearchnet", key, "no-docstring-or-parse")
                continue
            summary = vx.summary_sentence(facts["docstring"])
            if summary is None:
                ctx.reject("codesearchnet", key, "docstring-not-prose")
                continue
            code = vx.without_docstring(facts)
            if not ctx.fits(code):
                ctx.reject("codesearchnet", key, "length")
                continue
            lic = license_("LicenseRef-repository-content", False,
                           f"Function from github.com/{repo} under that repository's licence (CodeSearchNet kept only "
                           "repositories whose licence permits redistribution).")
            path_ref = row["func_path_in_repository"]
            refs = [{"text": path_ref, "kind": "path"}, {"text": row["func_name"], "kind": "identifier"}]
            common = dict(upstream="hf:code-search-net/code_search_net", upstream_id=row.get("func_code_url"),
                          revision=meta["revision"], store_version="python", row=row["__row"], lic=lic,
                          split=SPLIT_MAP[split], groups=[group_key("repo", repo)])
            doc = Doc(ctx, dataset="codesearchnet", key=key, artifact="code", text=code, title=f"{path_ref}:{row['func_name']}",
                      refs=refs, meta={"format": "python function", "repository": repo, "docstring_removed": True}, **common)
            name = facts["name"]
            doc.summary(f"Describe in one sentence what the function `{name}` does.", summary, origin="upstream-docstring",
                        general="Keep what a description of the function needs.")
            n = 0
            if facts["params"]:
                doc.extract(f"List the parameters of `{name}` in order, one per line.", "\n".join(facts["params"]), n,
                            method="python-ast-params")
                n += 1
            if 1 <= len(facts["calls"]) <= 15:
                doc.extract(f"List every function or method that `{name}` calls, once each, in order of first appearance, "
                            "one per line.", "\n".join(facts["calls"]), n, method="python-ast-calls")
            yield doc
            whole = row["whole_func_string"]
            if ctx.fits(whole):
                full = Doc(ctx, dataset="codesearchnet", key=key + "#whole", artifact="code", text=whole,
                           title=f"{path_ref}:{row['func_name']}", refs=refs,
                           meta={"format": "python function", "repository": repo}, **common)
                full.reconstruct()
                yield full
            made += 1


# ---------------------------------------------------------------------------------------------- HTML

def websrc(ctx: Ctx):
    """WebSRC pages (dataset tag ids and scripts removed): the dataset's questions, and CSS-selector extraction."""
    meta = _raw_meta(ctx, "websrc")
    lic = license_("CC-BY-4.0", False, "WebSRC v1.0 (X-LANCE/WebSRC_v1.0); page content from the source websites.")
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
                  split=carve(group), groups=[group], refs=exact_refs_from(html, limit=16),
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
        doc = Doc(ctx, dataset="xlam", key=r["id"], artifact="data", text=tools_text, title="tool schemas (JSON)",
                  upstream="hf:Salesforce/xlam-function-calling-60k", upstream_id=str(r["id"]), revision="26d14eb", row=i,
                  lic=lic, split=carve(group), groups=[group], source_role="schema",
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
            doc = Doc(ctx, dataset="repo_configs", key=f"{name}:{path}", artifact="data", text=text, title=path,
                      upstream=f"github:{name}", upstream_id=path, revision=head, row=f"{name}:{path}", lic=lic,
                      split=carve(group_key("repo", name)), groups=[group_key("repo", name)],
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
                  row=table_path, lic=lic, split=split if split == "test" else carve(group), groups=[group],
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
                      lic=lic, split=split, groups=[group],
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
                    groups_ = [f"swe-rebench-repo:{t.repo}", f"swe-rebench-instance:{t.instance_id}"]
                    doc = Doc(ctx, dataset="swe_tool_outputs", key=f"{t.id}:{call_id}", artifact="log", text=text,
                              title=f"bash: {command[:200]}", upstream="hf:nebius/SWE-rebench-openhands-trajectories",
                              upstream_id=t.id, revision=None, row=index, lic=lic, split=split_of(t.repo, 5), groups=groups_,
                              refs=exact_refs_from(text, limit=24),
                              meta={"format": "bash output", "tool": "bash", "command": command[:2000]},
                              notes={"instance_id": t.instance_id})
                    for m, (request, answer, method) in enumerate(extracts):
                        doc.extract(request, answer, m, method=method)
                    doc.reconstruct()
                    taken += 1
                    made += 1
                    yield doc


# ---------------------------------------------------------------------------------------------- build

ADAPTERS = {
    "cnn_dailymail": cnn_dailymail, "squad": squad, "quality": quality, "qmsum": qmsum,
    "codesearchnet": codesearchnet,
    "websrc": websrc,
    "xlam": xlam, "repo_configs": repo_configs,
    "wtq": wtq, "fetaqa": fetaqa,
    "swe_tool_outputs": swe_tool_outputs,
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
          only: list[str] | None = None, adapters: dict | None = None, log=print) -> dict:
    ctx = Ctx(raw, caps, max_source_chars, min_source_chars, seed)
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
    # 3. Purposes over one source pair up only among surviving records.
    compare = mark_compare(docs)
    records = [r for d in docs for r in d.records]
    ids = Counter(r["id"] for r in records)
    dup_ids = [i for i, c in ids.items() if c > 1]
    if dup_ids:
        raise ValueError(f"duplicate record ids: {dup_ids[:5]}")
    # 4. Split closure (with protected hits), then sealing; the content hash excludes split and groups.
    protected = json.loads(Path(protected_path).read_text()) if protected_path and Path(protected_path).exists() else None
    closure = close(records, protected=protected)
    conflicts = check_closed(records)
    if conflicts:
        raise ValueError(f"split closure failed: {conflicts[:5]}")
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
        entry = {"path": path, **info}
        if p.is_file():
            entry |= {"bytes": p.stat().st_size, "sha256": _file_sha256(p)}
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
        "rejected": dict(sorted(ctx.rejected.items())),
        "reject_examples": ctx.reject_examples,
        "files": files,
        "inputs": inputs,
        "admission": {"training_admission": False,
                      "reason": "pending view-stage qualification (TRAINING_RECIPE.md) and license review"},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    return manifest
