"""Upstream agent and research trajectories → `continue` port records (S1 §3.2, §4.1).

bgkit's mapped stores (`compaction_*`, `chunks_*`, `tool_slots*`, `code_swe_*`, `web_*`) were built
from these upstream corpora with bgkit's spine, slot and zoom conventions. Records are rebuilt here
from the upstream data instead, normalised to role/content messages by `trajectory.py`.

Each adapter yields `Trajectory` objects; `CORPORA` maps a corpus name to (adapter, windows per
trajectory). Upstream harness prompts that state the task (an OpenHands or OpenCode system prompt,
the terminus-2 instructions) are kept as the task statement: they are the upstream contract the
next message must follow, not bgkit machinery.
"""
from __future__ import annotations

import ast
import glob
import gzip
import json
import re
from pathlib import Path

from .common import Reject
from .records import group_key, strip_markup
from .trajectory import Trajectory, licence, message, split_think, task_hash, tool_call

TRAJ_EXT = Path("/mnt/external/bgkit-data/trajectories_ext")
WEB_SFT = Path("/mnt/external/bgkit-data/web_sft")
CONVERTER = "scripts/neuralese_data/agents.py@1"


def _lineage(store, row, upstream, *, upstream_id=None, revision=None, teacher=None, notes=None, project="upstream"):
    return {"project": project, "store": store, "store_version": None, "row": row, "upstream": upstream,
            "upstream_id": upstream_id, "upstream_revision": revision, "teacher": teacher, "converter": CONVERTER,
            "sha256": "", **({"notes": notes} if notes else {})}


def _teacher(model, provider=None):
    if not model:
        return "unknown"
    return {"model": str(model), "provider": str(provider or "dataset")}


def _openai_messages(raw: list, *, reasoning_key="reasoning_content", parse_calls=None) -> list:
    out = []
    for m in raw:
        role = m.get("role")
        if role == "developer":
            role = "system"
        if role not in ("system", "user", "assistant", "tool"):
            continue
        content = m.get("content")
        reasoning = m.get(reasoning_key)
        if role == "assistant" and reasoning is None:
            reasoning, content = split_think(content)
        calls = m.get("tool_calls")
        if isinstance(calls, str) and parse_calls:
            calls = parse_calls(calls)
        norm_calls = []
        for c in calls or []:
            fn = c.get("function") or {}
            norm_calls.append(tool_call(fn.get("name") or "tool", fn.get("arguments") if fn.get("arguments") is not None else "{}", c.get("id")))
        out.append(message(role, content if isinstance(content, str) else None, reasoning=reasoning,
                           tool_calls=norm_calls or None, name=m.get("name"), tool_call_id=m.get("tool_call_id")))
    return out


def _head(messages: list) -> int:
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            return i
    return len(messages)


def _parquet_groups(path: Path, columns=None):
    import pyarrow.parquet as pq

    f = pq.ParquetFile(str(path))
    offset = 0
    for batch in f.iter_batches(batch_size=64, columns=columns):  # small batches: rows can be very large
        rows = batch.to_pylist()
        for i, row in enumerate(rows):
            yield offset + i, row
        offset += len(rows)


def _limited(it, limit):
    for n, item in enumerate(it):
        if limit is not None and n >= limit:
            return
        yield item


# --- AgentTrove (terminus-2 terminal sessions) ---------------------------------------------------

_EPISODE_RE = re.compile(r"(\d+)$")
_COPY_RE = re.compile(r"_copy\d+$")


def agenttrove(limit=None):
    """Raw AgentTrove shards; the last episode of each trial."""
    files = sorted((TRAJ_EXT / "agenttrove" / "data").glob("train-*.parquet"))

    def gen():
        for path in files:
            last: dict = {}
            for i, row in _parquet_groups(path, ["trial_name", "episode"]):
                trial = row["trial_name"] or f"row{i}"
                m = _EPISODE_RE.search(str(row["episode"] or "0"))
                ep = int(m.group(1)) if m else 0
                if trial not in last or ep >= last[trial][0]:
                    last[trial] = (ep, i)
            keep = {i for _, i in last.values()}
            cols = ["conversations", "agent", "model", "task", "episode", "trial_name", "model_provider",
                    "original_source", "original_teacher", "result", "trace_source"]
            for i, row in _parquet_groups(path, cols):
                if i not in keep:
                    continue
                yield f"{path.name}:{i}", row

    for rid, row in _limited(gen(), limit):
        msgs = _openai_messages(row["conversations"] or [])
        if not msgs:
            continue
        head = _head(msgs)
        source_name = row.get("original_source") or "unknown"
        task = _COPY_RE.sub("", str(row.get("task") or "")) or task_hash(msgs[0].get("content") or "")
        trial = row.get("trial_name") or rid
        yield rid, Trajectory(
            id=f"upstream:agenttrove:{re.sub(r'[^A-Za-z0-9_.-]', '_', trial)}",
            family="trajectory_continuation_terminal", messages=msgs, head=head, noun="terminal session",
            lineage=_lineage("open-thoughts/AgentTrove", rid, "open-thoughts/AgentTrove", upstream_id=trial,
                             teacher=_teacher(row.get("original_teacher"), row.get("model_provider")),
                             notes={"original_source": source_name, "episode": row.get("episode"),
                                    "agent": row.get("agent"), "trace_source": row.get("trace_source")}),
            license=licence("Apache-2.0", False, "AgentTrove card licence; task material from the listed source datasets."),
            split="train",
            split_groups=[group_key("agenttrove-task", f"{source_name}:{task}"),
                          group_key("task", task_hash(next((m.get("content") or "" for m in msgs if m["role"] == "user"), task)))],
            outcome={"label": "teacher", "checked": None, "details": {"result": row.get("result")}},
        )


# --- OpenResearcher (harmony format, all seeds) ---------------------------------------------------

def _harmony_text(content) -> str:
    parts = []
    for c in content or []:
        if c.get("text"):
            parts.append(c["text"])
    return "\n".join(parts)


def _harmony_tools(system_content) -> list:
    tools = []
    for c in system_content or []:
        browser = (c.get("tools") or {}).get("browser") or {}
        for t in browser.get("tools") or []:
            params = _prune(t.get("parameters") or {})
            tools.append({"type": "function", "function": {"name": f"browser.{t.get('name')}",
                                                           "description": t.get("description") or "", "parameters": params}})
    return tools


def _prune(value):
    if isinstance(value, dict):
        return {k: _prune(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_prune(v) for v in value]
    return value


def harmony_messages(raw: list) -> tuple[list, list]:
    """Harmony (system/developer/user/assistant channels/tool) → messages and tool schemas."""
    out, tools, pending_reasoning = [], [], []
    for m in raw:
        role, channel, recipient = m.get("role"), m.get("channel"), m.get("recipient")
        text = _harmony_text(m.get("content"))
        if role == "system":
            tools = _harmony_tools(m.get("content"))
            continue
        if role == "developer":
            out.append(message("system", text))
        elif role == "user":
            out.append(message("user", text))
        elif role == "assistant":
            if recipient and recipient != "assistant":
                out.append(message("assistant", None, reasoning="\n\n".join(pending_reasoning) or None,
                                   tool_calls=[tool_call(recipient, text)]))
                pending_reasoning = []
            elif channel == "final":
                out.append(message("assistant", text, reasoning="\n\n".join(pending_reasoning) or None))
                pending_reasoning = []
            else:
                pending_reasoning.append(text)
        elif role == "tool":
            out.append(message("tool", text, name=m.get("name")))
    if pending_reasoning:
        out.append(message("assistant", None, reasoning="\n\n".join(pending_reasoning)))
    return out, tools


def openresearcher(limit=None):
    files = sorted((WEB_SFT / "OpenResearcher__OpenResearcher-Dataset").glob("seed_*/train-*.parquet"))

    def gen():
        for path in files:
            seed = path.parent.name
            for i, row in _parquet_groups(path, ["qid", "question", "answer", "messages", "status"]):
                yield f"{seed}/{path.name}:{i}", seed, row

    for rid, seed, row in _limited(gen(), limit):
        msgs, tools = harmony_messages(row["messages"] or [])
        if not msgs:
            continue
        question = strip_markup(row.get("question") or "")
        yield rid, Trajectory(
            id=f"upstream:openresearcher:{seed}:{row.get('qid')}",
            family="trajectory_continuation_research", messages=msgs, head=_head(msgs), noun="research session",
            lineage=_lineage("OpenResearcher/OpenResearcher-Dataset", rid, "OpenResearcher/OpenResearcher-Dataset",
                             upstream_id=f"{seed}:{row.get('qid')}", teacher="unknown",
                             notes={"question": question, "reference_answer": row.get("answer"),
                                    "format": "harmony (gpt-oss family)"}),
            license=licence("MIT", False, "OpenResearcher dataset card: MIT."),
            split="train",
            split_groups=[group_key("question", task_hash(question))],
            outcome={"label": "teacher", "checked": None, "details": {"status": row.get("status")}},
            tools=tools or None,
        )


# --- Nemotron-SFT-SWE v3.5 (OpenCode harness) -----------------------------------------------------

def _py_calls(text: str):
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return []
    return value if isinstance(value, list) else []


def nemotron_swe(limit=None):
    path = TRAJ_EXT / "nemotron_sft_swe_v35_filtered" / "nemotron_sft_swe_v35.jsonl.gz"

    def gen():
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for i, line in enumerate(stream):
                if line.strip():
                    yield i, json.loads(line)

    for i, row in _limited(gen(), limit):
        msgs = _openai_messages(row.get("messages") or [], parse_calls=_py_calls)
        if not msgs:
            continue
        head = _head(msgs)
        issue = next((m.get("content") or "" for m in msgs[:head] if m["role"] == "user"), "")
        tools = [{"type": "function", "function": {"name": t.get("name"), "description": t.get("description") or "",
                                                   "parameters": _prune(t.get("parameters") or {})}}
                 for t in row.get("tools") or [] if isinstance(t, dict)]
        yield i, Trajectory(
            id=f"upstream:nemotron-sft-swe-v3.5:{row.get('uuid') or i}",
            family="trajectory_continuation_swe", messages=msgs, head=head, noun="coding session",
            lineage=_lineage("nvidia/Nemotron-SFT-SWE-v3.5", i, "nvidia/Nemotron-SFT-SWE-v3.5", upstream_id=row.get("uuid"),
                             teacher="unknown", notes={"harness": "OpenCode", "source_file": row.get("source_file")}),
            license=licence("CC-BY-4.0", False, "Card: CC BY 4.0; embedded repository code under each repository's licence."),
            split="train", split_groups=[group_key("task", task_hash(issue))],
            outcome={"label": "teacher", "checked": None}, tools=tools or None,
        )


# --- Open-SWE-Traces (TypeScript/JavaScript, thinking) -------------------------------------------

def _bench_ids():
    path = TRAJ_EXT / "bench_dedupe" / "bench_keys.json"
    if not path.exists():
        return set()
    keys = json.loads(path.read_text()).get("keys", [])
    return {k[1] for k in keys if isinstance(k, list) and len(k) >= 2 and k[0] == "id"}


def open_swe_traces(limit=None):
    files = sorted((TRAJ_EXT / "open_swe_traces_ts_think").glob("*.jsonl.gz"))
    bench = _bench_ids()

    def gen():
        for path in files:
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                for i, line in enumerate(stream):
                    if line.strip():
                        yield f"{path.name}:{i}", json.loads(line)

    for rid, row in _limited(gen(), limit):
        msgs = _openai_messages(row.get("messages") or [])
        if not msgs:
            continue
        tools = []
        for t in row.get("tools") or []:
            try:
                tools.append(_prune(json.loads(t) if isinstance(t, str) else t))
            except json.JSONDecodeError:
                pass
        instance, repo = row.get("instance_id"), row.get("repo")
        if not instance or not repo:
            raise Reject("missing-group: instance or repo")
        meta = row.get("metadata") or {}
        teacher = (meta.get("teacher_model") or {}).get("name")
        resolved = bool(row.get("resolved"))
        protected = instance in bench
        yield rid, Trajectory(
            id=f"upstream:open-swe-traces:{row.get('trajectory_id') or rid}",
            family="trajectory_continuation_swe", messages=msgs, head=_head(msgs), noun="coding session",
            lineage=_lineage("nvidia/Open-SWE-Traces", rid, row.get("source") or "nvidia/Open-SWE-Traces",
                             upstream_id=row.get("trajectory_id"), teacher=_teacher(teacher),
                             notes={"instance_id": instance, "benchmark_protected": protected,
                                    "language": row.get("language"), "repository_license": row.get("license")}),
            license=licence("CC-BY-4.0", False, f"Card: CC BY 4.0; repository {repo} under {row.get('license') or 'its own licence'}."),
            split="test" if protected else "train",
            split_groups=[group_key("repo", repo), group_key("swe-instance", instance)],
            outcome=({"label": "checked", "checked": "swe-rebench-tests"} if resolved else {"label": "teacher", "checked": None})
                    | {"details": {"resolved": resolved}},
            tools=tools or None,
        )


# --- OpenSeeker + MiroVerse deep-research SFT ----------------------------------------------------

def openseeker_miroverse(limit=None):
    base = WEB_SFT / "Zephyr271828__openseeker-miroverse-mix-full"

    def gen():
        for split, name in (("train", "train.parquet"), ("validation", "validation.parquet")):
            for i, row in _parquet_groups(base / name):
                yield split, f"{name}:{i}", row

    for split, rid, row in _limited(gen(), limit):
        msgs = _openai_messages(row.get("messages") or [])
        if not msgs:
            continue
        src = str(row.get("source") or "")
        miro = src.lower().startswith("miroverse")
        lic = (licence("CC-BY-NC-4.0", True, "MiroVerse-v0.1 (gated, CC BY-NC 4.0); answers extracted by Qwen2.5-32B-Instruct.")
               if miro else licence("LicenseRef-openseeker", False, "OpenSeeker v1 SFT; card licence 'other', upstream terms apply."))
        question = strip_markup(row.get("question") or "")
        yield rid, Trajectory(
            id=f"upstream:openseeker-miroverse:{split}:{rid.split(':')[1]}",
            family="trajectory_continuation_research", messages=msgs, head=_head(msgs), noun="research session",
            lineage=_lineage("Zephyr271828/openseeker-miroverse-mix-full", rid, "Zephyr271828/openseeker-miroverse-mix-full",
                             teacher="unknown", notes={"question": question, "reference_answer": row.get("answer"), "source": src}),
            license=lic, split=split, split_groups=[group_key("question", task_hash(question))],
            outcome={"label": "teacher", "checked": None}, tools=[_prune(t) for t in row.get("tools") or []] or None,
        )


# --- WebShaper (gpt-oss-120b harmony transcripts) -------------------------------------------------

_HARMONY_SEG_RE = re.compile(r"<\|start\|>assistant")
_HARMONY_TOKEN_RE = re.compile(r"<\|(?:start|end|message|channel|constrain|call|return)\|>")
_HARMONY_HDR_RE = re.compile(r"^\s*<\|channel\|>(?P<channel>\w+)(?:\s+to=(?P<to>[\w.]+))?[^<]*?(?:<\|constrain\|>\w+)?<\|message\|>", re.S)


def _webshaper_messages(conversation: list, question: str) -> list:
    out = [message("user", f"Question: {question}")]
    pending = []
    for m in conversation:
        if m.get("role") == "tool":
            out.append(message("tool", m.get("content") or "", name="browser"))
            continue
        for seg in _HARMONY_SEG_RE.split(m.get("content") or ""):
            seg = seg.replace("<|end|>", "").replace("<|call|>", "").replace("<|return|>", "")
            h = _HARMONY_HDR_RE.match(seg)
            if not h:
                if seg.strip():
                    pending.append(_HARMONY_TOKEN_RE.sub("", strip_markup(seg)))
                continue
            body = _HARMONY_TOKEN_RE.sub("", seg[h.end():])
            if h.group("to"):
                out.append(message("assistant", None, reasoning="\n\n".join(pending) or None,
                                   tool_calls=[tool_call(h.group("to"), body.strip())]))
                pending = []
            elif h.group("channel") == "final":
                out.append(message("assistant", body, reasoning="\n\n".join(pending) or None))
                pending = []
            else:
                pending.append(body)
    if pending:
        out.append(message("assistant", None, reasoning="\n\n".join(pending)))
    return out


def webshaper(limit=None):
    path = WEB_SFT / "rl-rag__webshaper-gpt-oss-120b-260222" / "data" / "train-00000-of-00001.parquet"
    for i, row in _limited(_parquet_groups(path), limit):
        try:
            conversation = json.loads(row.get("conversation") or "[]")
        except json.JSONDecodeError:
            continue
        question = strip_markup(row.get("question") or "")
        msgs = _webshaper_messages(conversation, question)
        contaminated = bool(row.get("contaminated"))
        yield i, Trajectory(
            id=f"upstream:webshaper-gpt-oss:{row.get('qid')}:{row.get('traj_idx')}",
            family="trajectory_continuation_research", messages=msgs, head=1, noun="research session",
            lineage=_lineage("rl-rag/webshaper-gpt-oss-120b-260222", i, "Alibaba-NLP/WebShaper",
                             upstream_id=f"{row.get('qid')}:{row.get('traj_idx')}",
                             teacher={"model": "openai/gpt-oss-120b", "provider": "rl-rag (dataset)"},
                             notes={"question": question, "reference_answer": row.get("reference_answer"),
                                    "contaminated": contaminated}),
            license=licence("LicenseRef-unverified", False, "No licence in the local card; questions from Alibaba-NLP/WebShaper."),
            split="test" if contaminated else "train",
            split_groups=[group_key("question", task_hash(question))],
            outcome=({"label": "checked", "checked": "judge-vs-reference"} if row.get("correct") else
                     {"label": "failed", "checked": "judge-vs-reference"})
                    | {"details": {"correct": bool(row.get("correct")), "status": row.get("status")}},
        )


# --- Search-R1 SFT ---------------------------------------------------------------------------------

_QUESTION_RE = re.compile(r"Question:\s*(.+?)\s*$", re.S)


def search_r1(limit=None):
    files = sorted((WEB_SFT / "ThornZ__Search-R1-SFT").glob("*.json"))

    def gen():
        for path in files:
            data = json.loads(path.read_text())
            for i, row in enumerate(data):
                yield f"{path.stem}:{i}", path.stem, row

    for rid, variant, row in _limited(gen(), limit):
        msgs = _openai_messages(row.get("messages") or [])
        if not msgs:
            continue
        head = _head(msgs)
        first_user = next((m.get("content") or "" for m in msgs[:head] if m["role"] == "user"), "")
        q = _QUESTION_RE.search(first_user)
        question = q.group(1) if q else first_user[-500:]
        yield rid, Trajectory(
            id=f"upstream:search-r1-sft:{variant}:{rid.split(':')[1]}",
            family="trajectory_continuation_search", messages=msgs, head=head, noun="search session",
            lineage=_lineage("ThornZ/Search-R1-SFT", rid, "ThornZ/Search-R1-SFT", teacher="unknown",
                             notes={"question": question, "variant": variant}),
            license=licence("LicenseRef-unverified", False, "No card downloaded locally; Search-R1 retrieval over a Wikipedia dump."),
            split="train", split_groups=[group_key("question", task_hash(question))],
            outcome={"label": "teacher", "checked": None},
        )


CORPORA = {
    # name: (adapter, windows per trajectory)
    "agenttrove": (agenttrove, 1),
    "openresearcher": (openresearcher, 1),
    "nemotron-swe": (nemotron_swe, 2),
    "open-swe-traces": (open_swe_traces, 2),
    "openseeker-miroverse": (openseeker_miroverse, 2),
    "webshaper": (webshaper, 2),
    "search-r1": (search_r1, 1),
}
