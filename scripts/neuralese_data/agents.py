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
from .records import group_key, strip_markup, source
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
        elif role == "function":
            role = "tool"  # legacy OpenAI named function observation
        if role not in ("system", "user", "assistant", "tool"):
            continue
        content = m.get("content")
        reasoning = m.get(reasoning_key)
        if role == "assistant" and reasoning is None:
            reasoning, content = split_think(content)
        # Retain structured source observations instead of turning them into
        # empty messages. Source-specific adapters can interpret known channels;
        # unrecognized blocks remain exact JSON text, never fabricated calls.
        if content is not None and not isinstance(content,str):
            content=json.dumps(content,ensure_ascii=False)
        if reasoning is not None and not isinstance(reasoning,str):
            reasoning=json.dumps(reasoning,ensure_ascii=False)
        calls = m.get("tool_calls")
        legacy=m.get("function_call")
        if legacy is not None:
            if role!="assistant" or calls or not isinstance(legacy,dict) or not isinstance(legacy.get("name"),str):
                raise Reject("unsupported or conflicting original legacy function call")
            calls=[{"id":legacy.get("id"),"function":legacy}]
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

def _harmony_text(content) -> str | None:
    """Known source text stays verbatim; other observed structure stays JSON."""
    if content is None or isinstance(content,str):return content
    if isinstance(content,list):return "\n".join(_harmony_text(part) or '' for part in content)
    if isinstance(content,dict):
        present={key:value for key,value in content.items() if value is not None}
        if set(present)<={'type','text'} and present.get('type') in (None,'text','input_text','output_text'):
            if not isinstance(present.get('text'),str):raise Reject('structured text block has no original string text')
            return present['text']
    return json.dumps(content,ensure_ascii=False)


def _harmony_tools(system_content) -> list:
    tools=[]
    for block in system_content if isinstance(system_content,list) else []:
        if not isinstance(block,dict) or not isinstance(block.get('tools'),dict):continue
        for namespace,declaration in block['tools'].items():
            for fn in declaration.get('tools') or []:
                fn=_prune(fn)
                tools.append({'type':'function','function':{**fn,'name':namespace+'.'+fn['name']}})
    return tools


def _prune(value):
    if isinstance(value, dict):
        return {k: _prune(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_prune(v) for v in value]
    return value


def harmony_messages(raw: list) -> tuple[list,list]:
    """Preserve each original Harmony context/channel/dispatch/observation."""
    prepared=[];tools=[];pending=[]
    for index,original in enumerate(raw):
        item=dict(original);role=item.get('role');content=_harmony_text(item.get('content'))
        tools.extend(_harmony_tools(item.get('content')))
        item['content']=content
        if role=='assistant' and item.get('recipient') and item['recipient']!='assistant':
            if item.get('tool_calls'):raise Reject('research dispatch has two conflicting call representations')
            if not isinstance(content,str):raise Reject('research dispatch lacks original arguments')
            ident=f'original-call-{index}'
            item['tool_calls']=[tool_call(item['recipient'],content,ident)]
            item['content']=None;pending.append((ident,item['recipient']))
        elif role=='assistant' and item.get('channel')=='analysis':
            if item.get('reasoning_content') not in (None,content):raise Reject('research analysis conflicts with original reasoning')
            item['reasoning_content']=content;item['content']=None
        elif role=='tool':
            if not pending:raise Reject('research tool observation has no original preceding dispatch')
            ident,name=pending.pop(0)
            if item.get('tool_call_id') not in (None,ident) or item.get('name') not in (None,name):
                raise Reject('research tool observation conflicts with its original preceding dispatch')
            item['tool_call_id']=ident;item['name']=name
        elif role not in ('system','developer','user','assistant'):
            raise Reject('unsupported research role retained in original source')
        prepared.append(item)
    return _openai_messages(prepared),tools


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
                    | {"details": {
                        "resolved": resolved,
                        "checker_evidence": {
                            "origin": "upstream_resolved_status",
                            "raw_status_field": "resolved",
                            "raw_status": row.get("resolved"),
                            "local_execution": False,
                        },
                    }},
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


# Pinned expansion snapshots. Raw files and acquisition.json retain the complete
# original graphs; these views use the existing causal SessionHistory handoff.
EXPANSION_ROOT = Path("data/neuralese/external-expansion-20261014")
_JSON_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def expansion_messages(raw):
    """Lift JSON call tags into structured calls; preserve unrecognized dialects as text."""
    prepared = []
    for original in raw:
        item = dict(original)
        if 'reasoning_content' not in item and item.get('reasoning') is not None:
            item['reasoning_content'] = item['reasoning']
        if item.get('tool_calls'):
            item['tool_calls'] = [call if 'function' in call else
                                  {'id': call.get('id'), 'type': 'function',
                                   'function': {'name': call['name'], 'arguments': call.get('arguments', '{}')}}
                                  for call in item['tool_calls']]
        if item.get('role') == 'function' and (item.get('name') or item.get('tool_call_id')):
            item['role']='tool'  # named feedback is never a dispatch inferred from its JSON body
        if item.get('role') == 'function':
            # Evidence-subagent Qwen transcripts use this role for dispatches,
            # while legacy OpenAI transcripts use it for observations.
            reasoning, body = split_think(item.get('content'))
            try:
                call = json.loads(body) if isinstance(body, str) else None
            except ValueError:
                call = None
            if isinstance(call, dict) and isinstance(call.get('name'), str) and 'arguments' in call:
                item.update(role='assistant', content=None, reasoning_content=reasoning,
                            tool_calls=[tool_call(call['name'], call['arguments'], call.get('id'))])
            else:
                item['role'] = 'tool'
        content = item.get("content")
        if isinstance(content, list):
            # Keep non-text blocks too: silently dropping them destroys observations.
            content = json.dumps(content, ensure_ascii=False)
            item["content"] = content
        if item.get("role") == "assistant" and isinstance(content, str) and not item.get("tool_calls"):
            calls, consumed = [], []
            for match in _JSON_CALL.finditer(content):
                try:
                    value = json.loads(match.group(1))
                except ValueError:
                    continue
                if isinstance(value.get("name"), str) and "arguments" in value:
                    calls.append(tool_call(value["name"], value["arguments"], value.get("id")))
                    consumed.append(match.span())
            if calls:
                for start, end in reversed(consumed):
                    content = content[:start] + content[end:]
                item["content"], item["tool_calls"] = content, calls
        prepared.append(item)
    return _openai_messages(prepared)


def expansion_trajectory(ref, receipt, rid, row):
    """A task-grouped external teacher view; reported outcomes are not execution certificates."""
    metadata = row.get("metadata") or {}
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    msgs = expansion_messages(row.get("messages") or row.get("conversations") or [])
    if not msgs:
        raise Reject("empty-trajectory")
    request = next((m.get("content") or "" for m in msgs if m["role"] == "user"), "")
    task_group_text = row.get('_expansion_task_group_text', request)
    trajectory_id = metadata.get("trajectory_id") or metadata.get("source_trajectory_id") or row.get("task_id")
    source = str(metadata.get("source") or "")
    if not trajectory_id and source:
        trajectory_id = source.rsplit("/", 1)[0]
    trajectory_id = str(trajectory_id or task_hash(request))
    session_id = str(metadata.get("session_id") or metadata.get("agent_id") or rid)
    repo = receipt["repo"]
    card = receipt.get("card") or {}
    license_name = card.get("license") or "LicenseRef-upstream-unspecified"
    if not isinstance(license_name, str):
        license_name = "LicenseRef-upstream-custom"
    research = ref in ("D01", "D02", "D10", "D21", "D31")
    return Trajectory(
        id=f"upstream:expansion:{ref}:{trajectory_id}:{session_id}",
        family="trajectory_continuation_research" if research else "trajectory_continuation_coding",
        messages=msgs, head=_head(msgs), noun="research session" if research else "coding session",
        lineage=_lineage(repo, rid, repo, upstream_id=session_id, revision=receipt["revision"],
                         teacher="unknown", notes={"metadata": metadata, "communication": "derived-history",
                            "observed_graph_snapshot": str(EXPANSION_ROOT / ref),
                            "native_execution_verified": False}),
        license=licence(license_name, "nc" in license_name.lower(), "Pinned upstream card; owner accepts all licenses."),
        split=row.get('_expansion_split', 'train'), split_groups=[group_key(f"expansion-{ref.lower()}-task", trajectory_id),
                                     *([group_key("task", task_hash(task_group_text))] if task_group_text is not None else [])],
        outcome={"label": "teacher", "checked": None},
        tools=json.loads(row["tools"]) if isinstance(row.get("tools"), str) else row.get("tools"),
    )


def _expansion(ref, limit=None):
    root = EXPANSION_ROOT / ref
    receipt = json.loads((root / "acquisition.json").read_text())
    def rows():
        for path in sorted(root.rglob("*.jsonl")):
            with path.open() as stream:
                for i, line in enumerate(stream):
                    if line.strip():
                        yield f"{path.relative_to(root)}:{i}", json.loads(line)
        for path in sorted(root.rglob("*.parquet")):
            for i, row in _parquet_groups(path):
                if ref == 'D10' and path.name == 'eval.parquet':
                    row = {**row, '_expansion_split': 'test'}
                yield f"{path.relative_to(root)}:{i}", row
    for rid, row in _limited(rows(), limit):
        if ref == "D02":
            # Nested workers are original sessions, not synthesized by the adapter.
            # Keep their common root identity even when each worker asks a different question.
            root_id = task_hash(row["question"])
            parent = {**row, "metadata": {"trajectory_id": root_id, "session_id": "planner",
                      "source": row.get("source"), "nested_workers": len(row.get("subagents") or [])}}
            yield rid, expansion_trajectory(ref, receipt, rid, parent)
            for index, worker in enumerate(row.get("subagents") or []):
                child_id = f"{rid}:worker:{index}"
                child = {**worker, "metadata": {"trajectory_id": root_id, "session_id": f"worker:{index}",
                         "parent_session_id": "planner", "original_worker_index": index,
                         "relationship": "observed-nested-session"}}
                yield child_id, expansion_trajectory(ref, receipt, child_id, child)
        else:
            yield rid, expansion_trajectory(ref, receipt, rid, row)


def wideseek(limit=None):
    yield from _expansion("D01", limit)


def cooperator_dual_policy(limit=None):
    yield from _expansion("D08", limit)


def searchswarm(limit=None):
    yield from _expansion("D02", limit)


def owl_sft(limit=None):
    yield from _expansion("D21", limit)


def deepdive_trajectories(limit=None):
    """Published SFT sessions only; QA/RL rows are not observed tool histories."""
    root = EXPANSION_ROOT / 'D31'
    receipt = json.loads((root / 'acquisition.json').read_text())
    def rows():
        for path in sorted(root.rglob('trajectories_sft-*.parquet')):
            for index, row in _parquet_groups(path):
                yield f'{path.relative_to(root)}:{index}', row
    for rid, row in _limited(rows(), limit):
        question = row.get('question')
        if not isinstance(question, str) or not question.strip():
            raise Reject('missing-deepdive-question')
        metadata = {'trajectory_id': task_hash(question),
                    'session_id': str(row.get('id') if row.get('id') is not None else rid),
                    'source_row_id': row.get('id'), 'question': question,
                    'relationship': 'observed-research-session'}
        trajectory = expansion_trajectory('D31', receipt, rid,
            {**row, 'metadata': metadata, '_expansion_task_group_text': question})
        trajectory.split_groups.append(group_key('question', task_hash(question)))
        yield rid, trajectory


def evidence_subagent(limit=None):
    yield from _expansion('D10', limit)


def openthoughts_agent(limit=None):
    yield from _expansion('D29', limit)


def terminaltraj(limit=None):
    yield from _expansion('D33', limit)


def codeagent_sft(limit=None):
    """Alpaca coding responses as derived typed task handoffs; upstream outcomes remain unknown."""
    root = EXPANSION_ROOT / 'D05'
    receipt = json.loads((root / 'acquisition.json').read_text())
    rows = json.loads((root / 'CodeAgentSFTDataset.json').read_text())
    for index, row in _limited(enumerate(rows), limit):
        task = str(row.get('instruction') or '') + '\n' + str(row.get('input') or '')
        history = row.get('history') or []
        if history:
            task = json.dumps({'previous_history': history}, ensure_ascii=False) + '\n' + task
        target = row.get('output')
        if not isinstance(target, str) or not target.strip():
            raise Reject('empty-codeagent-target')
        converted = {'metadata': {'trajectory_id': task_hash(task), 'session_id': str(index),
                                  'relationship': 'derived-task-background'},
                     'messages': [{'role': 'user', 'content': 'Complete the coding task supplied in the background, following its requested format.'},
                                  {'role': 'assistant', 'content': target}]}
        trajectory = expansion_trajectory('D05', receipt, str(index), converted)
        trajectory.background = [source('document', text=strip_markup(task))]
        trajectory.split_groups = [group_key('expansion-d05-task', task_hash(task)),
                                   group_key('task', task_hash(task))]
        yield index, trajectory


def _literal_function_calls(text):
    """Serialize literal upstream dispatches without evaluating source code."""
    try:
        body = ast.parse(text).body
        calls = []
        for statement in body:
            if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
                return None
            call = statement.value
            parts, node = [], call.func
            while isinstance(node, ast.Attribute):
                parts.append(node.attr)
                node = node.value
            if not isinstance(node, ast.Name) or call.args or any(k.arg is None for k in call.keywords):
                return None
            parts.append(node.id)
            arguments = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
            calls.append(tool_call('.'.join(reversed(parts)), arguments))
        return calls or None
    except (SyntaxError, ValueError, TypeError):
        return None


def _initial_request_background(trajectory):
    """Use the existing typed background handoff for an otherwise history-free task."""
    for msg in trajectory.messages[:trajectory.head]:
        if msg['role'] == 'user' and msg.get('content'):
            trajectory.background.append(source('document', text=msg['content']))
            msg['content'] = 'Complete the coding task supplied in the background, following its requested format.'
            trajectory.lineage['notes']['relationship'] = 'derived-task-background'
            break
    return trajectory


def dolci_tool_use(limit=None):
    root = EXPANSION_ROOT / 'D30'
    receipt = json.loads((root / 'acquisition.json').read_text())
    def rows():
        for path in sorted(root.rglob('*.parquet')):
            for index, row in _parquet_groups(path):
                yield f'{path.relative_to(root)}:{index}', row
    for rid, row in _limited(rows(), limit):
        messages, tools, unparsed = [], [], []
        for original in row.get('messages') or []:
            msg = dict(original)
            schemas = msg.get('functions')
            if schemas:
                for schema in json.loads(schemas) if isinstance(schemas, str) else schemas:
                    if schema not in tools:
                        tools.append(schema)
            if msg.get('role') == 'environment':
                msg['role'] = 'tool'
            dispatch = msg.get('function_calls')
            if dispatch:
                calls = _literal_function_calls(dispatch)
                if calls:
                    msg['tool_calls'] = calls
                else:
                    msg['content'] = '\n'.join(filter(None, [msg.get('content'), dispatch]))
                    unparsed.append(dispatch)
            messages.append(msg)
        normalized = {**row, 'messages':messages, 'tools':tools,
                      'metadata':{'source':row.get('dataset_source'), 'session_id':row.get('id') or rid,
                                  'trajectory_id':row.get('id') or rid}}
        trajectory = expansion_trajectory('D30', receipt, rid, normalized)
        trajectory.lineage['notes']['unparsed_function_calls'] = unparsed
        yield rid, _initial_request_background(trajectory)


def cognitive_kernel(limit=None):
    for rid, trajectory in _expansion('D34', limit):
        trajectory.lineage['notes']['benchmark_source_file'] = rid.split(':', 1)[0]
        yield rid, _initial_request_background(trajectory)


def finding_right_fit(limit=None):
    """Primary scored parent/child histories; viewer parquet is an alternate view."""
    root = EXPANSION_ROOT / 'D06'
    receipt = json.loads((root / 'acquisition.json').read_text())

    def rows():
        for path in sorted((root / 'trajectories').rglob('*.jsonl.gz')):
            with gzip.open(path, 'rt') as stream:
                for index, line in enumerate(stream):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not row.get('trajectory_available') or not row.get('messages'):
                        continue
                    rid = f'{path.relative_to(root)}:{index}'
                    yield rid, row, row, 'parent'
                    for child_index, child in enumerate(row.get('subagents') or []):
                        if child.get('messages'):
                            yield f'{rid}:child:{child_index}', child, row, 'child'

    for rid, row, parent, role in _limited(rows(), limit):
        metadata = {'trajectory_id': task_hash(json.dumps([parent['benchmark'], parent['task_id']])),
                    'session_id': rid, 'benchmark': parent['benchmark'],
                    'task_id': parent['task_id'],
                    'harness': parent.get('harness'), 'harness_version': parent.get('harness_version'),
                    'source': parent.get('source'), 'canary': parent.get('canary'),
                    'parent_reward': parent.get('reward'), 'parent_reward_status': parent.get('reward_status'),
                    'session_role': role, 'observed_child_metadata': {k: v for k, v in row.items() if k != 'messages'} if role == 'child' else None,
                    'harness_events': row.get('harness_events') or row.get('events')}
        trajectory = expansion_trajectory('D06', receipt, rid, {**row, 'metadata': metadata})
        trajectory.lineage['teacher'] = _teacher(parent.get('model_id') or parent.get('model'))
        trajectory.lineage['notes']['score_is_parent_upstream_report_only'] = True
        reward = parent.get('reward')
        if parent.get('reward_status') != 'verified' or not isinstance(reward, (float, int)) or reward <= 0:
            trajectory.outcome['label'] = 'failed'
        yield rid, _initial_request_background(trajectory)


def deepnlp_orchestration(limit=None):
    """Distinct recorded histories of alternative sessions, without invented edges."""
    root = EXPANSION_ROOT / 'D07'
    receipt = json.loads((root / 'acquisition.json').read_text())

    def rows():
        with (root / 'example_deepnlp_multi_agents_202601.json').open() as stream:
            for line_index, line in enumerate(stream):
                if not line.strip():
                    continue
                for task, sessions in json.loads(line).items():
                    for session, value in sessions.items():
                        histories = {}
                        for snapshot_index, snapshot in enumerate(value['function_calls']):
                            key = json.dumps({'messages': snapshot['messages'], 'tools': snapshot['tools']},
                                             sort_keys=True, separators=(',', ':'))
                            histories.setdefault(key, []).append(snapshot_index)
                        for history_index, (key, indexes) in enumerate(histories.items()):
                            row = json.loads(key)
                            row['metadata'] = {'trajectory_id': str(task),
                                               'session_id': f'{session}:{history_index}',
                                               'source_session_id': value.get('session_id'),
                                               'trace_id': value.get('trace_id'),
                                               'snapshot_indexes': indexes,
                                               'selected_calls': [value['function_calls'][i].get('tool_calls') for i in indexes]}
                            yield f'{line_index}:{task}:{session}:{history_index}', row, value.get('model')

    for rid, row, model in _limited(rows(), limit):
        trajectory = expansion_trajectory('D07', receipt, rid, row)
        trajectory.lineage['teacher'] = _teacher(model)
        trajectory.lineage['notes']['communication'] = 'alternative-session-history'
        yield rid, _initial_request_background(trajectory)


def agentsynth(limit=None):
    """Primary D11 transcripts; mock generation and verification remain source facts."""
    root = EXPANSION_ROOT / 'D11'
    receipt = json.loads((root / 'acquisition.json').read_text())

    def rows():
        with (root / 'train.jsonl').open() as stream:
            for index, line in enumerate(stream):
                if line.strip():
                    yield index, json.loads(line)

    for index, row in _limited(rows(), limit):
        rid = str(row.get('id') or f'train.jsonl:{index}')
        metadata = {**(row.get('metadata') or {}), 'trajectory_id': rid, 'session_id': rid,
                    'source_mode': row.get('mode'), 'generator_model': row.get('generator_model'),
                    'source_reported_success': row.get('success'),
                    'source_reported_verification': row.get('verification')}
        trajectory = expansion_trajectory('D11', receipt, f'train.jsonl:{index}',
                                          {**row, 'metadata': metadata})
        trajectory.lineage['teacher'] = _teacher(row.get('generator_model'))
        trajectory.lineage['notes']['primary_source'] = 'train.jsonl'
        trajectory.lineage['notes']['verification_is_upstream_report_only'] = True
        if row.get('success') is False:
            trajectory.outcome['label'] = 'failed'
        yield index, _initial_request_background(trajectory)


def agentic_coding_sessions(limit=None):
    """D40's serialized coding sessions, retaining reported outcomes and SWE grouping."""
    root = EXPANSION_ROOT / 'D40'
    receipt = json.loads((root / 'acquisition.json').read_text())
    bench = _bench_ids()
    for index, row in _limited(_parquet_groups(root / 'sessions.parquet'), limit):
        raw = json.loads(row['messages_json'])
        messages = []
        for original in raw:
            msg = dict(original)
            if msg.get('tool_calls_json'):
                msg['tool_calls'] = json.loads(msg['tool_calls_json'])
            messages.append(msg)
        meta = json.loads(row.get('ground_truth_meta_json') or '{}')
        instance = meta.get('instance_id') or row.get('source_id')
        normalized = {'messages':messages, 'metadata':{
            'trajectory_id':instance, 'session_id':row['session_id'],
            'source_dataset':row.get('source_dataset'), 'ground_truth':meta}}
        t = expansion_trajectory('D40', receipt, str(index), normalized)
        t.split_groups.append(group_key('swe-instance', instance))
        if instance in bench:
            t.split = 'test'
        resolved = meta.get('resolved')
        t.outcome = {'label':'failed' if resolved is False else 'teacher', 'checked':None,
                     'details':{'reported_resolved':resolved, 'origin':'upstream_metadata',
                                'executed_locally':False}}
        t.lineage['teacher'] = _teacher(row.get('recorded_model'))
        t.lineage['notes']['benchmark_protected'] = instance in bench
        yield str(index), _initial_request_background(t)


def _gatc_rows(root):
    from natlang_neuralese.common.jsonio import open_jsonl_text
    with open_jsonl_text(root / 'sample/messages/messages.jsonl.zst') as stream:
        for index, line in enumerate(stream):
            if line.strip():
                yield index, json.loads(line)


def gatc_sessions(limit=None):
    """Complete pinned messages with evaluation protection resolved before quality selection."""
    from .splits import UnionFind
    root = EXPANSION_ROOT / 'D03'
    receipt = json.loads((root / 'acquisition.json').read_text())
    groups, evaluation_ids = UnionFind(), []
    for index, row in _gatc_rows(root):
        tid = str(row.get('trajectory_id') or index)
        groups.find(tid)
        if row.get('parent_trajectory_id'):
            groups.union(tid, str(row['parent_trajectory_id']))
        if row.get('template_group'):
            groups.union(tid, 'template:' + str(row['template_group']))
        if row.get('is_eval'):
            evaluation_ids.append(tid)
    evaluation_roots = {groups.find(tid) for tid in evaluation_ids}
    for index, row in _limited(_gatc_rows(root), limit):
        tid = str(row.get('trajectory_id') or index)
        component = groups.find(tid)
        metadata = {k: row.get(k) for k in ('trajectory_id', 'parent_trajectory_id', 'template_group',
                    'is_eval', 'eval_kind', 'models', 'repo_id', 'actor_id', 'repo', 'prompt_language',
                    'last_test_result', 'git_commit_made', 'ft_issues')}
        converted = {**row, 'metadata': {**metadata, 'session_id': tid},
                     '_expansion_task_group_text': next((m.get('content') for m in row.get('messages', [])
                         if m.get('role') == 'user' and not m.get('is_wrapper') and m.get('content')), None),
                     '_expansion_split': 'test' if component in evaluation_roots else 'train'}
        t = expansion_trajectory('D03', receipt, str(index), converted)
        t.split_groups.append(group_key('gatc-component', component))
        t.lineage['notes']['source_quality_flags'] = [flag for flag in row.get('ft_issues') or []
                                                     if flag in ('incomplete_tool_args', 'withheld_text')]
        yield index, _initial_request_background(t)


CORPORA.update({'gatc-sessions': (gatc_sessions, 4), "wideseek": (wideseek, 4), "cooperator-dual-policy": (cooperator_dual_policy, 4),
                "searchswarm": (searchswarm, 4), "owl-sft": (owl_sft, 4), "codeagent-sft": (codeagent_sft, 1),
                'evidence-subagent': (evidence_subagent, 4), 'openthoughts-agent': (openthoughts_agent, 4),
                'terminaltraj': (terminaltraj, 4), 'dolci-tool-use': (dolci_tool_use, 4),
                'cognitive-kernel': (cognitive_kernel, 4),
                'deepdive-trajectories': (deepdive_trajectories, 4),
                'agentsynth': (agentsynth, 4),
                'deepnlp-orchestration': (deepnlp_orchestration, 4),
                'finding-right-fit': (finding_right_fit, 4),
                'agentic-coding-sessions': (agentic_coding_sessions, 4)})

from .acquired_agents import (open_swe_acquired, swe_rebench_acquired,
                              nemotron_terminal_acquired, cooperdata_acquired,
                              toucan_acquired, toolmind_acquired, nemotron_agentic_acquired,
                              ultradata_acquired, swe_zero_acquired, coderforge_acquired, orchard_acquired)
CORPORA.update({'open-swe-acquired': (open_swe_acquired, 4),
                'swe-rebench-acquired': (swe_rebench_acquired, 4),
                'nemotron-terminal-acquired': (nemotron_terminal_acquired, 4),
                'cooperdata-acquired': (cooperdata_acquired, 4)})
CORPORA.update({'toucan-acquired': (toucan_acquired, 4),
                'toolmind-acquired': (toolmind_acquired, 4),
                'nemotron-agentic-acquired': (nemotron_agentic_acquired, 4),
                'ultradata-acquired': (ultradata_acquired, 4),
                'swe-zero-acquired': (swe_zero_acquired, 4),
                'coderforge-acquired': (coderforge_acquired, 4),
                'orchard-acquired': (orchard_acquired, 4)})

from .acquired_agents import qwen_cooperation_acquired, openresearcher_structured_acquired, agent_collusion_raw_acquired
CORPORA.update({'qwen-cooperation-acquired':(qwen_cooperation_acquired,4),
                'openresearcher-structured-acquired':(openresearcher_structured_acquired,4),
                'agent-collusion-raw-acquired':(agent_collusion_raw_acquired,4)})

from .acquired_agents import team_cooperation_acquired
CORPORA['team-cooperation-acquired']=(team_cooperation_acquired,4)

from .acquired_diagnosis import mast_diagnosis_acquired,mascope_diagnosis_acquired,d12_feature_status_diagnoses
CORPORA.update({'mast-diagnosis-acquired':(mast_diagnosis_acquired,4),
                'mascope-diagnosis-acquired':(mascope_diagnosis_acquired,4),
                'd12-feature-status-diagnoses-acquired':(d12_feature_status_diagnoses,4)})

from .acquired_archives import cooper_archive_acquired
CORPORA['cooper-archive-acquired']=(cooper_archive_acquired,4)

from .acquired_transformed_coop import qwen_comm4_acquired
CORPORA['qwen-comm4-acquired']=(qwen_comm4_acquired,4)
