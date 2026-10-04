"""bgkit task stores → port records (S1 §3.1, §4.1).

bgkit stores share one text schema: family, framing, context, prompt, instruction, tool_name,
tool_args, target, split, meta. Converters keep content and supervision and drop bgkit's harness:
compression prompts become write-site instructions phrased as the consumer's need, the
`<|reserved_6|>` sentinel and other chat markup are removed, `ANSWER:` prefixes are stripped.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .common import Reject
from .records import (VERSION, exact_refs_from, group_key, license_, seal, source, strip_answer_prefix,
                      strip_markup)

TASKS_DIR = Path("/home/werg/bgkit-data-nvme/bgkit2/tasks")
CONVERTER = "scripts/neuralese_data/bgkit.py@1"

_PROMPT_RE = re.compile(r"^\s*Compress the (?P<what>.+?) to(?: answer| digest)?:\s*(?P<need>.*?)\s*$", re.S)
_WORKSPACE_RE = re.compile(r"/workspace/([A-Za-z0-9_.-]+?)__([A-Za-z0-9_.-]+?)(?:__[^/\s]*)?(?:/|\s|$)")

QA_STORES = {
    # store: (family, upstream, licence, group namespace builder)
    "qa_squad": ("qa_extractive", "rajpurkar/squad", ("CC-BY-SA-4.0", False, "")),
    "qa_squad_article_v2": ("qa_extractive_article", "rajpurkar/squad", ("CC-BY-SA-4.0", False, "")),
    "qa_hotpotqa": ("qa_multihop", "hotpotqa/hotpot_qa", ("CC-BY-SA-4.0", False, "")),
    "qa_hotpotqa_distractor": ("qa_multihop_distractor", "hotpotqa/hotpot_qa", ("CC-BY-SA-4.0", False, "")),
    "qa_triviaqa": ("qa_trivia", "mandarjoshi/trivia_qa", ("Apache-2.0", False, "Evidence documents are web and Wikipedia text under their own terms.")),
    "qa_narrativeqa": ("qa_narrative", "deepmind/narrativeqa", ("Apache-2.0", False, "Stories are Project Gutenberg texts and movie scripts under their own terms.")),
    "qa_quality_mcq": ("qa_mcq", "nyu-mll/quality", ("CC-BY-4.0", False, "")),
    "qa_multineedle": ("qa_needles", "rajpurkar/squad (needles in filler)", ("CC-BY-SA-4.0", False, "SQuAD passages among Wikipedia filler.")),
    "qa_multineedle_long_fam": ("qa_needles_long", "rajpurkar/squad (needles in filler)", ("CC-BY-SA-4.0", False, "SQuAD passages among Wikipedia filler.")),
    "babi_filler_v3": ("qa_babi_filler", "facebook/babi_qa + wikitext filler",
                       ("LicenseRef-mixed", False, "bAbI tasks BSD-3-Clause; filler text from wikitext (CC-BY-SA-3.0) or PG-19.")),
}
# Stores in tool_slots framing whose slots are documents and whose target answers one question.
SLOT_QA_STORES = {
    # store: (family, upstream, licence, system role of the consumer, noun)
    "memory_recall_v2": ("memory_recall", "Conversation Chronicles (+ bgkit teacher questions)",
                         ("LicenseRef-unverified", False, "Sessions from Conversation Chronicles (meta source chronicles); questions and answers by a bgkit teacher; upstream licence to be confirmed."),
                         "remembered conversation sessions"),
    "memory_qa_v2_pairs": ("memory_qa", "Conversation Chronicles (+ bgkit teacher questions)",
                           ("LicenseRef-unverified", False, "Sessions from Conversation Chronicles (meta source chronicles+teacher); questions and answers by a bgkit teacher; upstream licence to be confirmed."),
                           "remembered conversation sessions"),
    "memory_qa_v2_user_assistant": ("memory_qa_user_assistant", "Conversation Chronicles (+ bgkit teacher questions)",
                                    ("LicenseRef-unverified", False, "User/assistant sessions (Conversation Chronicles style); questions and answers by a bgkit teacher; upstream licence to be confirmed."),
                                    "remembered conversation sessions"),
    "memory_qa_s2_ua": ("memory_qa_user_assistant", "Conversation Chronicles (+ bgkit teacher questions)",
                        ("LicenseRef-unverified", False, "User/assistant sessions (Conversation Chronicles style); questions and answers by a bgkit teacher; upstream licence to be confirmed."),
                        "remembered conversation sessions"),
    "chatqa2_long_sft": ("qa_long_context", "nvidia/ChatQA2-Long-SFT-data",
                         ("LicenseRef-unverified", False, "nvidia/ChatQA2-Long-SFT-data; no card downloaded locally, NVIDIA dataset terms to be confirmed."),
                         "context"),
    "chatqa2_narrativeqa": ("qa_narrative_long", "nvidia/ChatQA2-Long-SFT-data (NarrativeQA)",
                            ("LicenseRef-unverified", False, "NarrativeQA within nvidia/ChatQA2-Long-SFT-data; no card downloaded locally."),
                            "story"),
}
OVERVIEW_STORES = {"file_overview": "file_overview", "repo_overview_v1": "repo_overview"}
TOOL_DIGEST_STORES = {"tool_digest_v2": "run1", "tool_digest_ext_v3": "ext_synth"}
REPO_STORES = {"repo_qa_file_v3": "repo_qa", "repo_localisation_v3": "repo_localisation"}
ALL_STORES = [*QA_STORES, *TOOL_DIGEST_STORES, *REPO_STORES, *SLOT_QA_STORES, *OVERVIEW_STORES]

SPLIT_MAP = {"train": "train", "eval": "test", "validation": "validation", "test": "test"}


def store_version(store: str) -> str | None:
    m = re.search(r"_(v\d+[a-z]?)$", store)
    return m.group(1) if m else None


def _meta(row) -> dict:
    raw = row.get("meta") or "{}"
    try:
        value = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except json.JSONDecodeError as exc:
        raise Reject(f"bad-meta: {exc}") from exc
    return value if isinstance(value, dict) else {}


def _split(row) -> str:
    split = SPLIT_MAP.get(str(row.get("split") or "").strip())
    if not split:
        raise Reject(f"unknown-split: {row.get('split')!r}")
    return split


def _parse_prompt(prompt: str) -> tuple[str, str]:
    m = _PROMPT_RE.match(strip_markup(prompt or ""))
    if not m:
        raise Reject(f"unparsed-prompt: {prompt[:80]!r}")
    return m.group("what").strip(), m.group("need").strip()


def _lineage(store: str, row_index: int, upstream: str | None, *, upstream_id=None, revision=None, teacher=None,
             notes=None) -> dict:
    out = {"project": "bgkit", "store": store, "store_version": store_version(store), "row": row_index,
           "upstream": upstream, "upstream_id": upstream_id, "upstream_revision": revision, "teacher": teacher,
           "converter": CONVERTER, "sha256": ""}
    if notes:
        out["notes"] = notes
    return out


def _result_type(what: str) -> str:
    what = what.lower()
    if "passages" in what:
        return "Neuralese<Passages>"
    if "passage" in what:
        return "Neuralese<Passage>"
    if "file" in what:
        return "Neuralese<SourceFile>"
    if "result" in what:
        return "Neuralese<ToolResult>"
    return "Neuralese<Document>"


def _qa_groups(store: str, meta: dict, row_index: int) -> list[str]:
    if store in ("qa_squad", "qa_squad_article_v2"):
        title = meta.get("title") or str(meta.get("doc", "")).split("#")[0]
        if not title:
            raise Reject("missing-group: squad title")
        return [group_key("wiki", title)]
    if store in ("qa_hotpotqa", "qa_hotpotqa_distractor"):
        qid = meta.get("id") or meta.get("doc")
        if not qid:
            raise Reject("missing-group: hotpot id")
        return [group_key("hotpotqa-q", qid)]
    if store == "qa_triviaqa":
        if not meta.get("id"):
            raise Reject("missing-group: triviaqa id")
        return [group_key("triviaqa-q", meta["id"])]
    if store == "qa_narrativeqa":
        if not meta.get("doc"):
            raise Reject("missing-group: narrativeqa doc")
        return [group_key("narrativeqa-doc", meta["doc"])]
    if store == "qa_quality_mcq":
        if not meta.get("doc"):
            raise Reject("missing-group: quality doc")
        return [group_key("quality-doc", meta["doc"])]
    if store in ("qa_multineedle", "qa_multineedle_long_fam"):
        if not meta.get("doc"):
            raise Reject("missing-group: needle doc")
        return [group_key("needle-doc", meta["doc"])]
    if store == "babi_filler_v3":
        doc = str(meta.get("doc") or "")
        if not doc:
            raise Reject("missing-group: babi doc")
        return [group_key("babi-story", doc.rsplit(":", 1)[0])]
    raise Reject(f"no-group-rule: {store}")


def convert_qa(store: str, row_index: int, row: dict) -> dict:
    family, upstream, (spdx, nc, lic_notes) = QA_STORES[store]
    meta = _meta(row)
    context = strip_markup(row.get("context") or "").strip()
    if not context:
        raise Reject("empty-source")
    what, need = _parse_prompt(row.get("prompt") or "")
    instruction = strip_markup(row.get("instruction") or "").strip()
    question = instruction.split("\nOptions:")[0].strip() or need
    target_value = strip_answer_prefix(strip_markup(row.get("target") or "")).strip()
    if not target_value:
        raise Reject("empty-target")
    if store == "qa_quality_mcq":
        options = meta.get("options") or []
        if len(options) < 2:
            raise Reject("missing-options")
        letters = "ABCDEFGH"
        consumer_text = question + "\nOptions:\n" + "\n".join(f"({letters[i]}) {o}" for i, o in enumerate(options))
        consumer_text += "\nAnswer with the letter of the correct option."
        target = {"kind": "choice", "value": target_value, "alternatives": []}
        if target_value not in letters[:len(options)]:
            raise Reject(f"bad-choice: {target_value!r}")
    else:
        consumer_text = question
        answers = [*(meta.get("answers") or []), *(meta.get("aliases") or [])]
        alternatives = []
        for a in answers:
            a = str(a).strip()
            if a and a != target_value and a not in alternatives:
                alternatives.append(a)
        target = {"kind": "text", "value": target_value, "alternatives": alternatives}
    title = meta.get("title")
    record = {
        "version": VERSION,
        "id": f"bgkit:{store}:{row_index}",
        "family": family,
        "task": "consume",
        "sources": [source("passage" if "passage" in what else "document", context, title=title)],
        "writer": {
            "instructions": f"Read the {what} so that this question can be answered: {question}",
            "instructions_general": f"Read the {what} so that later questions about it can be answered.",
            "result_type": _result_type(what),
            "context": [],
        },
        "consumer": {"context": [{"role": "user", "content": consumer_text}], "withheld": ["sources"]},
        "target": target,
        "outcome": {
            "label": "gold", "checked": "reference-answer",
            "details": {"checker_evidence": {
                "origin": "upstream_reference_answer",
                "local_execution": False,
            }},
        },
        "lineage": _lineage(store, row_index, upstream, upstream_id=str(meta.get("id") or meta.get("doc") or "") or None),
        "license": license_(spdx, nc, lic_notes),
        "split": _split(row),
        "split_groups": _qa_groups(store, meta, row_index),
    }
    return seal(record)


def _repo_from_workspace(text: str) -> str | None:
    m = _WORKSPACE_RE.search(text)
    return f"{m.group(1)}/{m.group(2)}" if m else None


def convert_tool_digest(store: str, row_index: int, row: dict, task_repos: dict) -> dict:
    meta = _meta(row)
    context = strip_markup(row.get("context") or "").strip()
    if not context:
        raise Reject("empty-source")
    what, purpose = _parse_prompt(row.get("prompt") or "")
    purpose = purpose.rstrip(" .")
    if not purpose:
        raise Reject("empty-purpose")
    tool = (row.get("tool_name") or "").strip() or "tool"
    try:
        args = json.loads(row.get("tool_args") or "{}")
    except json.JSONDecodeError:
        args = {"raw": row.get("tool_args")}
    target = strip_answer_prefix(strip_markup(row.get("target") or "")).strip()
    if not target:
        raise Reject("empty-target")
    repo, revision = None, None
    task = task_repos.get(meta.get("task_id") or "")
    if task:
        repo, revision = task.get("repo"), task.get("commit")
    repo = repo or _repo_from_workspace(context) or _repo_from_workspace(json.dumps(args))
    groups = [group_key("repo", repo)] if repo else []
    if meta.get("task_id"):
        groups.append(group_key("bgkit-task", meta["task_id"]))
    if not groups:
        key = meta.get("key")
        if not key:
            raise Reject("missing-group: no repo, task or key")
        groups.append(group_key("tool-digest-key", key))
    record = {
        "version": VERSION,
        "id": f"bgkit:{store}:{row_index}",
        "family": "tool_digest",
        "task": "consume",
        "sources": [source("tool_output", context, meta={"tool": tool, "arguments": args},
                           exact_refs=exact_refs_from(context))],
        "writer": {
            "instructions": f"Read this {tool} result for: {purpose}",
            "instructions_general": f"Read this {tool} result so that later steps of the task can use it.",
            "result_type": "Neuralese<ToolResult>",
            "context": [],
        },
        "consumer": {"context": [{"role": "user", "content":
                                  f"Write a compact digest of the {tool} result for: {purpose}. Keep every exact "
                                  f"identifier, path, line number and value that matters."}],
                     "withheld": ["sources"]},
        "target": {"kind": "text", "value": target, "alternatives": []},
        "outcome": {"label": "teacher", "checked": None,
                    "details": {"generator": TOOL_DIGEST_STORES[store], "raw_chars": meta.get("raw_chars")}},
        "lineage": _lineage(store, row_index, f"github:{repo}" if repo else None, upstream_id=meta.get("task_id") or meta.get("key"),
                            revision=revision, teacher="unknown",
                            notes={"bgkit_generator": TOOL_DIGEST_STORES[store]}),
        "license": license_("LicenseRef-repository-content", False,
                            "Tool output from public repositories under their own licences; digest written by a bgkit teacher."),
        "split": _split(row),
        "split_groups": groups,
    }
    return seal(record)


def convert_repo(store: str, row_index: int, row: dict, task_repos: dict, heldout_repos: set) -> dict:
    meta = _meta(row)
    raw = strip_markup(row.get("context") or "")
    first, _, body = raw.partition("\n")
    m = re.match(r"^# file:\s*(\S+)", first)
    if not m:
        raise Reject("unparsed-file-header")
    path = m.group(1)
    body = body.strip("\n")
    if not body.strip():
        raise Reject("empty-source")
    repo = meta.get("repo")
    if not repo:
        raise Reject("missing-group: repo")
    question = strip_markup(row.get("instruction") or "").strip()
    target_value = strip_answer_prefix(strip_markup(row.get("target") or "")).strip()
    if not question or not target_value:
        raise Reject("empty-question-or-target")
    task = task_repos.get(meta.get("task_id") or "", {})
    family = REPO_STORES[store]
    split = _split(row)
    if repo in heldout_repos:
        split = "test"
    record = {
        "version": VERSION,
        "id": f"bgkit:{store}:{row_index}",
        "family": family,
        "task": "consume",
        "sources": [source("file", body, title=path, exact_refs=[{"text": path, "kind": "path"}],
                           meta={"repository": repo})],
        "writer": {
            "instructions": f"Read the file {path} of {repo} so that this question can be answered: {question}",
            "instructions_general": f"Read the file {path} of {repo} so that later questions about the repository can be answered.",
            "result_type": "Neuralese<SourceFile>",
            "context": [],
        },
        "consumer": {"context": [{"role": "user", "content": f"In the repository {repo}: {question}"}],
                     "withheld": ["sources"]},
        "target": {"kind": "text", "value": target_value, "alternatives": []},
        "outcome": {"label": "gold", "checked": "fix-commit" if family == "repo_localisation" else (task.get("verifier") and f"static-analysis:{task['verifier']}") or "static-analysis",
                    "details": {"answer_type": meta.get("answer_type")}},
        "lineage": _lineage(store, row_index, f"github:{repo}", upstream_id=meta.get("task_id"),
                            revision=task.get("commit") or meta.get("fix_commit")),
        "license": license_("LicenseRef-repository-content", False, f"Source file from {repo} under that repository's licence."),
        "split": split,
        "split_groups": [group_key("repo", repo)],
    }
    return seal(record)


def load_task_repos(tasks_dir: Path = TASKS_DIR) -> dict:
    """task_id → {repo, commit, verifier} from bgkit's repo task files (all versions, held-out included)."""
    out: dict = {}
    for path in sorted(tasks_dir.glob("repo_*_tasks*.jsonl")):
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                t = json.loads(line)
                if t.get("task_id"):
                    out.setdefault(t["task_id"], {"repo": t.get("repo"), "commit": t.get("commit"), "verifier": t.get("verifier")})
    return out


def load_heldout_repos(tasks_dir: Path = TASKS_DIR) -> set:
    path = tasks_dir / "eval_repos_heldout.txt"
    return {line.strip() for line in path.read_text().splitlines() if line.strip()} if path.exists() else set()


def converter_for(store: str, task_repos: dict, heldout_repos: set):
    if store in QA_STORES:
        return lambda i, row: convert_qa(store, i, row)
    if store in TOOL_DIGEST_STORES:
        return lambda i, row: convert_tool_digest(store, i, row, task_repos)
    if store in REPO_STORES:
        return lambda i, row: convert_repo(store, i, row, task_repos, heldout_repos)
    if store in SLOT_QA_STORES:
        return lambda i, row: convert_slot_qa(store, i, row)
    if store in OVERVIEW_STORES:
        return lambda i, row: convert_overview(store, i, row, heldout_repos)
    raise KeyError(f"no bgkit converter for {store}")


def _json_list(raw, what: str) -> list:
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError as exc:
        raise Reject(f"bad-{what}: {exc}") from exc
    if not isinstance(value, list):
        raise Reject(f"bad-{what}: not a list")
    return value


def _consumer_messages(raw) -> list[dict]:
    """The consumer's chat with bgkit's slot sentinels removed; the request is kept verbatim."""
    out = []
    for m in _json_list(raw, "instruction"):
        if not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant"):
            continue
        content = m.get("content") if isinstance(m.get("content"), str) else ""
        content = re.sub(r"(?:<\|reserved_6\|>\s*)+", "", content)
        content = re.sub(r"^(Memory|Context):\s*\n+", "", content.strip())
        content = strip_markup(content).strip()
        if content:
            out.append({"role": m["role"], "content": content})
    return out


def convert_slot_qa(store: str, row_index: int, row: dict) -> dict:
    """tool_slots stores (memory, ChatQA2): slots become sources; the consumer's question is the purpose."""
    family, upstream, (spdx, nc, lic_notes), noun = SLOT_QA_STORES[store]
    meta = _meta(row)
    slots = [strip_markup(str(s)).strip() for s in _json_list(row.get("context"), "context")]
    slots = [s for s in slots if s]
    if not slots:
        raise Reject("empty-source")
    consumer = _consumer_messages(row.get("instruction"))
    request = next((m["content"] for m in reversed(consumer) if m["role"] == "user"), "")
    if not request:
        raise Reject("empty-question-or-target")
    request = re.sub(r"^Question:\s*", "", request).strip()
    target = strip_answer_prefix(strip_markup(row.get("target") or "")).strip()
    if not target:
        raise Reject("empty-target")
    if "episode" in meta:
        groups = [group_key("memory-episode", f"{meta.get('source', 'memory')}:{meta['episode']}")]
    elif meta.get("doc"):
        groups = [group_key("chatqa2-doc", str(meta["doc"])[:200])]
    else:
        raise Reject("missing-group: no episode or doc")
    role = "document" if family.startswith("memory") else "passage"
    record = {
        "version": VERSION,
        "id": f"bgkit:{store}:{row_index}",
        "family": family,
        "task": "consume",
        "sources": [source(role, s) for s in slots],
        "writer": {
            "instructions": f"Read the {noun} so that this question can be answered: {request}",
            "instructions_general": f"Read the {noun} so that later questions about them can be answered.",
            "result_type": "Neuralese<Memory>" if family.startswith("memory") else "Neuralese<Passages>",
            "context": [],
        },
        "consumer": {"context": consumer, "withheld": ["sources"]},
        "target": {"kind": "text", "value": target, "alternatives": []},
        "outcome": {"label": "teacher" if "teacher" in str(meta.get("source", "")) or family.startswith("memory") else "gold",
                    "checked": None if family.startswith("memory") else "reference-answer",
                    "details": {k: meta[k] for k in ("qtype", "n_sessions", "n_slots") if k in meta}},
        "lineage": _lineage(store, row_index, upstream, upstream_id=str(meta.get("episode") or meta.get("doc") or "")[:200] or None,
                            teacher="unknown" if family.startswith("memory") else None),
        "license": license_(spdx, nc, lic_notes),
        "split": _split(row),
        "split_groups": groups,
    }
    return seal(record)


def convert_overview(store: str, row_index: int, row: dict, heldout_repos: set) -> dict:
    """file_overview (describe a file) and repo_overview_v1 (locate the file for a question in a repository map)."""
    family = OVERVIEW_STORES[store]
    meta = _meta(row)
    repo, path = meta.get("repo"), meta.get("path")
    if not repo:
        raise Reject("missing-group: repo")
    context = strip_markup(row.get("context") or "").strip()
    if not context:
        raise Reject("empty-source")
    question = strip_markup(row.get("instruction") or "").strip()
    target = strip_answer_prefix(strip_markup(row.get("target") or "")).strip()
    if not question or not target:
        raise Reject("empty-question-or-target")
    split = "test" if repo in heldout_repos else _split(row)
    if family == "file_overview":
        sources = [source("file", context, title=path, exact_refs=[{"text": path, "kind": "path"}] if path else [],
                          meta={"repository": repo})]
        writer = {"instructions": f"Read the file {path} of {repo} so that its purpose, contents and role in the repository can be described.",
                  "result_type": "Neuralese<SourceFile>", "context": []}
        consumer_text = f"In the repository {repo}, give an overview of what {path} contains and what it is for."
        outcome = {"label": "teacher", "checked": None}
        teacher = "unknown"
    else:
        sources = [source("document", context, title=f"{repo} module map", meta={"repository": repo},
                          exact_refs=exact_refs_from(context))]
        writer = {"instructions": f"Read the module map of {repo} so that the file relevant to this question can be located: {question}",
                  "instructions_general": f"Read the module map of {repo} so that files can later be located in it.",
                  "result_type": "Neuralese<RepositoryMap>", "context": []}
        consumer_text = f"In the repository {repo}: {question}"
        outcome = {"label": "gold", "checked": "path-in-repository"}
        teacher = None
    record = {
        "version": VERSION,
        "id": f"bgkit:{store}:{row_index}",
        "family": family,
        "task": "consume",
        "sources": sources,
        "writer": writer,
        "consumer": {"context": [{"role": "user", "content": consumer_text}], "withheld": ["sources"]},
        "target": {"kind": "text", "value": target, "alternatives": []},
        "outcome": outcome,
        "lineage": _lineage(store, row_index, f"github:{repo}", upstream_id=path, teacher=teacher),
        "license": license_("LicenseRef-repository-content", False,
                            f"Content from {repo} under that repository's licence" + ("; overview written by a bgkit teacher." if teacher else ".")),
        "split": split,
        "split_groups": [group_key("repo", repo)],
    }
    return seal(record)
