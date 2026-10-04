"""Schnitzeljagd multi-turn agent episodes (`turns`) → continuation records (S1 §3.3, §4.2).

Each episode has a task (the query after the "Use the stored …" framing), alternating assistant and
environment turns, and supports (protocol, policy, know-how, worked examples). The supports become
background sources; environment turns become user messages (observations, customer replies, tool
results), as the episode presented them to the agent.
"""
from __future__ import annotations

import re
from pathlib import Path

from .common import Reject, jsonl_rows
from .records import group_key, source, strip_markup
from .schnitzel import CORPORA_DIR, SPLIT_FILES, _STORED_RE, _source_licence, corpus_short
from .trajectory import Trajectory, licence, message

# domain → (family, upstream, licence (spdx, noncommercial, notes), session noun)
DOMAINS = {
    "agentbank": ("agent_continuation_mixed", "Solaris99/AgentBank", ("Apache-2.0", False, ""), "agent session"),
    "agentinstruct": ("agent_continuation_mixed", "zai-org/AgentInstruct",
                      ("LicenseRef-unverified", False, "Card metadata states no licence; AgentTuning code is Apache-2.0."), "agent session"),
    "alfworld": ("agent_continuation_household", "alfworld/alfworld (ETO expert trajectories)",
                 ("MIT", False, "ALFWorld code and ALFRED data MIT; trajectories from agent-eto/eto-sft-trajectory (Apache-2.0)."), "household task"),
    "scienceworld": ("agent_continuation_science", "allenai/ScienceWorld gold paths", ("Apache-2.0", False, ""), "science experiment"),
    "webshop": ("agent_continuation_webshop", "agent-eto/eto-sft-trajectory (WebShop)",
                ("Apache-2.0", False, "ETO trajectories Apache-2.0; WebShop environment MIT."), "shopping session"),
    "apigen_mt": ("agent_continuation_tool_policy", "Salesforce/APIGen-MT-5k", ("CC-BY-NC-4.0", True, "Gated upstream; tau-bench tool outputs."),
                  "customer-service session"),
}
CORPORA = [
    "tasks-agentbank-20260927", "tasks-agentinstruct-20260927", "tasks-alfworld-20260927", "tasks-alfworld-bg-20260927",
    "tasks-scienceworld-20260927", "tasks-scienceworld-bg-20260927", "tasks-webshop-20260927", "tasks-apigen-mt-20260927",
]
BACKGROUND_ROLES = {"protocol": "tool_doc", "policy": "document", "tool_doc": "tool_doc", "know_how": "document",
                    "worked_example": "document"}
CONVERTER = "scripts/neuralese_data/schnitzel_turns.py@1"


def _groups(domain: str, episode: dict) -> list[str]:
    prov = episode.get("provenance") or {}
    verify = episode.get("verify") or {}
    if domain == "alfworld" and verify.get("game"):
        return [group_key("alfworld-game", verify["game"].rsplit("/", 2)[0])]
    if domain == "scienceworld" and prov.get("group"):
        return [group_key("scienceworld-task", prov["group"])]
    eid = episode.get("episode_id")
    if not eid:
        raise Reject("missing-group: no episode id")
    return [group_key(f"sdkb-{domain}", eid)]


def trajectory(corpus: str, split: str, row_index: int, episode: dict) -> Trajectory:
    prov = episode.get("provenance") or {}
    domain = prov.get("domain") or prov.get("dataset")
    if domain not in DOMAINS:
        raise Reject(f"unknown-domain: {domain}")
    family, upstream, (spdx, nc, notes), noun = DOMAINS[domain]
    if corpus.endswith("-bg") or "-bg-" in corpus:
        family += "_bg"
    query = strip_markup(episode.get("query") or "")
    m = _STORED_RE.match(query)
    task = (query[m.end():] if m else query).strip()
    if not task:
        raise Reject("empty-request")
    messages = [message("user", task)]
    for t in episode.get("turns") or []:
        text = strip_markup(t.get("text") or "")
        if text.strip():
            messages.append(message("assistant" if t.get("role") == "assistant" else "user", text))
    background = []
    for s in episode.get("supports") or []:
        text = strip_markup(s.get("text") or "").strip()
        if text:
            background.append(source(BACKGROUND_ROLES.get(s.get("kind"), "document"), text,
                                     meta={"kind": s.get("kind")}, license=_source_licence(s.get("provenance") or {})))
    verify = episode.get("verify") or {}
    reward = verify.get("reward")
    eid = re.sub(r"\s+", "_", str(episode.get("episode_id") or row_index))
    return Trajectory(
        id=f"sdkb:{corpus_short(corpus)}:{eid}", family=family, messages=messages, head=1, noun=noun,
        lineage={"project": "schnitzeljagd", "store": corpus, "store_version": (re.search(r"2026\d{4}", corpus) or [None])[0],
                 "row": f"{split}:{row_index}", "upstream": upstream, "upstream_id": episode.get("episode_id"),
                 "upstream_revision": None, "teacher": None, "converter": CONVERTER,
                 "notes": {"environment": episode.get("environment"),
                           "verify": {k: v for k, v in verify.items() if k != "calls"}}},
        license=licence(spdx, nc, notes), split=split, split_groups=_groups(domain, episode),
        outcome={
            "label": "gold", "checked": "expert-trajectory",
            "details": {
                "checker_evidence": {
                    "origin": "upstream_expert_trajectory_annotation",
                    "local_execution": False,
                    "reward_field_present": "reward" in verify,
                    "raw_reward": reward,
                },
                **({"reward": reward} if reward is not None else {}),
            },
        },
        background=background,
    )


def iter_corpus(corpus: str, limit_per_split: int | None, root: Path = CORPORA_DIR):
    base = root / corpus
    for split, filename in SPLIT_FILES.items():
        path = base / filename
        if not path.exists():
            continue
        lim = None if limit_per_split is None else (limit_per_split if split == "train" else max(1, limit_per_split // 5))
        for i, episode in jsonl_rows(path, lim):
            yield split, i, episode
