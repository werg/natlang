"""The `neuralese_rl` recipe stage (plans/neuralese/S7_RL.md §6): RL that trains the full direct checkpoint.

One implementation per concept: this stage owns only the RL data decision. The update itself is the shared direct
trainer (`train/direct_sft.py`: its loss, optimizer, checkpointing, held CE, task-success cadence and drift monitor),
continued from the pre-RL direct checkpoint with `--fresh-monitor-reference`, so the drift monitor's response KL is
measured against the pre-RL weights.

Mode `rft` (rejection-sampling fine-tuning, S7 step 0): the policy's own successful rollouts
(`natlang.rollout/1`, reward from `eval.task_packet.classify_case` via `rl.task_rollouts`) of tasks it solves only
sometimes (0 < success rate <= `--rft-max-success-rate`) become whole-trajectory documents: each successful model turn
is rendered with the shared serving renderer (`data.text_corpus.native_gold_packet`, the checkpoint's prompt
rendering) with its reply as the supervised suffix. Tasks the policy always solves add nothing; tasks it never solves
need teacher successes (passed as further rollout files). A deterministic fraction of tasks (by task id) is held out
for the trainer's held CE. Mode `grpo` uses exact captured sampling coordinates and token-level clipped
policy gradients through the same direct trainer. A registered production held
corpus may supply ordinary CE checks without withholding fresh RL train tasks.

usage: python -m natlang_neuralese.rl.stage --rollouts R.jsonl [--rollouts T.jsonl] --rl-tokenizer SNAPSHOT
       --out OUT [--rl-mode rft] [--rft-max-success-rate 0.75] [--rft-held-fraction 0.125]
       [--rl-prompt-rendering program-header] <direct_sft arguments: --heads, --continue-from, ...>
"""
from __future__ import annotations

from ..common.jsonio import read_jsonl

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "natlang.s7-rl-stage-data/1"


def held_task(task_id: str, fraction: float) -> bool:
    """Deterministic task-level held split (the whole task's documents go to one side)."""
    if fraction <= 0:
        return False
    digest = int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16)
    return digest / 0xFFFFFFFF < fraction


def rft_tasks(rollouts: Iterable[dict[str, Any]], *, max_success_rate: float) -> set[str]:
    """Tasks the policy solves only sometimes: 0 < success rate <= max_success_rate over scored rollouts."""
    scored: dict[str, list[float]] = {}
    for rollout in rollouts:
        if rollout.get("reward") is not None:
            scored.setdefault(rollout["task_id"], []).append(rollout["reward"].get("quality"))
    return {task for task, values in scored.items()
            if 0 < sum(v == 1.0 for v in values) / len(values) <= max_success_rate}


def select_rft_documents(rollouts: Iterable[dict[str, Any]], *, max_success_rate: float,
                         held_fraction: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The successful turns to supervise, as {task_id, split, messages, tools, target}, and a selection receipt.

    A task qualifies when its scored rollouts have 0 < success rate <= max_success_rate (with teacher rollouts the
    rate counts them too). Unscored rollouts (reward null) are ignored entirely."""
    tasks: dict[str, list[dict[str, Any]]] = {}
    for rollout in rollouts:
        if rollout.get("reward") is None:
            continue
        tasks.setdefault(rollout["task_id"], []).append(rollout)
    documents, selected, skipped = [], [], {"never_solved": 0, "always_solved_above_rate": 0}
    qualifying = rft_tasks((r for members in tasks.values() for r in members), max_success_rate=max_success_rate)
    for task_id in sorted(tasks):
        members = tasks[task_id]
        successes = [r for r in members if r["reward"].get("quality") == 1.0]
        rate = len(successes) / len(members)
        if not successes:
            skipped["never_solved"] += 1
            continue
        if task_id not in qualifying:
            skipped["always_solved_above_rate"] += 1
            continue
        split = "test" if held_task(task_id, held_fraction) else "train"
        selected.append({"task_id": task_id, "success_rate": rate, "successes": len(successes), "split": split})
        for rollout in successes:
            for turn in rollout.get("turns") or []:
                documents.append({"task_id": task_id, "split": split, "messages": turn["messages"],
                                  "tools": turn.get("tools"), "target": turn["target"]})
    receipt = {"schema": SCHEMA, "mode": "rft", "max_success_rate": max_success_rate, "held_fraction": held_fraction,
               "tasks_seen": len(tasks), "tasks_selected": selected, "skipped": skipped,
               "documents": len(documents)}
    return documents, receipt


def render_rows(documents: list[dict[str, Any]], tokenizer, prompt_rendering: str) -> list[dict[str, Any]]:
    """Direct-trainer text rows (stream_text_rows contract) through the shared serving renderer."""
    from ..data.text_corpus import native_gold_packet, tokenizer_fingerprint
    fingerprint = tokenizer_fingerprint(tokenizer)
    rows = []
    for document in documents:
        text, ids, boundary = native_gold_packet(tokenizer, document["messages"], document["target"],
                                                 document["tools"], prompt_rendering=prompt_rendering)
        rows.append({"text": text, "token_ids": ids, "supervised_suffix_start": boundary,
                     "tokenizer_sha256": fingerprint, "split": document["split"],
                     "source_groups": ["s7-task:" + document["task_id"]], "sampling_group": hashlib.sha256(text.encode()).hexdigest(),
                     "text_cohort": "s7-rft", "prompt_rendering": prompt_rendering})
    return rows


def parse(argv=None) -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rollouts", type=Path, action="append", required=True)
    p.add_argument("--rl-tokenizer", type=Path, required=True, help="the base snapshot whose tokenizer the trainer uses")
    p.add_argument("--rl-mode", choices=("rft", "grpo"), default="rft")
    p.add_argument("--rl-held-text-data", type=Path, help="registered production corpus; read test rows only")
    p.add_argument("--rl-held-documents", type=int, default=64)
    p.add_argument("--rl-block-bundle", type=Path, help="exact value-channel block bundle manifest")
    p.add_argument("--rl-policy-id", help="SHA-256 of the full checkpoint that sampled the rollouts")
    p.add_argument("--rl-clip-low", type=float, default=0.2)
    p.add_argument("--rl-clip-high", type=float, default=0.28)
    p.add_argument("--rft-max-success-rate", type=float, default=0.75)
    p.add_argument("--rft-held-fraction", type=float, default=0.125)
    p.add_argument("--rl-prompt-rendering", choices=("full", "program-header"), default="program-header")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_known_args(argv)


def main(argv=None) -> None:
    a, rest = parse(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    rollouts = [row for path in a.rollouts for row in read_jsonl(path)]
    if a.rl_mode == "grpo":
        from .full_policy import build_documents
        from ..common.hashing import sha256_file_hex
        # Only the actually sampled checkpoint may provide missing behavior
        # scores. Full state continuation preserves the existing optimizer.
        if "--continue-from" not in rest:
            raise ValueError("GRPO requires --continue-from sampled full checkpoint")
        checkpoint = Path(rest[rest.index("--continue-from") + 1])
        if sha256_file_hex(checkpoint) != a.rl_policy_id:
            raise ValueError("GRPO collection policy differs from continuation checkpoint")
        documents = build_documents(rollouts, policy_id=a.rl_policy_id, block_bundle=a.rl_block_bundle)
        for d in documents:
            d['split'] = ('test' if not a.rl_held_text_data and held_task(d['task_id'], a.rft_held_fraction) else 'train')
        receipt = {'schema': SCHEMA, 'mode': 'grpo', 'documents': len(documents),
                   'tasks_selected': sorted({d['task_id'] for d in documents}), 'skipped': {},
                   'policy_id': a.rl_policy_id}
    else:
        documents, receipt = select_rft_documents(rollouts, max_success_rate=a.rft_max_success_rate,
                                                 held_fraction=0. if a.rl_held_text_data else a.rft_held_fraction)
    if not any(d["split"] == "train" for d in documents) or (not a.rl_held_text_data and not any(d["split"] == "test" for d in documents)):
        raise SystemExit("RL stage needs train and held documents: " + json.dumps(receipt["skipped"]))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(a.rl_tokenizer))
    if a.rl_mode == 'grpo':
        from .sampling import capture_items
        from ..data.text_corpus import tokenizer_fingerprint
        fingerprint = tokenizer_fingerprint(tokenizer)
        bundle = json.loads(a.rl_block_bundle.read_text()) if a.rl_block_bundle else {'blocks': {}}
        rows = []
        for index, document in enumerate(documents):
            prompt, target, _ = capture_items(document['sampling'])
            def coordinates(items):
                ids = []
                for kind, value in items:
                    if kind == 'tok':
                        ids.append(value)
                    else:
                        # Shape-only placeholders: neither updates nor held CE
                        # consume them. Both use exact token/block replay.
                        ids.extend([0] * bundle['blocks'][value]['length'])
                return ids
            prompt_ids, target_ids = coordinates(prompt), coordinates(target)
            payload = json.dumps({'sample': index, 'capture': document['sampling']}, sort_keys=True)
            key = hashlib.sha256(payload.encode()).hexdigest()
            document['document_sha256'] = key
            document['multiplicity'] = 1
            rows.append({'id': key, 'document_sha256': key, 'text': 'RL replay planning ' + payload,
                'token_ids': prompt_ids + target_ids, 'tokenizer_sha256': fingerprint,
                'supervised_suffix_start': len(prompt_ids), 'split': document['split'],
                'source_groups': ['s7-task:' + document['task_id']], 'sampling_group': key,
                'text_cohort': 's7-grpo', 'prompt_rendering': a.rl_prompt_rendering,
                'rl_planning_only': True})
        spec = a.out / 'rl-objective.json'
        spec.write_text(json.dumps({'schema': 'natlang.full-policy-grpo/1', 'policy_id': a.rl_policy_id,
            'clip_low': a.rl_clip_low, 'clip_high': a.rl_clip_high,
            'block_bundle': str(a.rl_block_bundle.resolve()) if a.rl_block_bundle else None,
            'documents': documents}, ensure_ascii=False) + '\n')
        rest += ['--rl-objective', str(spec)]
    else:
        rows = render_rows(documents, tokenizer, a.rl_prompt_rendering)
    if a.rl_held_text_data:
        from ..common.jsonio import iter_jsonl_rows
        from ..common.hashing import sha256_file_hex
        if a.rl_held_documents < 1:
            raise ValueError('production RL needs positive held-document coverage')
        held = []
        for _, row in iter_jsonl_rows(a.rl_held_text_data, split='test'):
            held.append(row)
            if len(held) == a.rl_held_documents:
                break
        if not held:
            raise ValueError('registered RL held corpus has no test rows')
        # The shared loader independently enforces source-group/text separation.
        rows.extend(held)
        receipt['held'] = {'kind': 'ordinary-production-token-CE',
            'path': str(a.rl_held_text_data), 'sha256': sha256_file_hex(a.rl_held_text_data),
            'documents': len(held), 'selected_split': 'test'}
    data = a.out / "rl-text-data.jsonl"
    data.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    from ..common.hashing import sha256_file_hex
    receipt.update(rollout_files={str(p): sha256_file_hex(p) for p in a.rollouts}, text_data=str(data),
                   text_data_sha256=sha256_file_hex(data))
    (a.out / "rl-data-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "rl_data", **{k: receipt[k] for k in ("documents", "skipped")},
                      "tasks_selected": len(receipt["tasks_selected"])}), flush=True)
    from ..train import direct_sft
    direct_sft.main([*rest, "--out", str(a.out), "--text-data", str(data), "--fresh-monitor-reference"])


if __name__ == "__main__":
    main()
