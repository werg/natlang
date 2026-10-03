"""Pure, fail-closed primitives for checkpoint-bound training exclusions.

An exclusion is applied to the already established training permutation. It
does not rerun the random group split, which could change held-out membership.
The caller must persist the source split and the excluded row identities.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Sequence

from scripts.audit_training_mix import modality
from scripts.corpus import records
from scripts.training_append import ids_digest
from collections import Counter
import re


@dataclass(frozen=True)
class ExclusionOrder:
    rows: list[dict]
    previous_cursor: int
    cursor: int
    consumed_removed_ids: tuple[str, ...]
    future_removed_ids: tuple[str, ...]
    previous_order_sha256: str
    filtered_order_sha256: str


def pending_source_review_entry(source_review_text: str, source_id: str) -> bool:
    """Check the status of this exact object in the TS/JSON source-review list."""
    escaped = re.escape(str(source_id))
    match = re.search(r"(?:\bid\s*:\s*'" + escaped + r"'|\"id\"\s*:\s*\"" + escaped + r"\")", source_review_text)
    if match is None:
        return False
    next_entry = re.search(r"\n\s*\{", source_review_text[match.end():])
    array_end = source_review_text.find("\n];", match.end())
    ends = [v for v in ((match.end() + next_entry.start()) if next_entry else -1, array_end) if v >= 0]
    end = min(ends) if ends else len(source_review_text)
    block = source_review_text[match.start():end]
    return bool(re.search(r"(?:\bstatus\s*:\s*'pending'|\"status\"\s*:\s*\"pending\")", block))


def filter_training_order(train_rows: Sequence[dict], excluded_ids: Sequence[str], cursor: int) -> ExclusionOrder:
    """Remove exact training row IDs while preserving all other order positions.

    `cursor` counts attempted rows in this run. This transition requires no
    prior skipped examples so that the target adjustment is unambiguous.
    """
    rows = list(train_rows)
    ids = [str(row["id"]) for row in rows]
    excluded = [str(value) for value in excluded_ids]
    if len(set(ids)) != len(ids):
        raise ValueError("training permutation contains duplicate row IDs")
    if not excluded or len(set(excluded)) != len(excluded):
        raise ValueError("exclusion IDs must be a nonempty unique list")
    if not isinstance(cursor, int) or cursor < 0 or cursor > len(rows):
        raise ValueError("cursor must be within the first training permutation")
    missing = set(excluded) - set(ids)
    if missing:
        raise ValueError(f"excluded rows are not in the training split: {sorted(missing)}")

    positions = {row_id: index for index, row_id in enumerate(ids)}
    consumed = tuple(row_id for row_id in excluded if positions[row_id] < cursor)
    future = tuple(row_id for row_id in excluded if positions[row_id] >= cursor)
    filtered = [row for row in rows if str(row["id"]) not in set(excluded)]
    new_cursor = cursor - len(consumed)
    remaining_old = [row_id for row_id in ids[cursor:] if row_id not in set(excluded)]
    remaining_new = [str(row["id"]) for row in filtered[new_cursor:]]
    if remaining_new != remaining_old:
        raise AssertionError("filtered order does not preserve the exact remaining old permutation")
    return ExclusionOrder(
        rows=filtered,
        previous_cursor=cursor,
        cursor=new_cursor,
        consumed_removed_ids=consumed,
        future_removed_ids=future,
        previous_order_sha256=ids_digest(ids),
        filtered_order_sha256=ids_digest([str(row["id"]) for row in filtered]),
    )


def exclusion_target(old_target_examples: int, trained_examples: int,
                     future_removed_count: int) -> int:
    """Keep consumed exposure; subtract only excluded rows still ahead of cursor."""
    if min(old_target_examples, trained_examples) < 0 or future_removed_count < 0:
        raise ValueError("example counts must be nonnegative")
    if trained_examples > old_target_examples:
        raise ValueError("checkpoint has trained beyond its original target")
    target = old_target_examples - future_removed_count
    if target < trained_examples:
        raise ValueError("exclusion would put target behind the durable checkpoint")
    return target


def validate_exclusion_checkpoint_resume(manifest: dict, state: dict,
                                         current_checkpoint_files: dict[str, str],
                                         *, accumulation: int,
                                         manifest_sha256: str) -> bool:
    """Validate either the staged boundary or a later checkpoint on the same transition.

    The parent artifact hashes are exact at the transition boundary. Once training
    advances, optimizer/model/RNG bytes must change; later resumes instead prove
    the same file layout, transition identity, monotone cursor, and one consumed
    row per new trained example.

    Returns True only for the untouched transition-boundary checkpoint.
    """
    if accumulation < 1:
        raise ValueError("exclusion resume accumulation must be positive")
    parent = manifest.get("checkpoint", {})
    marker = state.get("exclusion_transition", {})
    files = parent.get("files", {})
    if not isinstance(files, dict) or "state.json" not in files:
        raise ValueError("exclusion manifest lacks parent checkpoint artifact pins")
    expected_files = {name: sha for name, sha in files.items() if name != "state.json"}
    if set(current_checkpoint_files) != set(expected_files):
        raise ValueError("checkpoint file layout differs from exclusion parent")

    parent_step = int(parent.get("step", -1))
    parent_cursor = int(parent.get("cursor", -1))
    parent_examples = int(parent.get("trained_examples", -1))
    transition_cursor = parent_cursor - len(marker.get("consumed_removed_ids", []))
    if (parent_step < 0 or parent_cursor < 0 or parent_examples < 0 or
            marker.get("parent_step") != parent_step or
            marker.get("parent_cursor") != parent_cursor or
            marker.get("cursor") != transition_cursor or
            marker.get("trained_examples_preserved") != parent_examples or
            marker.get("parent_state_sha256") != parent.get("state_sha256") or
            marker.get("manifest_sha256") != manifest_sha256):
        raise ValueError("exclusion transition marker differs from approved parent")

    trained = state.get("trained_examples")
    cursor = state.get("cursor")
    step = state.get("step")
    if (not isinstance(trained, int) or not isinstance(cursor, int) or not isinstance(step, int) or
            trained < parent_examples or cursor < transition_cursor or step < parent_step or
            state.get("skipped", 0) != 0):
        raise ValueError("exclusion checkpoint moved behind its transition boundary")
    example_delta = trained - parent_examples
    cursor_delta = cursor - transition_cursor
    if cursor_delta != example_delta:
        raise ValueError("exclusion checkpoint cursor and trained-example progress diverged")
    expected_step = parent_step + (example_delta + accumulation - 1) // accumulation
    if step != expected_step:
        raise ValueError("exclusion checkpoint step does not match consumed examples")

    at_boundary = example_delta == 0
    if at_boundary:
        if step != parent_step or cursor != transition_cursor or current_checkpoint_files != expected_files:
            raise ValueError("staged exclusion checkpoint artifacts differ from exact parent bytes")
    return at_boundary


def exclusion_split_manifest(previous_split: dict, filtered_train_rows: Sequence[dict],
                             excluded_ids: Sequence[str]) -> dict:
    """Describe the same held-out group assignment with selected train rows removed."""
    result = dict(previous_split)
    rows = list(filtered_train_rows)
    programs = sorted({str(row.get("program_id") or row["id"].rsplit("-", 1)[0]) for row in rows})
    if not set(previous_split.get("held_programs", ())).isdisjoint(programs):
        raise ValueError("filtered training rows overlap the protected held-out programs")
    result["train_programs"] = programs
    result["train_turns"] = len(rows)
    result["exclusion"] = {"version": 1, "row_ids_sha256": ids_digest(list(excluded_ids)),
                           "excluded_rows": len(excluded_ids),
                           "heldout_assignment_preserved": True}
    return result


def post_exclusion_mix_summary(data_path, train_rows: Sequence[dict],
                              excluded_ids: Sequence[str], target: float) -> dict:
    """Recompute the reducer mix over the exact training rows after exclusion."""
    selected = {str(row["id"]) for row in train_rows} - set(map(str, excluded_ids))
    counts, families, seen = Counter(), Counter(), set()
    for row in records(data_path):
        row_id = str(row["id"])
        if row_id in selected:
            if row_id in seen:
                raise ValueError(f"duplicate row ID while deriving exclusion mix: {row_id}")
            seen.add(row_id)
            counts[modality(row)] += 1
            families[row.get("task_family", row.get("family", "unknown"))] += 1
    if seen != selected:
        raise ValueError("filtered training IDs do not match physical source rows")
    total = sum(counts.values())
    reducers = counts["directory-reducer"] + counts["file-reducer"]
    share = reducers / total if total else 0.0
    return {"version": "natlang.training_source_exclusion_mix/1",
            "measure": "unique admitted train rows after exact source exclusion",
            "training_rows": total, "excluded_rows": len(excluded_ids),
            "reducer_rows": reducers, "reducer_share": share,
            "target_reducer_share": target, "target_met": bool(total) and share >= target,
            "by_modality": dict(counts), "by_task_family": dict(families),
            "selected_row_ids_sha256": ids_digest(sorted(selected))}


def transition_state(parent_state: dict, *, exclusion_manifest_sha256: str | None,
                     order: ExclusionOrder, target_examples: int,
                     split_sha256: str | None = None,
                     parent_state_sha256: str | None = None) -> dict:
    """Create a copied-checkpoint state update without changing optimizer files."""
    state = dict(parent_state)
    corpus = parent_state.get("corpus")
    if not isinstance(corpus, dict) or not isinstance(corpus.get("steps"), int):
        raise ValueError("parent checkpoint has no intact scheduler horizon")
    if parent_state.get("skipped", 0) != 0:
        raise ValueError("exclusion transition currently requires zero skipped rows")
    if parent_state.get("cursor") != order.previous_cursor:
        raise ValueError("exclusion order cursor differs from parent checkpoint")
    if parent_state.get("trained_examples") != order.previous_cursor:
        raise ValueError("exclusion transition requires one trained example per consumed cursor row")
    if exclusion_manifest_sha256 is not None and len(exclusion_manifest_sha256) != 64:
        raise ValueError("exclusion manifest must have a SHA-256 identity")

    updated_corpus = dict(corpus)
    updated_corpus["target_examples"] = target_examples
    if split_sha256 is not None:
        updated_corpus["split_sha256"] = split_sha256
    if exclusion_manifest_sha256 is not None:
        updated_corpus["exclusion_manifest_sha256"] = exclusion_manifest_sha256
    state["cursor"] = order.cursor
    # Preserve the already completed examples, step, optimizer and RNG history.
    state["corpus"] = updated_corpus
    transition = {
        "version": 1,
        "parent_step": parent_state["step"],
        "parent_state_sha256": parent_state_sha256,
        "parent_corpus_identity": corpus,
        "parent_cursor": order.previous_cursor,
        "cursor": order.cursor,
        "trained_examples_preserved": parent_state["trained_examples"],
        "target_examples": target_examples,
        "schedule_steps_preserved": corpus["steps"],
        "consumed_removed_ids": list(order.consumed_removed_ids),
        "future_removed_ids": list(order.future_removed_ids),
        "optimizer_scheduler_rng_files_unchanged": True,
    }
    if exclusion_manifest_sha256 is not None:
        transition["manifest_sha256"] = exclusion_manifest_sha256
    state["exclusion_transition"] = transition
    return state


def bind_transition_manifest_sha(state_update: dict, manifest_sha256: str) -> dict:
    """Bind a state update to the final immutable manifest bytes without editing input."""
    if len(manifest_sha256) != 64:
        raise ValueError("exclusion manifest must have a SHA-256 identity")
    state = copy.deepcopy(state_update)
    if not isinstance(state.get("corpus"), dict) or not isinstance(state.get("exclusion_transition"), dict):
        raise ValueError("state update is not a source-exclusion transition")
    state["corpus"]["exclusion_manifest_sha256"] = manifest_sha256
    state["exclusion_transition"]["manifest_sha256"] = manifest_sha256
    return state
