# Pop teacher-generation closeout (2026-10-15)

## Owner decision and boundary

Finish the already prepared Luna seed 52118 queues, curate and synchronize their accepted trajectories, then leave generation stopped. Seed 52117 is already terminal and undergoing curation. Do not let the generic replenisher prepare seed 52119 or another repair cycle as part of this closeout. DGX owns its own generation stop point and reports it through `scripts/coord.py`.

The Pop `natlang-pop-luna-replenish.timer` and `natlang-pop-free-gemini-replenish.timer` are disabled. They must remain disabled after the final corpus sync. Pausing the worker queues for a full code check does not cancel their source-pinned cases; resume only seed 52118's exact queue directories. Ordinary Luna seed 52118 uses the v3 frozen runtime pinned in its `runtime-pin.json`. An older repair campaign remains a quality-held review artifact, not an admitted corpus.

## Finish the exact pending campaign

Read coordination first with your own reader identity, then verify Pop owns the work:

```sh
python3 scripts/coord.py --as pop-<agent-role> inbox --ack
systemctl --user is-enabled natlang-pop-luna-replenish.timer natlang-pop-free-gemini-replenish.timer
```

For seed 52118, use the existing source and queues in `/srv/storage/natlang-artifacts/free-provider-generation-20261013-v1/provider-luna-held-shapes-s52118-v1`. To resume just that prepared campaign without invoking the generic replenisher:

```sh
PYTHONPATH=scripts python3 - <<'PY'
from pathlib import Path
from scripts.replenish_luna_held_shapes import complete
campaign = Path('/srv/storage/natlang-artifacts/free-provider-generation-20261013-v1/provider-luna-held-shapes-s52118-v1')
print('terminal:', complete(campaign))
PY
```

`complete(campaign)` reconciles exact queue journals and launches missing worker units under the memory ledger. Repeat after running units finish; inspect any `needs_review` or noncomplete result before deciding a retry or an exclusion. Never run `scripts/replenish_luna_held_shapes.py` itself during closeout: its normal tail prepares the next seed when the current campaign closes.

Once terminal, collect accepted parents with `scripts/collect_luna_accepted_parents.py`, passing seed 52117's accepted parents as `--previous-parents` and `runs/natlang-held-task-packet-v3-20261012/cases.jsonl` as `--held`. Materialize native turns, run `curate_native_teacher_turns.py` once with empty exact semantic holds, run `audit_teacher_action_shape.py`, inspect long and suspicious roots, then curate again with structural and manual exact whole-parent holds. Register only the final `approved-turns.jsonl` source with `register_corpora.py`. Render LFM/Mellum program-header and whole-trajectory twins with `render_native_turn_corpus.py`, register with `register_native_turn_render.py --via-worktree`, and `sync_training_corpora.py` both source and render IDs to DGX. Verify remote SHA receipts. Remove only unregistered native/preaudit duplicates after registration and remote verification. The seed 52116/52117 review directories contain exact command patterns, source entries and semantic-review examples.

## Restart later

Before restarting generation, check `plans/HANDOVER.md`, this document, the current source-review manifest and `scripts/coord.py` inbox/status on both machines. Confirm the training union actually admits the newly registered corpus IDs; a registry entry or successful copy does not prove union admission. Decide whether to revise folder-reducer cases, which frequently pass the oracle through crisp semantic shortcuts, and whether to improve answer normalization for numeric-with-context and verbose Hotpot responses. Rejected rows with answer-equivalence or provider gateway errors are not DPO negatives.

Future repair queues should use `gpt-6-sol` with medium reasoning as pinned in their new campaign manifests; ordinary generation remains Luna low. The four-case Sol pilot admitted two clean roots and its source/render corpora are registered. Verify the pending shared-code publication that adds this repair-teacher pin before starting another repair cycle.

Only after deciding on new cases and a target training need should a Pop owner re-enable the replenisher timer. Run `python3 scripts/coord.py --as pop-<agent-role> inbox --ack` before changing shared resources, and brief DGX with exact corpus IDs and code commit. Keep code synchronized through `origin/main`; use source manifests and `scripts/sync_training_corpora.py` for data, not source-tree copies.
