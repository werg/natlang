# Pop teacher-generation closeout (2026-10-15)

## Owner decision and boundary

The prepared Luna seed 52118 queues finished 93/93 and were curated, registered, rendered for both target tokenizers, and exact SHA verified on DGX. No Pop generation workers remain. The seed 52119 campaign directory does not exist. DGX owns its own generation stop point and reports it through `scripts/coord.py`.

The Pop `natlang-pop-luna-replenish.timer` and `natlang-pop-free-gemini-replenish.timer` are disabled and should stay disabled until a new generation decision. Ordinary Luna seed 52118 used the v3 frozen runtime pinned in its `runtime-pin.json`. The prior Luna repair cycle yielded no admitted roots or preference pairs; its discarded model outputs and duplicate review copies were deleted. The raw and selected handoff source files remain because the admitted Sol pilot references their exact path and hashes.

The final three Pop seed cohorts are all registered and SSH SHA verified on DGX:

| Seed | Admitted roots / native decisions | Source corpus | LFM/Mellum render corpus |
| --- | ---: | --- | --- |
| 52116 | 79 / 348 | `luna-held-shapes-s52116-curated-20261015-v1` | `luna-held-shapes-s52116-render-20261015-v1` |
| 52117 | 74 / 333 | `luna-held-shapes-s52117-curated-20261015-v1` | `luna-held-shapes-s52117-render-20261015-v1` |
| 52118 | 71 / 274 | `luna-held-shapes-s52118-curated-20261015-v1` | `luna-held-shapes-s52118-render-20261015-v1` |

These three cohorts add 224 independently reviewed roots and 955 native decisions before any global union filtering. Registry and remote copy are complete; training-recipe admission into the next union remains a separate DGX-owned check.

## Verify the stopped state or recover an interrupted seed

Read coordination first with your own reader identity, then verify Pop owns the work:

```sh
python3 scripts/coord.py --as pop-<agent-role> inbox --ack
systemctl --user is-enabled natlang-pop-luna-replenish.timer natlang-pop-free-gemini-replenish.timer
```

Seed 52118 is already terminal. If an interruption or restore ever requires replaying its exact queues, use the existing source and queues in `/srv/storage/natlang-artifacts/free-provider-generation-20261013-v1/provider-luna-held-shapes-s52118-v1`. To reconcile just that prepared campaign without invoking the generic replenisher:

```sh
PYTHONPATH=scripts python3 - <<'PY'
from pathlib import Path
from scripts.replenish_luna_held_shapes import complete
campaign = Path('/srv/storage/natlang-artifacts/free-provider-generation-20261013-v1/provider-luna-held-shapes-s52118-v1')
print('terminal:', complete(campaign))
PY
```

`complete(campaign)` returns true for the current terminal state. It reconciles exact queue journals and launches missing worker units under the memory ledger if a restore is incomplete. Inspect any `needs_review` or noncomplete result before deciding a retry or an exclusion. Running `scripts/replenish_luna_held_shapes.py` itself can prepare the next seed when the current campaign closes.

For a future seed, collect accepted parents with `scripts/collect_luna_accepted_parents.py`, passing the immediately preceding seed's accepted parents as `--previous-parents` and `runs/natlang-held-task-packet-v3-20261012/cases.jsonl` as `--held`. Materialize native turns, run `curate_native_teacher_turns.py` once with initial exact semantic holds, run `audit_teacher_action_shape.py`, inspect long and suspicious roots, then curate again with structural and manual exact whole-parent holds. Register the final source with `register_corpora.py`, ensuring only `approved-turns.jsonl` is SFT input. Render LFM/Mellum program-header and whole-trajectory twins with `render_native_turn_corpus.py`, register with `register_native_turn_render.py --via-worktree`, and `sync_training_corpora.py` both source and render IDs to DGX. Verify remote SHA receipts. Remove only unregistered native/preaudit duplicates after registration and remote verification. The seed 52116–52118 review directories contain source entries and semantic-review examples.

## Restart later

Before restarting generation, check `plans/HANDOVER.md`, this document, the current source-review manifest and `scripts/coord.py` inbox/status on both machines. Confirm the training union actually admits the newly registered corpus IDs; a registry entry or successful copy does not prove union admission. Decide whether to revise folder-reducer cases, which frequently pass the oracle through crisp semantic shortcuts, and whether to improve answer normalization for numeric-with-context and verbose Hotpot responses. Rejected rows with answer-equivalence or provider gateway errors are not DPO negatives.

Future repair queues use `gpt-6-sol` with medium reasoning as pinned in their new campaign manifests by commit `bef7edf942`; ordinary generation remains Luna low. The four-case Sol pilot admitted two clean roots and its source/render corpora are registered. The Pop full gate for that change passed focused and TypeScript checks but exposed two shared neuralese attention-kernel parity failures owned by DGX; those failures are tracked in `plans/HANDOVER.md`.

Only after deciding on new cases and a target training need should a Pop owner re-enable the replenisher timer. Run `python3 scripts/coord.py --as pop-<agent-role> inbox --ack` before changing shared resources, and brief DGX with exact corpus IDs and code commit. Keep code synchronized through `origin/main`; use source manifests and `scripts/sync_training_corpora.py` for data, not source-tree copies.
