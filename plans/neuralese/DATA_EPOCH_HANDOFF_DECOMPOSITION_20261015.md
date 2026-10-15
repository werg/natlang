# Preserve consumed updates when data and epoch coverage change together

The existing data-continuation program reconstructs the predecessor schedule, records consumed document identities, and schedules only the unseen new-data suffix. Its placeholder prefix represents completed optimizer updates, not new data batches.

1. Retain that declared continuation mechanism and exact checkpoint step/input checks.
2. Verify an epoch-only handoff by comparing every consumed batch, as before.
3. For a simultaneous declared data continuation, verify the recorded consumed-update count and empty prefix instead of comparing the predecessor batches against new-data placeholders. Keep unsupported epoch/QAT checks for epoch-only handoffs.
4. Record the verification route in the existing epoch-budget event. This is crisp exact-state validation; source admission and training policy remain unchanged.

## Reuse prepared windows for data-only continuation plans

1. Ask the existing semantic-key cache for a validated immutable window entry before streaming/tokenizing rows.
2. Project those exact cached windows through the same light-descriptor helper as the streamed path; select the held set using the same shared selector.
3. On absence, unsupported fingerprint or failed validation, retain the existing streamed planner. Both predecessor and new-data planning already call this single helper.
4. Verify descriptor, deterministic batch-plan and held identity equality; no new cache format, source reader or admission policy.

## Add E2E mixing while preserving an absent curriculum

1. Keep the existing explicit objective-continuation declaration and canonical previous-component hashes.
2. When both saved and requested quantization are absent and only E2E mixing changes, record the exact objective change without inventing a quantization curriculum. Backbone weights, optimizer, RNG and data coverage keep their existing restoration mechanisms.
3. Continue refusing removal of an existing curriculum and family changes without a curriculum; existing ramp and member-weight jump checks remain.
4. This is exact state validation in the shared trainer, not a new training policy or evaluator.

## Preserve stacked MuonSR optimizer state when adding projections

1. Keep the shared exact-name optimizer alignment and explicit declaration of fresh projection slots.
2. Validate each MuonSR `grad_norm` against the parameter's leading stack dimensions: the optimizer computes norms over its last two matrix dimensions. Momentum remains full parameter shape.
3. Keep every saved norm and momentum tensor unchanged, including its original dtype through Torch optimizer loading; reject an incorrect stack size or any other undeclared shape change.
4. Verify the actual stacked optimizer's state survives adding a projection, including malformed-state rejection. This is crisp optimizer schema validation; the mixed objective and training policy are unchanged.

## Explicit same-data legacy window-coverage upgrade

The shared continuation planner can already reconstruct a legacy document-level schedule and replay its unknown window coverage conservatively. A same-input resume never entered that path. Add one declared `upgrade_window_coverage` option through the existing recipe windowing handoff and CLI. It selects that existing continuation path only for schema/1 records; subsequent schema/2 resumes remain ordinary resumes. Record the requested upgrade and conservative coverage counts, retain exact checkpoint update/optimizer/RNG, and build the plan on CPU before a useful data handoff. No separate trainer or automatic restart is introduced.
