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
