# Preserve consumed updates when data and epoch coverage change together

The existing data-continuation program reconstructs the predecessor schedule, records consumed document identities, and schedules only the unseen new-data suffix. Its placeholder prefix represents completed optimizer updates, not new data batches.

1. Retain that declared continuation mechanism and exact checkpoint step/input checks.
2. Verify an epoch-only handoff by comparing every consumed batch, as before.
3. For a simultaneous declared data continuation, verify the recorded consumed-update count and empty prefix instead of comparing the predecessor batches against new-data placeholders. Keep unsupported epoch/QAT checks for epoch-only handoffs.
4. Record the verification route in the existing epoch-budget event. This is crisp exact-state validation; source admission and training policy remain unchanged.
