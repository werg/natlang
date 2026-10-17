# Continue after an already applied withdrawal

Owner policy remains comprehensive adaptive training with preserved optimizer,
RNG and earned exposure. A later launch may retain an immutable withdrawal proof
while its current source already excludes every rejected row. Requiring the old
predecessor in that case blocked LFM's October 17 startup before any update.

The recovery decision is the existing supervisor policy: authenticate the saved
history and resume useful training. The trainer supplies exact verification:

1. Restore the matching SFT frame through the existing shared continuation.
2. Match the requested proof's full SHA against that frame's withdrawal history.
3. For a new withdrawal, require and verify the exact predecessor as before.
4. For an applied withdrawal, reuse the existing verifier without rebuilding the
   obsolete predecessor. It still checks model/schema, evidence hash and absence
   of rejected targets, and returns the identical training state.
5. Preserve the recorded history, optimizer, RNG and visit counts. No past update
   is undone and no rejected target is credited or reintroduced.

This is a crisp extension of existing identity verification, not a new recovery
policy or model-facing change. The actual tiny-model continuation test removes
the predecessor file and argument after application, verifies unchanged exposure
and history, and confirms that subsequently corrupted evidence still fails.
