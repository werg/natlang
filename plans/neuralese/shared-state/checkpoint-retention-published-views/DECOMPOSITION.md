# Retain current published full state and its inference consumers

The owner's storage policy remains natural language: keep current verified
recovery, required parents and actual evaluation/publication dependencies;
remove superseded checkpoint exports after replacement durability is verified.
A filename from an older lineage is not a stronger recovery authority.

1. Read the explicitly selected current recovery and consumer declarations.
2. Authenticate private readback, exact content identity, role, storage run,
   actual listed object and declared step. Native filenames encode their step;
   owned content-addressed full publications obtain it from the selected state
   or exact inference-parent declaration. Never invent a filename alias as
   evidence that the actual object exists.
3. Retain the actual full object named by each verified inference derivation,
   pending publication and active consumer. Preserve identity aliases through
   the existing shared custody checks.
4. Use the existing retention candidate policy, write exact deletion intents,
   apply only its selected superseded files, and verify removal. Unknown owned
   publications still require explicit discard declarations.

Steps2–4 are crisp helpers because storage identity, custody and deletion must
be exact. This extends the existing shared `checkpoint_retention.py`; it does
not add a per-model mechanism or change training, inference, quality thresholds
or source-data admission. Exercise owned full recovery, current/ahead views,
pending views and damaged custody before production cleanup resumes.
