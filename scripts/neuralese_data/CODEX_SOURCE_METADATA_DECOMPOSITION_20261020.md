# Preserve exact Codex bookkeeping without inventing conversation

The existing acquired-session policy uses original full trajectories and rejects
incomplete causal context. Codex Git snapshots describe source bookkeeping;
they are neither assistant messages nor observations of a model tool call.

1. The shared Codex reader validates the released snapshot's exact field shapes.
2. It records the original row and canonical payload digest as auxiliary source
   evidence. Original source bytes remain bound by the acquisition manifest.
3. It keeps rollback/abort rows as unresolved history evidence in the existing
   missing-ancestry field. The unchanged quality gate prevents positive admission
   until an exact history reconstruction supports that source.
4. It retains all original conversational messages and call/result identities.
   Unknown response item kinds and malformed snapshots fail closed.

These are crisp protocol/schema and provenance verifiers. No new task,
model-facing text, generated target, communication edge or success is authored.
Successful normalization is not training admission.
