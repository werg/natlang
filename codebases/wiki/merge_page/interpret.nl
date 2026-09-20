---
args:
  base: WikiPage
  updates: WikiUpdate[]
returns: MergeDraft
---
Reconcile the meaning of edits to each stable block. All updates have the same
base revision and are presented in stable ID order. Preserve compatible
changes, and include unresolved alternatives when meanings conflict. Account
for every update ID exactly once in accounted. Keep the original block IDs.
For a code cell, return runnable source only if the combined meaning is clear;
otherwise preserve an unresolved conflict. No crisp convergence law is
promised. The same model, source, presentation and seed must be pinned by the
host to test repeatability across replicas.
