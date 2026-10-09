---
description: Build invalidation, cache validity (natural-language implementation). Whether a task's recorded outputs are still valid, from the digests the workspace sees now and the ledger entry of its last successful run.
args:
  task: Task
  evidence: Evidence
returns: CheckedValidity
---
Decide whether task can be settled from its recorded run instead of running again. evidence holds what the workspace
sees now (fingerprint, inputs, outputs, each file's sha256, null when the file is missing) and what it recorded
(evidence.recorded, null when there is no record). Compare exactly in eval, in this order, and stop at the first
mismatch:

1. evidence.recorded is null: invalid, reason "no record of an earlier run".
2. evidence.recorded.fingerprint differs from evidence.fingerprint: invalid, reason "declaration changed".
3. For each of evidence.inputs, in order: the recorded input with the same path is missing, or its sha256 differs
   from the current sha256: invalid, reason "input changed: " and the path. A current sha256 of null differs.
4. For each of evidence.outputs, in order: a current sha256 of null: invalid, reason "output missing: " and the path.
   The recorded output with the same path is missing, or its sha256 differs: invalid, reason "output modified: "
   and the path.
5. No mismatch: valid, reason "up to date".

Return { valid, reason }.
