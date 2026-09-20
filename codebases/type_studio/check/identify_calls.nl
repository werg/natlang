---
description: Propose bounded call paths from pseudocode and cite actual witness IDs where present.
args:
  target: Target
  context: Context
returns: CallClaim[]
---
Identify up to eight plausible calls in `target.body`. For a trace-backed call, use an
exact `evidence_id` from `context.witnesses` and copy its observed argument types.
For a path inferred only from prose, leave `evidence_id` empty and explain the
hypothesis. Do not present an inferred path as an executed call. The host will check
which claims actually match frozen witnesses and known callee signatures.
