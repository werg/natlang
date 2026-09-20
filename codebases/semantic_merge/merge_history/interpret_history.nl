---
description: Decide how all causally presented changes affect the meaning of the base document.
args:
  base: Document
  updates: Update[]
  policy: Text
returns: Draft
---
Reconcile `args/updates` against `args/base/text` under `args/policy`. This is a semantic
decision: read what each author intends; do not treat textual proximity as sufficient.
Write the merged document to `text`. Put an update ID in `applied` only when its intended
change is represented in that document. For unresolved or incompatible intents, put every
affected ID in an `alternatives` entry with a candidate proposal and reason. Explain the
decision briefly. Account for every update exactly once; the exact validator will reject
missing, unknown or multiply counted IDs. Do not invent a deterministic convergence rule.
