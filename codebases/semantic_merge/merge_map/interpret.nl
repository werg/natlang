---
description: Semantically reconcile edits to a keyed map.
args:
  base: State
  updates: Update[]
  policy: Text
returns: Draft
---
Interpret each update's intended change to `base.fields`. A key rename may preserve the
identity of a field; equal spelling does not prove equal meaning. Distinguish independent
keys from conflicting claims about one fact. Return the proposed typed map, exact applied
update IDs, unresolved alternatives with affected IDs, and a short explanation. Preserve
all unresolved source text in alternative proposals. Follow `policy`; do not use a last
writer rule unless the policy explicitly requests one.
