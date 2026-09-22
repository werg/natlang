---
description: Semantically reconcile membership of a set-like collection.
args:
  base: State
  updates: Update[]
  policy: string
returns: Draft
---
Interpret adds, removals, renames and references to possibly synonymous members.
Decide whether two labels really denote one member from supplied context, not from
spelling alone. A concurrent add and remove of the same member may express incompatible
intentions; keep them as alternatives unless `policy` resolves them. Return a duplicate-
free typed member list, exact applied update IDs and an explanation.
