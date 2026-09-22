---
description: Semantically reconcile permission grants, denials and scope changes.
args:
  base: State
  updates: Update[]
  policy: string
returns: Draft
---
Interpret the named subjects, resources and actions literally. A revocation may refer to
a broad grant or one narrow exception. Do not infer that a later grant author intended to
override a concurrent denial. Keep conflicting authority claims as alternatives unless
`policy` explicitly resolves them. Return typed rules and account for every update ID.
This is a merge proposal; the authorization system must still enforce its own policy.
