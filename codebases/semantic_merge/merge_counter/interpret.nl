---
description: Semantically reconcile operations on a measured count.
args:
  base: State
  updates: Update[]
  policy: Text
returns: Draft
---
Read each update as an intent about the measured `base.value` and `base.unit`.
"Add three" can compose with an independent increment; "the correct total is three"
may instead replace a mistaken earlier value. Do not treat every number as an increment,
nor impose an arithmetic CRDT rule. If a correction and another operation are ambiguous
in order or meaning, retain alternatives. Return the typed value, exact applied IDs and
an explanation of the interpretation.
