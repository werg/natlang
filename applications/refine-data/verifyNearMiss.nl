---
description: Check, independently of the writer, that an edited value fails a property its original satisfies.
args:
  predicate: string
  original: unknown
  edited: unknown
returns: Verification
---
Decide for each of original and edited, on its own, whether the value is predicate. Judge only the values against the
property. Then say whether edited differs from original in one detail (minimal) and name in reason what decides the two
verdicts. Return { original_holds, edited_holds, minimal, reason }.
