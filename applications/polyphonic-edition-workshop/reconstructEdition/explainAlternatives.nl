---
description: Explain material differences among actual admissible editions without inventing a winner.
args:
  checked: CheckedEdition[]
  editorial: EditorialIndex
returns: AmbiguityFinding[]
---
1. Select only actual admissible=true editions for the supported alternative comparison. Preserve
   null/false findings elsewhere in the certificate rather than using them as positive ambiguity evidence.
2. Join their source-level claimAssignments by original claim ID. Identify claims whose supported
   target assignments differ, then compare the actual performed events affected by those differences.
3. Explain why the supplied authority, dependency histories and ensemble constraints distinguish some
   readings but leave these targets unresolved. Shared numerical totals alone do not prove equivalence.
4. Return an AmbiguityFinding per material unresolved source claim with original target IDs, differing
   written event IDs, original witnesses and a concise reason. Repeated occurrences may expose the same
   distinction repeatedly but do not count as independent editorial choices.
5. A single supported edition needs no invented ambiguity. Several behavior-equivalent serializations
   of one edition need no new source hypothesis; generated trace IDs are presentation only.
