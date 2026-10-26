---
args: {"source": "EvidenceItem", "prior": "OperativeUnit[]"}
returns: "OperativeUnit[]"
---
Locate every operative clause, preserving its exact quotation and source version. Check each span through verifyQuotedSpan(source.text, start, end, quotation). Bind actors and consequential actions, then call resolveNestedExceptionScope(units, source) before returning. A failed span requires repair from the original text, not a paraphrased quotation. Conditions and unless clauses must remain explicit nodes with their scoped targets.
