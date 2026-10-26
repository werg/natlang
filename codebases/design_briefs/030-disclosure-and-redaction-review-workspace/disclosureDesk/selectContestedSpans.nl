---
args: { scope: DisclosureScopeDecision, inspections: DocumentInspection[] }
returns: SpanSelection[]
---
For each document, select only source spans responsive to permitted topics and date window. Preserve source hash and locators. Record exclusion reason for each nonresponsive span; if the requested fact depends on surrounding qualifiers, include the qualifier rather than extracting an unbounded passage.

For each inspected document, call selectResponsiveSpans(scope, document). Append that document's returned permitted and excluded spans to the complete selection set. Do not reuse another document's selection; retain a source reference for every included or excluded span.
