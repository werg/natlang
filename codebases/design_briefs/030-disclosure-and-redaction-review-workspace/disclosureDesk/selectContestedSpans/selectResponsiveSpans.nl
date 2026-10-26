---
args: { scope: DisclosureScopeDecision, document: DocumentInspection }
returns: SpanSelection
---
Inspect this document's own text and metadata layers. Return only spans responsive to the permitted topic and date window. For each excluded span, state why it falls outside scope. If the permitted fact depends on a qualifier or condition, include that adjacent span; do not extract an unbounded page. Bind every selection to this document's source hash and source references.
