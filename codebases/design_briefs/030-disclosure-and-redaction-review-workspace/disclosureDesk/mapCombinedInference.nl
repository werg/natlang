---
args: { selections: SpanSelection[], inspections: DocumentInspection[] }
returns: RedactionPlan
---
Review selected spans together across documents for identity reconstruction and sensitive facts revealed only in combination. List concrete inference pathways and mitigation; do not assume separate fragments are harmless when combined.
