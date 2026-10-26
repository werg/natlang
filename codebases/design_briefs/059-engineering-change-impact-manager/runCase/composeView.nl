---
args: { review: ReviewPacket, input: CaseInput }
returns: ViewModel
---
Compose an accessible audience-specific view. Preserve status and uncertainty; return structured data only.
Return the typed result, preserving missing facts as missing and citing evidence IDs where a decision depends on evidence.
