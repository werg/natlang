---
args: { candidate: WorkpaperConclusion, decisions: DecisionsWorkpaperSignoff, evidence: EvidencePage[] }
returns: DraftWorkpaperSignoff
---
1. Construct an evidence-backed WorkpaperSignoff record from established atoms, using the fields declared for WorkpaperSignoff in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Bind sign-off to exact source and conclusion versions. When a late journal or source correction arrives, trace affected assertions and invalidate only dependent approvals; preserve the prior signed workpaper as historical evidence.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
