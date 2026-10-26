---
args: { candidate: CloseTask, decisions: DecisionsWorkpaperConclusion, evidence: EvidencePage[] }
returns: DraftWorkpaperConclusion
---
1. Construct an evidence-backed WorkpaperConclusion record from established atoms, using the fields declared for WorkpaperConclusion in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Assemble the reconciliation, explanations, and supporting documents into a reviewer-ready workpaper. Distinguish verified facts from management explanations, quantify residual differences exactly, and avoid conclusions broader than the evidence supports.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
