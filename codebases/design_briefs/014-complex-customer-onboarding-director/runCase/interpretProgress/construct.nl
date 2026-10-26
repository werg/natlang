---
args: { candidate: OnboardingTask, decisions: DecisionsProgressFinding, evidence: EvidencePage[] }
returns: DraftProgressFinding
---
1. Construct an evidence-backed ProgressFinding record from established atoms, using the fields declared for ProgressFinding in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Read system events and participant updates against each task’s completion contract. Distinguish an attempted import from a reconciled migration and attendance from demonstrated setup capability; request the smallest missing verification artifact.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
