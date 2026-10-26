---
args: { candidate: ContributionTask, decisions: DecisionsResponseSection, evidence: EvidencePage[] }
returns: DraftResponseSection
---
1. Construct an evidence-backed ResponseSection record from established atoms, using the fields declared for ResponseSection in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Draft consistent answers using supported claims and approved commercial positions. Tailor explanations to the customer’s evaluation criteria without inventing facts; preserve declared exceptions and reconcile promises across technical, pricing, and delivery sections.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
