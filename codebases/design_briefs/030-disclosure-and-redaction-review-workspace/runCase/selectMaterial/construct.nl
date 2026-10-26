---
args: { candidate: DisclosureScope, decisions: DecisionsDisclosureSelection, evidence: EvidencePage[] }
returns: DraftDisclosureSelection
---
1. Construct an evidence-backed DisclosureSelection record from established atoms, using the fields declared for DisclosureSelection in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Identify responsive content at passage level, retaining enough context for accurate interpretation. Separate nonresponsive material from restricted responsive material so reviewers can see whether an omission is about relevance or permission.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
