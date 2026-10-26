---
args: { candidate: RedlineFinding, decisions: DecisionsClauseOffer, evidence: EvidencePage[] }
returns: DraftClauseOffer
---
1. Construct an evidence-backed ClauseOffer record from established atoms, using the fields declared for ClauseOffer in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Assemble a coherent draft, issue list, and internal rationale with confidential fallback positions excluded from the external copy. Bind approval to the exact package and record which issues remain open rather than presenting a partial agreement as final.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
