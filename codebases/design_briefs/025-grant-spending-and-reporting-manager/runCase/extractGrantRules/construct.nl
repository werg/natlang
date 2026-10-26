---
args: { candidate: GrantReport, decisions: DecisionsGrantCondition, evidence: EvidencePage[] }
returns: DraftGrantCondition
---
1. Construct an evidence-backed GrantCondition record from established atoms, using the fields declared for GrantCondition in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Parse the agreement, budget schedules, and amendments into spending, milestone, and reporting conditions. Distinguish required outcomes from aspirational impact language, and preserve restrictions that apply only to specific funding tranches or time periods.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
