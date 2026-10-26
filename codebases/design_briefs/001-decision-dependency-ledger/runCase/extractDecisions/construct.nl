---
args: { candidate: DecisionCase, decisions: DecisionsDecisionAssertion, evidence: EvidencePage[] }
returns: DraftDecisionAssertion
---
1. Construct an evidence-backed DecisionAssertion record from established atoms, using the fields declared for DecisionAssertion in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Segment meetings at topic boundaries; distinguish a suggestion, assent, and an authorized resolution. Link explicit objections and preserve the speaker rather than attributing a collective decision from an unchallenged statement.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
