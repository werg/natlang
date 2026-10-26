---
args: { candidate: AmendmentCase, decisions: DecisionsAmendmentTarget, evidence: EvidencePage[] }
returns: DraftAmendmentTarget
---
1. Construct an evidence-backed AmendmentTarget record from established atoms, using the fields declared for AmendmentTarget in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Read amendment language against the referenced agreement edition. Resolve section references, defined terms, and indirect targets; distinguish correction of wording from a substantive change and retain ambiguous references for review rather than choosing the nearest heading.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
