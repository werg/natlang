---
args: { candidate: ResearchProject, decisions: DecisionsInquiryFrame, evidence: EvidencePage[] }
returns: DraftInquiryFrame
---
1. Construct an evidence-backed InquiryFrame record from established atoms, using the fields declared for InquiryFrame in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Interpret the decision that research should inform and decompose it into answerable questions. Identify plausible rival explanations and what the available corpus cannot establish; avoid converting an exploratory question into a predetermined conclusion.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
