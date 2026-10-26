---
description: Construct a AgreementReview draft for reviewOutcome.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: AgreementTrial[] }
returns: DraftAgreementReview
---
Assemble the AgreementReview as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Final agreement reflects actual consent and does not erase unresolved disagreement.
