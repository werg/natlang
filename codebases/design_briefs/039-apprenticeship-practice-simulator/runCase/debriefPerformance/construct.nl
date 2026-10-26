---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftPracticeFeedback
---
Construct a PracticeFeedback draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Compare the observable trajectory with procedure requirements and scenario facts. Explain specific effective and ineffective decisions, distinguish outcome luck from sound process, and propose a next scenario targeting demonstrated gaps rather than generic advice.
