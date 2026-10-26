---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftSourceFieldMeaning
---
Construct a SourceFieldMeaning draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Reconcile dictionaries, sample records, application notes, and domain-owner explanations. Identify fields whose meaning depends on record type or history, and distinguish null, not applicable, unknown, and deliberately withheld values before proposing transformations.
