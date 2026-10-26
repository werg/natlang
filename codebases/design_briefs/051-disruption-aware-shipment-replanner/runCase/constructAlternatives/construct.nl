---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftShippingAlternative
---
Construct a ShippingAlternative draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Generate permitted rerouting, split-shipment, consolidation, or substitution options. Explain which commitments each option preserves or changes, and exclude alternatives violating handling requirements or relying on unverified carrier capacity.
