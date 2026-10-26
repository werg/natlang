---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftSiteInstruction
---
Construct a SiteInstruction draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Reconcile site notes, messages, and drawing annotations to identify the actual requested action and its issuer. Distinguish clarification, proposed change, and authorized instruction; preserve conflicting versions and do not infer commercial authority from site presence.
