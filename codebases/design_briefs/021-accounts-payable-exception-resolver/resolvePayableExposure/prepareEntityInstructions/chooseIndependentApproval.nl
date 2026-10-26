---
args: {"entity": "string", "submitter": "string", "amount": "Money", "delegations": "PolicyCandidate[]"}
returns: "string"
---
Resolve the current entity and amount-specific approval delegation from supplied signed rules. Exclude the claimant, preparer and any role explicitly barred from approving this liability. Compare thresholds in the declared currency using exact minor units only. Return the named independent role or person; if no applicable authority can be established, return a precise unresolved approval question rather than selecting the submitter.
