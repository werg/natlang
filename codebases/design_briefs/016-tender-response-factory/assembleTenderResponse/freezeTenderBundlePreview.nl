---
description: Freeze a review proposal only when exact bundle validation, coverage, quantity and certificate gates agree.
args:{request:BidPreparationRequest;goal:GoalContract;obligations:TenderRequirement[];sections:ResponseSection[];portalLimits:TenderGuardrail;addendum:TenderGuardrail;quantities:TenderGuardrail;certificate:TenderGuardrail}
returns:TenderBundleProposal
---
Return no bundle when any mandatory obligation is uncovered, any guardrail blocks, or the deterministic validation does not bind the current response hash and addendum revision. Otherwise bind case, response hash, file-set hash and addendum revision into the approval proposal. Set portalState not-submitted unless a matching authoritative receipt already exists; an approval or frozen preview is not an upload.
