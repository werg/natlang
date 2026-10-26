---
args: { request: BidPreparationRequest, addenda: EvidencePage[], responses: TenderResponse[] }
returns: TenderGuardrail
---
Order addenda by authenticated issue sequence and identify changed eligibility, response, attachment, and deadline requirements before parsing and freezing the tender obligations. Invalidate only prior approvals depending on changed material; never draft against an unincorporated addendum.
