---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftBidScope
---
Construct a BidScope draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Read each bid, qualification, schedule, and addendum as a package. Extract functional scope rather than relying on trade headings, distinguish allowances from firm inclusions, and retain conditional exclusions that affect interfaces with other bidders.
