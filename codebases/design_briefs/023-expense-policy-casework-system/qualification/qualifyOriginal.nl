---
args: { draft: DraftExpenseLine, candidate: ExpenseCase, evidence: EvidencePage[], policy: PolicySelection }
returns: GateVerdict
readout: decision
---
1. Apply the application acceptance gate to the candidate (candidate: ExpenseCase): exact totals must reconcile to source receipts; unverified values remain blocked. Weigh the draft (draft: DraftExpenseLine), evidence (evidence: EvidencePage[]) and policy (policy: PolicySelection).

2. Return "pass" only if source evidence (evidence: EvidencePage[]) establishes each required condition for the candidate (candidate: ExpenseCase), with the applicable authority, effective period, and scope (policy: PolicySelection).

3. Return "fail" if decisive evidence (evidence: EvidencePage[]) establishes that a required condition is false, or if the draft (draft: DraftExpenseLine) contradicts an authoritative in-scope record (policy: PolicySelection).

4. Return "needs-review" if required evidence (evidence: EvidencePage[]) is missing, contradictory, out of date, outside scope, or lacks established authority (policy: PolicySelection). Do not turn unsearched material into negative evidence.
