---
args: { goal: GoalContract, policy: PolicySelection, candidates: SourcingPackage[], evidence: EvidencePage[] }
returns: StageResultCounteroffer
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Draft conditional counteroffers that exchange permitted concessions for improvements in the whole package. Keep supplier-confidential information isolated and make dependencies explicit so acceptance of one term cannot accidentally commit the buyer to an unapproved package.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual Counteroffer fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: No disclosure of rival bids or offer beyond the signed negotiation mandate.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
