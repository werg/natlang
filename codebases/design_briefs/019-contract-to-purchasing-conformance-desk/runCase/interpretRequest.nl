---
args: { goal: GoalContract, policy: PolicySelection, candidates: PurchaseReview[], evidence: EvidencePage[] }
returns: StageResultPurchaseIntent
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Read item descriptions with the requester’s purpose and supporting documents. Distinguish a legitimate functional need from a preferred brand or category label, and identify ambiguity that changes contract coverage or approval authority.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual PurchaseIntent fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Quantity and purpose are explicit; missing critical scope is not guessed.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
