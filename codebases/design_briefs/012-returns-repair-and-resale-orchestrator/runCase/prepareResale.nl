---
args: { goal: GoalContract, policy: PolicySelection, candidates: CustodyEvent[], evidence: EvidencePage[] }
returns: StageResultResaleDisclosure
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Produce a truthful resale or disposal packet from verified work and remaining defects. Reconcile grade terminology with marketplace rules, remove prior-owner data, and hold listing until required functional checks and ownership transfer are complete.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ResaleDisclosure fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Disclosure covers known defects; data-removal and title checks pass before listing.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
