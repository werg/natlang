---
args: { goal: GoalContract, policy: PolicySelection, candidates: EvidenceFinding[], evidence: EvidencePage[] }
returns: StageResultDisputeRecommendation
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Apply supplied marketplace rules to established facts and explicitly conditional alternatives for unresolved facts. Explain how the outcome changes under plausible resolutions and route cases outside delegated interpretation to authorized reviewers.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual DisputeRecommendation fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Outcome follows the selected rule edition; uncertainty cannot silently become an adverse finding.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
