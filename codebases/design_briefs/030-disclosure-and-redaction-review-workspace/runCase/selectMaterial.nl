---
args: { goal: GoalContract, policy: PolicySelection, candidates: DisclosureScope[], evidence: EvidencePage[] }
returns: StageResultDisclosureSelection
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Identify responsive content at passage level, retaining enough context for accurate interpretation. Separate nonresponsive material from restricted responsive material so reviewers can see whether an omission is about relevance or permission.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual DisclosureSelection fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: No whole-document inclusion merely because one passage is relevant.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
