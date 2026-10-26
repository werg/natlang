---
args: { goal: GoalContract, policy: PolicySelection, candidates: CodeDefinition[], evidence: EvidencePage[] }
returns: StageResultCodingDecision
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Code bounded passages with their necessary context, then compare independent assessments on a sampled overlap. Investigate disagreements as possible codebook defects; when a definition changes, mark all affected earlier decisions stale before recoding.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual CodingDecision fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Code-version consistency and no silent replacement of disagreement with majority confidence.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
