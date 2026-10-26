---
args: { goal: GoalContract, policy: PolicySelection, candidates: CodingDecision[], evidence: EvidencePage[] }
returns: StageResultResearchFinding
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Build findings around supported mechanisms and recurring patterns, actively search for counterexamples, and distinguish an absent observation from evidence of absence. Return unresolved questions and a traceable path from each conclusion to the coded passages.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ResearchFinding fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: No causal or population-level claim beyond the study design; counterevidence cannot be omitted.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
