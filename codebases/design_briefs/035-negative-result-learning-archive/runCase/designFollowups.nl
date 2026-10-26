---
args: { goal: GoalContract, policy: PolicySelection, candidates: FailureExplanation[], evidence: EvidencePage[] }
returns: StageResultDiscriminatingExperiment
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Propose bounded low-risk follow-ups that distinguish rival explanations with minimal changes. Specify what each outcome would mean before execution and avoid repeating the same invalid procedure or adding so many changes that the result becomes uninterpretable.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual DiscriminatingExperiment fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Follow-up has a discriminating prediction and preserves approved safety and resource constraints.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
