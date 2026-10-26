---
args: { goal: GoalContract, policy: PolicySelection, candidates: IntegrationWave[], evidence: EvidencePage[] }
returns: StageResultInterfaceScenario
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Replay cross-team cases using proposed terminology and responsibilities. Look for steps both teams assume the other performs; turn failures into interface decisions and rerun only affected scenarios before approving a migration wave.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual InterfaceScenario fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: No unowned critical handoff and no source distinction silently discarded.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
