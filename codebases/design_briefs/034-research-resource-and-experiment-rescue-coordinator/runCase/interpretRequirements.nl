---
args: { goal: GoalContract, policy: PolicySelection, candidates: ResearchSchedule[], evidence: EvidencePage[] }
returns: StageResultExperimentRequirement
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Read approved protocols and researcher requests to distinguish essential experimental conditions from convenient preferences. Identify sample timing, equipment capabilities, and required personnel without changing scientific procedures or inferring unapproved substitutions.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ExperimentRequirement fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Essential conditions are source-backed and unknown compatibility remains blocked.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
