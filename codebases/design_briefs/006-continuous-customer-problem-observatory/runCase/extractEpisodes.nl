---
args: { goal: GoalContract, policy: PolicySelection, candidates: ProblemStudy[], evidence: EvidencePage[] }
returns: StageResultProblemEpisode
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Read enough surrounding conversation to identify the attempted job, the obstacle, and its consequence. Separate a product complaint from a requested feature, and split independently resolvable problems while retaining links to the original interaction.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ProblemEpisode fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Every episode preserves the customer's actual situation and does not invent severity.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
