---
args: { goal: GoalContract, policy: PolicySelection, candidates: DiscriminatingExperiment[], evidence: EvidencePage[] }
returns: StageResultResearchLesson
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Publish a scoped lesson with conditions under which it applies and links to contradictory or superseding evidence. Update recommendations when new attempts arrive, preserving the historical knowledge state and uncertainty rather than rewriting failures into hindsight certainty.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ResearchLesson fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Lessons do not generalize beyond tested conditions or erase negative evidence.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
