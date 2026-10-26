---
args: { goal: GoalContract, policy: PolicySelection, candidates: FunctionalRequirement[], evidence: EvidencePage[] }
returns: StageResultSubstituteCandidate
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Search catalogs and supplier descriptions for equivalent functions, including alternative mechanisms where permitted. Explain how each candidate would serve the use, identify required adaptations, and avoid treating a familiar name or matching dimension as sufficient.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual SubstituteCandidate fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Candidate generation cannot relax essential constraints or invent supplier specifications.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
