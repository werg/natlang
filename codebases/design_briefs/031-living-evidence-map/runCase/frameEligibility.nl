---
args: { goal: GoalContract, policy: PolicySelection, candidates: EvidenceReview[], evidence: EvidencePage[] }
returns: StageResultEligibilityCriterion
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Translate the research question into explicit screening criteria and distinguish mandatory eligibility from useful descriptive metadata. Define ambiguous boundaries with examples before full screening and preserve questions requiring expert methodological judgment.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual EligibilityCriterion fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Criteria are frozen by version and usable on unseen records.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
