---
args: { goal: GoalContract, policy: PolicySelection, candidates: Onboarding[], evidence: EvidencePage[] }
returns: StageResultOnboardingOutcome
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Convert broad goals into observable outcomes and confirm what onboarding does not include. Distinguish an executive aspiration from an agreed deliverable, and identify decisions requiring customer authority before technical work begins.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual OnboardingOutcome fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Each committed outcome has an owner and a verifiable acceptance condition.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
