---
args: { goal: GoalContract, policy: PolicySelection, candidates: DisputeCase[], evidence: EvidencePage[] }
returns: StageResultDisputeEvent
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Align listings, messages, shipping records, and submissions without equating timestamps with event occurrence. Attribute disputed statements to their speakers, preserve edited listings, and mark contradictions rather than manufacturing a single settled narrative.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual DisputeEvent fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Every event distinguishes asserted facts from independently corroborated records.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
