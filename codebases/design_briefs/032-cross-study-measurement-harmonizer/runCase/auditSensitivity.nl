---
args: { goal: GoalContract, policy: PolicySelection, candidates: HarmonizedSlice[], evidence: EvidencePage[] }
returns: StageResultHarmonizationAudit
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Compare reasonable alternative mappings and cohort restrictions to reveal dependence on harmonization choices. Produce an analyst-facing report distinguishing stable findings from those driven by assumptions and identify measurements that must remain separate.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual HarmonizationAudit fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Sensitivity alternatives are justified and do not selectively hide inconvenient results.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
