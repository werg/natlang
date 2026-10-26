---
args: { goal: GoalContract, policy: PolicySelection, candidates: InquiryFrame[], evidence: EvidencePage[] }
returns: StageResultSamplingPlan
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Define strata from the inquiry, not from convenient examples, then request exact selection within each stratum. Include negative and atypical cases, record nonresponse, and freeze a sampling manifest before coding the main set.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual SamplingPlan fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Manifest preserves inclusion probabilities where applicable and explains purposive selection elsewhere.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
