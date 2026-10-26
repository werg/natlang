---
args: { goal: GoalContract, policy: PolicySelection, candidates: ReturnCase[], evidence: EvidencePage[] }
returns: StageResultConditionAssessment
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Reconcile customer descriptions, inspection notes, and supplied images or structured image annotations. Distinguish cosmetic damage, functional symptoms, and unverified allegations; preserve uncertainty and request evidence that changes routing rather than generic photo uploads.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ConditionAssessment fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: No unsupported diagnosis or condition grade; evidence provenance remains visible.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
