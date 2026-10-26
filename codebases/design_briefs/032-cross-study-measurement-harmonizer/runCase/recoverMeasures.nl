---
args: { goal: GoalContract, policy: PolicySelection, candidates: Harmonization[], evidence: EvidencePage[] }
returns: StageResultMeasureDefinition
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Read dictionaries, questionnaires, protocols, and usage notes together to recover what each variable measures. Distinguish similar labels from equivalent constructs and record timing, respondent, and collection-method differences that could change interpretation.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual MeasureDefinition fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: No mapping based only on column name or numeric range.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
