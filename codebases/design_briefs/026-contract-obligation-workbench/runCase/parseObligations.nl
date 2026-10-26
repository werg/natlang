---
args: { goal: GoalContract, policy: PolicySelection, candidates: ContractCase[], evidence: EvidencePage[] }
returns: StageResultContractObligation
---
1. Freeze the supplied candidate IDs and evidence partitions; account for every candidate as accepted, unresolved, rejected, or explicitly excluded with a reason.
2. For every frozen candidate, bind prepared = prepare(candidate, goal, policy, evidence). Retain its bounded source spans and missing-field list.
3. Bind decisions = decide(prepared, goal, policy). Apply this domain algorithm: Extract atomic obligations with actor, beneficiary, trigger, deadline, and exceptions. Separate descriptions and aspirations from operative provisions, retain cross-references, and preserve unknown roles rather than assigning responsibility from nearby names.
4. Bind draft = construct(candidate, decisions, evidence). Assemble the actual ContractObligation fields; keep conflicting values distinct and leave unknown fields absent.
5. Bind auditResult = audit(draft, candidate, evidence, policy). The audit returns a finite gate verdict and repair target.
6. Accept a candidate only when auditResult.verdict is "pass" and this gate holds: Each obligation retains all material conditions and supporting clause spans.. Preserve unresolved candidates, questions, rejected alternatives, consumed references, and complete coverage.
