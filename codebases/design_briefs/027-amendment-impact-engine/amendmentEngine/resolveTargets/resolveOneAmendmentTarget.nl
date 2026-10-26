---
args: { baseline: GoalContract, amendmentId: string, editions: EvidencePage[], sectionResolution: AmendmentGuardrail }
returns: AmendmentTarget|null
---
Resolve this amendment only against the executed clause edition and source-linked old-section reference. Return its exact target clause IDs and operation when evidence supports one mapping. If two clauses remain plausible, return no target and retain the ambiguity in the caller's blocker set. Never use upload order or a later clause renumbering as target evidence.
