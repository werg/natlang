---
args: { baseline: GoalContract; ids: string[]; editions: EvidencePage[]; sectionResolution: AmendmentGuardrail }
returns: AmendmentTargetMap
---
Resolve each requested amendment against the signed executed edition and clause map. Match obsolete section references using the authenticated edition only. If one section number maps to multiple clauses, retain all candidates and block operation binding until evidence distinguishes them. Preserve amendment identity, exact operation and source locator; never treat a draft upload or document position as an executed target.For each amendment ID, bind target = resolveOneAmendmentTarget(baseline, amendmentId, editions, sectionResolution). Append a target only when the executed clause edition and the amendment operation are source-linked; retain ambiguous mappings in the unresolved target set.
