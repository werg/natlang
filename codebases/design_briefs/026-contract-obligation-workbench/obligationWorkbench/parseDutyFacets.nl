---
args: { scope: GoalContract; contracts: ContractCase[]; evidence: EvidencePage[] }
returns: DutyFacetLedger
---
Read only the executed agreement editions bound by scope. For each distinct obligation preserve actor, modality, action, triggering event, deadline rule, exceptions, defined terms, amendment lineage and source locator. Do not combine clauses because they share a subject or actor. A missing exception, source location or party identity remains a draft facet with the exact gap. Return obligations and unresolved facet questions with their executed-edition refs.