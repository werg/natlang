---
args: { duties: DutyFacetLedger; evidence: EvidencePage[]; disputed: ObligationGuardrail }
returns: ClauseRelationGraph
---
Build directed links only where the agreement text or executed amendment establishes dependency, exception, supersession or conflict. Preserve relation scope and applicable edition. Carry every disputed clause ID and competing reading from `disputed` into unresolved precedence and the edge alternatives; no downstream task may rely on one unresolved interpretation. If two readings create different trigger or remedy behavior, keep both paths and state the discriminator evidence needed; do not choose the more severe interpretation by default. Return edges, isolated obligations and unresolved precedence.
For each plausible duty pair, bind relation = classifyOneDutyRelation(leftDuty, rightDuty, evidence). Add only the child’s source-supported directed relation; retain ambiguous precedence and conflicting readings in the graph’s unresolved paths. Do not create symmetric edges unless the source supports both directions.
