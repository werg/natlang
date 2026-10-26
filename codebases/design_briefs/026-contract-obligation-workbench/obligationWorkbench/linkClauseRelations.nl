---
args: { duties: DutyFacetLedger; evidence: EvidencePage[] }
returns: ClauseRelationGraph
---
Build directed links only where the agreement text or executed amendment establishes dependency, exception, supersession or conflict. Preserve relation scope and applicable edition. If two readings create different trigger or remedy behavior, keep both paths and state the discriminator evidence needed; do not choose the more severe interpretation by default. Return edges, isolated obligations and unresolved precedence.