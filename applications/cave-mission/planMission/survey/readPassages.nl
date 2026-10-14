---
description: Consume per-connection NL meanings using the actual returned frame and landmark inventory.
args:
  clauses: SourceClause[]
  frames: FrameSurvey
returns: PassageAssertion[]
---
1. Capture original clauses and actual FrameSurvey. Construct a scoped NL inventory literal returning
   SurveyUnitInventory and directly await it on these originals. It identifies each physical-connection
   reading unit with source primary/supporting clause IDs and a focus, including negative claims.
   These nonlandmark units preserve originalLocalName:null.
2. Its inventory joins descriptions only when source scope establishes the same connection. A source
   clause with several endpoint pairs yields several units. Reverse prohibition is supporting evidence
   for the same present one-way route; explicit absence remains a distinct physical claim.
3. Require nonempty primaryClauseIds, passage kinds, existing original IDs and unique unit identities.
   An invalid inventory reports its unit and expected passage/source scope for repair. For each unit gather
   its original clauses and call interpretPassage(unit, primaryClauses, supportingClauses, frames).
   The typed FrameSurvey supplies actual endpoint/frame keys; no second landmark table is created.
4. Consume the actual nullable PassageItemMeaning. Copy physicalExistence, allowed direction,
   endpoints, frame, kind, cost and condition only from its returned assertion. Missing decisive
   source meaning fails this contribution with the original unit and missing premise for repair; this
   array result must not silently omit an unresolved selected connection.
5. For each established assertion require nonempty exact clause-keyed evidence. Bind sourceId/path
   from its actual original witness after substring validation. Check endpoint keys belong to the
   supplied framed inventory; a failed source or endpoint check fails this contribution for repair.
6. Retain absent/unknown physical connection meanings and their negative witnesses. Preserve reverse
   prohibition and full-carrier restrictions on present rows. Cost may remain null when not established;
   no action cost is invented for a nonexistent physical route.
7. Merge exact duplicates only. Conflicting scope/direction/cost meanings remain separate witnessed
   assertions for hypothesis-specific reconciliation. Return these actual typed rows, preserving the
   distinction between physical absence, forbidden direction and unmet carried-state condition.
