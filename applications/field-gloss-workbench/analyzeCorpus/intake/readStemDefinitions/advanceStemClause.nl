---
description: Judge one original stem definition and preserve its actual typed reading.
args:
  state: StemDefinitionFrontier
returns: StemDefinitionFrontier
---
1. If pendingClauseIds is empty, return state. Otherwise select its first exact original SourceClause; preserve all prior completed entries and the unchanged remaining source universe.
2. Directly await one scoped nl<StemClauseReading> judgment capturing this clause and its actual source-role entry. Decide whether the clause defines one or more stem surfaces, only supplies entity/context information, or leaves an unresolved reference. Return clauseId, kind, stems, reason, witnesses and unresolved details.
3. For a defined surface, preserve its exact original spelling and source-stated entity or event identity. A stem's own final characters do not become a grammatical marker merely because another hypothesis would segment a similar form.
4. Bind actualReading to the awaited result. Require its clauseId to equal this original unit and each decided stem to have a nonempty source surface and exact original quote; bind witness sourceId/path from the cited original clause. Do not replace its meaning with a case-specific JavaScript lookup.
5. Append a read, unresolved or failed StemClauseEntry, retaining actualReading when available and a concrete missing premise or failure otherwise. Append actualReading.unresolved and any unresolved/failed entry reason to state.unresolved. Preserve all earlier successful returns if this unit cannot be decided.
6. Bind next:StemDefinitionFrontier with this one original pending ID removed, the actual entry appended once and the updated unresolved list. Check unique original coverage and a strictly shorter pending list.
7. Finish immediately with eval({code:"return next;",finish:true}). readStemDefinitions alone owns the next original unit and its final projection.
