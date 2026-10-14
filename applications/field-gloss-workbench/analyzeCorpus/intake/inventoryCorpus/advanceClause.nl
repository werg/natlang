---
description: Judge one original clause scope and preserve the actual typed inventory result.
args:
  state: CorpusReadFrontier
returns: CorpusReadFrontier
---
1. If pendingClauseIds is empty, return state. Otherwise remove its first original ID and bind clause to that exact original SourceClause. Preserve all earlier entries and remaining original IDs.
2. Directly invoke a typed scoped nl<ClauseRoleReading> literal capturing this actual clause and source identity only. Inventory the clause's authored function: governing rule, session/prompt/situation, actual utterance/note/correction, entity/stem/domain declaration, context or patch authorization.
3. Return every actually established role when a clause has several. A descriptive example inside a rule remains its rule/context role unless the original source establishes an attestation; do not invent a new elicitation event.
4. The literal owns role classification only. It returns no segmentation, grammatical relation, translated meaning, candidate survivor or replacement gloss. Unknown source role remains an actual unresolved premise rather than a guessed linguistic conclusion.
5. Bind actualReading to the actual awaited result. Validate clauseId equality and nonempty exact original quotes for decided roles; parent binds sourceId/path from the original clause. Retain that actual reading unchanged in ClauseInventoryEntry.
6. For semantic uncertainty retain status unresolved and its actual reason. For malformed/transport failure retain status failed, actual failure and original clause, with actualReading null; preserve all earlier successes.
7. Bind next:CorpusReadFrontier to the shorter original pending list and actual appended entry/reasons. Finish immediately with eval({code:"return next;",finish:true}); parent iteration owns the next clause.
