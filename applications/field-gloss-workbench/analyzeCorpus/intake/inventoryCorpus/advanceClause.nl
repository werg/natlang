---
description: Judge one original clause scope and preserve the actual typed inventory result.
args:
  state: CorpusReadFrontier
returns: CorpusReadFrontier
---
1. If pendingClauseIds is empty, return state. Otherwise select its first original ID and bind clause to that exact original SourceClause. Keep that ID pending until its typed reading is valid; preserve all earlier entries.
2. Bind actualReading:ClauseRoleReading from a direct awaited scoped literal over this original clause, using the idiom `const actualReading: ClauseRoleReading = await nl<ClauseRoleReading>`...`(clause);` with the source identity in its instructions. Its roles are ClauseRole[] and exact source quotes are a separate Witness[]; retain the literal's actual result. Inventory only the clause's authored function: governing rule, session/prompt/situation, actual utterance/note/correction, entity/stem/domain declaration, context or patch authorization.
3. Return every actually established role when a clause has several. A descriptive example inside a rule remains its rule/context role unless the original source establishes an attestation; do not invent a new elicitation event.
4. The literal owns role classification only. It returns no segmentation, grammatical relation, translated meaning, candidate survivor or replacement gloss. Unknown source role remains an actual unresolved premise rather than a guessed linguistic conclusion.
5. Validate actualReading before consuming this pending ID: its clauseId equals the original ID; every role is one declared ClauseRole string; decided roles have a nonempty exact quote in a separate Witness whose clauseId matches and whose sourceId/path are bound to this original clause. Retain the actual semantic reading unchanged in ClauseInventoryEntry.
6. A malformed return or transport failure keeps this same original clause pending and all earlier successful entries intact. Give one concise exact recovery error and obtain a valid typed reading for this same item before completing the step. Source uncertainty is different: retain status unresolved, the actual reading and its stated missing premise.
7. Only after the typed return passes those checks, bind next:CorpusReadFrontier to the shorter original pending list and one appended actual entry/reasons. Finish immediately with eval({code:"return next;",finish:true}); parent iteration alone owns the following original clause.
