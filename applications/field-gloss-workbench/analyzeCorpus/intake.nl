---
description: Recover original corpus roles and finite candidate domains before grammatical interpretation.
args:
  snapshot: CorpusSnapshot
  request: LinguisticRequest
returns: CorpusSource
---
1. Check actual case identity and unique original document IDs/paths. Extract bracketed source clauses with their exact text and provenance; this extraction is exact I/O, not interpretation.
2. Call inventoryCorpus(clauses) and bind actualInventory. Its one-clause typed frontier retains actual role judgments, unknowns and failures. Consume returned source roles only; inventory discovers original scope and does not supply segmentation, marker meaning, candidate winners or translations. One clause may have several roles; retain each actual link.
3. Use actualInventory entries to select actualRules from original governing rule/authority clauses. Call readCandidateDomains(clauses, actualInventory, actualRules) and bind actualDomains. Call readStemDefinitions(clauses, actualInventory) and bind actualDefinitions. Consume actualDomains.slots as actualSlots and actualDefinitions.stems as actualStems; retain their original coverage entries and unresolved reasons. Each named reader obtains source meaning from an awaited one-clause scoped NL decision rather than an inventory label, a large inline array or a later surviving analysis. A slot value is a hypothesis choice, not a governing answer.
4. Bind actualSessions to readSessions(clauses), then actualUtterances to readUtterances(clauses, actualSessions, actualRules). Retain the actual returned situations, surface utterances and note-role judgments; never infer consultant agreement from the scene description alone.
5. Bind actualCorrections to readCorrections(clauses, actualUtterances.notes, actualSessions, actualRules). Preserve correction target, adopted version and explicit exclusions separately from the student's copied gloss.
6. Verify all returned original IDs belong to the original clause/document domain and all nonempty semantic decisions cite exact original quotes. Bind sourceId/path from those actual clauses. Unknown attribution or target scope remains typed unresolved rather than an empty success inventory.
7. Return CorpusSource retaining actualInventory, the actual source rules, actualSlots, actualStems and returned indices. Carry both readers' unresolved reasons into CorpusSource.unresolved. Request targets/repair scope do not enter the candidate selection judgment as preferred conclusions.
