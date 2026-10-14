---
description: Recover original corpus roles and finite candidate domains before grammatical interpretation.
args:
  snapshot: CorpusSnapshot
  request: LinguisticRequest
returns: CorpusSource
---
1. Check actual case identity and unique original document IDs/paths. Extract bracketed source clauses with their exact text and provenance; this extraction is exact I/O, not interpretation.
2. Construct a typed scoped NL inventory over the actual clauses. Identify corpus-rule, session, prompt, situation, utterance, field-note, correction, entity/stem and candidate-domain roles. One clause may have several roles; retain each link without inventing an utterance from a descriptive example.
3. Read source-visible entity/event inventories, stem readings and candidate-slot values from that actual inventory. A slot value is a hypothesis choice, not a governing answer. Preserve meanings and quoted defining clauses for every value.
4. Call readSessions(clauses), then readUtterances(clauses, actualSessions). Bind the actual returned situations, surface utterances and note-role judgments; never infer consultant agreement from the scene description alone.
5. Call readCorrections(clauses, actualUtterances.notes, actualSessions). Preserve correction target, adopted version and explicit exclusions separately from the student's copied gloss.
6. Verify all returned original IDs belong to the original clause/document domain and all nonempty semantic decisions cite exact original quotes. Bind sourceId/path from those actual clauses. Unknown attribution or target scope remains typed unresolved rather than an empty success inventory.
7. Return CorpusSource with the actual source rules, finite candidates, original stems and returned indices. Request targets/repair scope do not enter the candidate selection judgment as preferred conclusions.
