---
description: Perform each actual edition through finite occurrences and stateful voice readings.
args:
  editions: EditionSet
  protocol: NotationProtocol
returns: PerformedEdition[]
---
1. Visit each actual complete EditionHypothesis. Preserve its chosen written options and the actual
   protocol occurrence order; no canonical event table is available to this stage.
2. Initialize PerformanceFrontier with nextOccurrenceIndex=0, the fixed occurrences, this hypothesis,
   actual protocol, empty performed events and unresolved source findings.
3. Call advanceOccurrence.iterateOn with measure occurrences.length-nextOccurrenceIndex. The changing
   state selects its occurrence; other arguments remain static context. Consume each returned frontier.
4. Require all actual occurrence IDs processed and every chosen event in the matching section present
   once per occurrence/voice. A repeated section has fresh occurrence identity but the same source option.
5. Return one PerformedEdition per actual hypothesis with its returned events and uncertainties.
   This stage does not eliminate an edition because another one sounds more conventional.
