---
description: Propose only source-authorized slot repairs from actual analysis results.
args:
  request: RepairRequest
  source: CorpusSource
  graph: EvidenceGraph
  grammar: GrammarReport
  translations: TranslationReport
returns: PatchProposal
---
1. Resolve the actual original requested note and slot; read original patch authorization clauses and allowed path. Keep its exact original span/text rather than treating a whole document as replaceable.
2. Join this slot to actual effective-note correction authority, surviving hypotheses, rejected-prefix evidence and target ambiguities. These actual objects inform the repair; no ideal gloss is supplied.
3. Call judgePatch(request, originalNote, source, graph, grammar, translations). Bind actualMeaning, retaining supported/rejected hypothesis IDs and its actual uncertainty policy.
4. For status replace, locate only the source-authorized exact original slot span and construct one GlossPatch with original before text and actual returned replacement. Preserve clause markers and every untargeted transcription/role/context slot.
5. For preserve return no edit and retain the actual source reason. For unresolved return no executable replacement while retaining missing authority or undecided grammar distinctions.
6. Check allowed original path, exact start/end/before text and unchangedNoteIds. Exact patch validation cannot decide which gloss to choose; the actual child judgment owns that policy.
7. Return actual request, patches, unchanged original note IDs and unresolved premises. Final audit independently challenges the semantic repair and validates source preservation.
