---
description: Audit actual produced state without replacing it with a rebuilt ledger.
args:
  source: DialogueSource
  scoped: ScopedConversation
  episodePlan: EpisodePlan
  evolution: PerspectiveEvolution
  answers: QueryAnswers
  request: DialogueRequest
returns: PerspectiveReport
---
1. Start findings with the actual scoped.scopeFindings. Call checkNoKnowledgeLeak(source, scoped, episodePlan, evolution). Consume actual findings about
   availability, private scope, endorsement and chronology; retain failures rather than masking them.
2. For each actual QueryAnswer call replayAnswer(answer, source, scoped, episodePlan, evolution).
   Consume its actual audit result. This is independent semantic path inspection, not app re-execution.
3. Classify actual uncertainty before treating it as a defect. Source-explicit ambiguous pronoun/tone
   or a known still-open event window is valid uncertainty when its alternatives/pending scope were
   preserved. An absent child contribution, invented attribution or lost recipient scope is a defect.
   Return a witnessed LedgerFinding for that distinction; explain original missing premises rather than
   silently turning known ambiguity into a preferred answer.
4. Check original query coverage and order exactly. Validate each original turn/receipt and active act
   contribution has its appropriate actual source/episode lineage or an explicit unresolved reason.
5. Check generated proof/act/belief/revision IDs have valid references and each witness preserves exact
   original clause identity/path/nonempty substring. Exact quotation existence does not certify entailment.
6. Keep actual evolution.history and deltas unchanged. Audit observations cannot overwrite a sender,
   participant belief, condition activation or intended quotation interpretation with a preferred answer.
7. Return PerspectiveReport containing actual source/scoped/episode/evolution/answer contributions and
   all findings/audits. Concatenate unresolved source, stage and failed-audit obligations with original IDs.
8. Distinguish source/canonical certificate validity from actual live NL execution and literal consumption.
   This pure report performs no dispatch, correction, key release or other irreversible external action.
