---
description: Interpret one event under its inherited speaker and truth scope.
args:
  record: SourceRecord
  parent: FrameDefinition | null
returns: FrameIndex
---
1. Read the record's own frame declaration. Establish its kind, owner and parent before separating
   events. A quoted voice inherits a discourse location, not the truth authority of that location.
2. Extract candidate event spans with scene IDs, participant names and exact source quotations.
   Preserve an assertion of seeing an object separately from actually observing its contents.
3. In eval declare local FrameJudgment with assertedMeanings: string[], observationMeanings:
   string[], reportedMeanings: string[], uncertainty: string[], witnesses: {sourceId: string; path: string; quote: string}[]. Capture the
   declared frame, inherited owner, scope wording and this record as a compact local context.
4. Construct nl.with<FrameJudgment> whose changing input is one event span. Its instructions are:
   interpret this span in the captured frame; separate narrator-established action, speaker claims,
   perceptual access and quoted reports; preserve unresolved agency and source-backed uncertainty.
5. Directly await that literal for each inventoried event span. Consume its returned collections as
   the EventView meanings. The literal's witnesses must refer to actual text from this record.
6. Put ambiguity explanations in unresolved. A literal cannot invent an omniscient narrator to
   upgrade a memory or treat an abandoned proposal as an event that happened in the selected take.
7. Return the frame definition and interpreted events with exact IDs and scene membership. An
   event without a visible identity receives a unique derived ID with its derivation stated in witnesses.
8. Keep the asserted, observed and reported arrays distinct in the returned typed contribution.
   Later authority and knowledge stages rely on this distinction; no merged paraphrase substitutes.
9. Append LocalDecision records to localDecisions only after each literal returns. Use its site path,
   actual event input IDs, consumed meaning distinction and exact returned witnesses. These records
   trace a returned contribution; an ID without a returned judgment must never be recorded.
