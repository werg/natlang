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
   string[], reportedMeanings: string[], uncertainty: string[], quotes: string[]. Capture the
   declared frame, inherited owner, scope wording and this record as a compact local context.
4. Construct nl.with<FrameJudgment> whose changing input is one event span. Its instructions are:
   interpret this span in the captured frame; separate narrator-established action, speaker claims,
   perceptual access and quoted reports; preserve unresolved agency and source-backed uncertainty.
5. Directly await that literal for each inventoried event span. The literal returns semantic meaning
   arrays and exact quotes from record.text. For each returned quote, this parent constructs a Witness
   with sourceId=record.id and path=record.path from the original SourceRecord, then calls
   storyEvidence.checkWitnesses({revision: record.sha256, records: [record]}, witnesses).
   Consume the returned meaning arrays only with the exact passing witnesses. A failed quote check
   becomes an unresolved evidence finding and cannot license a source-grounded asserted meaning.
6. Put ambiguity explanations in unresolved. A literal cannot invent an omniscient narrator to
   upgrade a memory or treat an abandoned proposal as an event that happened in the selected take.
7. Return the frame definition and interpreted events with exact IDs and scene membership. An
   event without a visible identity receives a unique derived ID with its derivation stated in witnesses.
8. Keep the asserted, observed and reported arrays distinct in the returned typed contribution.
   Later authority and knowledge stages rely on this distinction; no merged paraphrase substitutes.
9. Append LocalDecision records to localDecisions only after each literal returns. Use its site path,
   actual event input IDs, consumed meaning distinction and exact parent-bound witnesses. These records
   trace a returned contribution; an ID without a returned judgment must never be recorded.

10. The named call receives the complete original SourceRecord. Keep that record as the source
    authority throughout event interpretation; inherited frame context changes discourse ownership,
    never the record identity, path or hash. Bind EventView and LocalDecision witnesses at step 5.
