---
description: Interpret original source roles before joining chronology and participation.
args:
  snapshot: DialogueSnapshot
  request: DialogueRequest
returns: DialogueSource
---
1. Read original documents as immutable sourceId/path/text triples. Extract bracketed clause boundaries
   exactly; every SourceClause preserves its original ID, document path and complete clause text.
2. Construct a scoped NL inventory literal capturing these clauses and the closed request. Directly call
   it with the original clauses. Ask for turns, rule clauses, proposition vocabulary and physical facts;
   its instructions distinguish authenticated speech, receipt metadata, private note and verified event.
3. For each returned turn require its original source clause ID and nonempty exact supporting quote.
   Bind sourceId/path from the original clause. Content is the exact spoken/note substring, excluding
   editorial metadata and its enclosing transcription delimiters. Internal quotation delimiters remain.
   Speaker, audience, temporal frame and visibility are NL
   interpretations; original paths and source IDs are exact plumbing. Preserve nullable attribution.
4. Require the returned proposition keys to be those actually declared by the visible output vocabulary.
   Copy their meanings from that returned reading, with exact clause support. A source ID that resembles
   a date is opaque; only the actual timestamp supplies chronology.
5. For each physical fact retain the returned proposition/value, occurrence time and completeness horizon.
   A complete negative event log may establish a bounded absence. An incomplete silence cannot do so.
   Store these facts separately from participant availability; no actor inherits them at intake.
6. Call readChronology(clauses, turns), readParticipation(clauses, turns) and readReceipts(clauses, turns,
   physicalFacts). Pass the original arrays and consume each actual child result independently.
7. Check original turn identity is unique and every actor reference belongs to actual participation or
   remains unresolved. A private note names its owner, without creating an audience of other participants.
8. Return DialogueSource with caseId, original clauses, actual inventory contributions and the three child
   registries. Concatenate unresolved contributions; do not omit a problematic document to make it closed.
