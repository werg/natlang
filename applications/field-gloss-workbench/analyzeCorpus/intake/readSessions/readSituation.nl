---
description: Interpret depicted event roles and discourse context without supplying a grammar.
args:
  session: SessionReading
  clauses: SourceClause[]
returns: SituationReading
---
1. Capture this actual session and selected original situation clauses in a typed scoped NL SituationReading literal. Do not capture grammar candidates, final translations or gloss repairs.
2. Identify the named entities physically depicted and the actual agent/patient/event relation if the source establishes one. Preserve a negative event statement as eventOccurs=false, and missing occurrence evidence as null.
3. Identify source-established discourse topic and contrast alternatives independently of event agent/patient. A topic can be a nonagent, and a described true fact need not be part of the elicitation assertion.
4. Distinguish an explicit event-time description from an unstated temporal horizon. If nobody asked or established whether the event will later happen, preserve that absence rather than converting it to a future denial.
5. If several situation readings are possible, retain the undecided fields and supporting original clauses. Do not force one role by mapping the first noun in the utterance to the first pictured participant.
6. Return actual source witnesses for every decided relation. Parent binds original clause identity and checks exact quotation; it does not rewrite a semantically uncertain scene into a complete one.
7. The resulting scene is contextual evidence. Later response-scope judgment decides which of its fields a particular consultant response licenses for grammar comparison.
