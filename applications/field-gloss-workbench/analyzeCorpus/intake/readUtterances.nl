---
description: Recover exact surfaces and distinct field-note speech roles.
args:
  clauses: SourceClause[]
  sessions: SessionIndex
  rules: SourceClause[]
returns: UtteranceIndex
---
1. Inventory original utterance IDs and exact quoted language surfaces. Preserve the authored character sequence and spacing; do not pre-segment productive-looking endings during extraction.
2. Join each utterance to its actual session/prompt/situation/context by original references. Reused surface strings remain separate utterance events when elicited under different prompts.
3. Inventory each original note unit with its declared author and referenced utterance/slot. Call interpretNote(noteClauses, sessions, rules) for each unit; consume the actual role and judgment it returns.
4. A student's proposed translation is not a positive consultant judgment. An accepted utterance does not automatically accept every translation written beside it. Retain separate note IDs so correction scope can distinguish these cases.
5. A quoted example in an explanatory note is not a new attestation unless the source establishes an elicited or spontaneously used occurrence. Mark that distinction in actual note meaning and unresolved reasons.
6. Require each original language surface to match its exact source substring. Validate original joins and bind original witnesses; uncertain session linkage stays null instead of being guessed from directory order.
7. Return actual surface utterances and note readings, preserving all source-authorized duplicate events without collapsing different judgments of identical strings.
