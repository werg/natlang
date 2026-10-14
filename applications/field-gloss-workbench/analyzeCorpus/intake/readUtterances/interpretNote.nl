---
description: Judge what a note asserts and who owns its authority.
args:
  clauses: SourceClause[]
  sessions: SessionIndex
returns: NoteReading
---
1. Construct a typed scoped NL NoteReading literal from this original note and the actual session/consultant context. Capture no current grammar hypothesis.
2. Decide whether the note transcribes a form, proposes a gloss, records a consultant acceptance/rejection, adopts a revised gloss or merely supplies background context. Preserve null judgment when no actual acceptability verdict is stated.
3. Join any judgment to the original utterance and the actual targeted gloss slot. If the consultant rejects a translation but accepts the surface form, retain that scoped distinction in meaning rather than assigning rejected to the entire attestation.
4. Distinguish author identity from attributed consultant authority. A later copied note inherits neither agreement nor adoption merely because it repeats the consultant's name.
5. Preserve explicitly qualified judgments, including a form usable only under a particular topic prompt. An unqualified note cannot erase that condition from an earlier original statement.
6. Return exact original witnesses and unresolved target/authority premises. Parent binds provenance and consumes kind/judgment; it does not convert free explanatory wording into an exact label oracle.
7. If a compound source paragraph contains several independently targeted note units, preserve the selected unit's scope; its siblings remain separate inventory units for later calls.
