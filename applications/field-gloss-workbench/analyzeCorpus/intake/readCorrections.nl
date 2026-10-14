---
description: Retain targeted correction and actual note-version adoption separately.
args:
  clauses: SourceClause[]
  notes: NoteReading[]
  sessions: SessionIndex
returns: CorrectionIndex
---
1. Inventory original explicit correction units and their named original notes/slots. Keep source order and distinguish a consultant correction from the researcher's later note adoption.
2. For each unit call judgeCorrectionScope(unitClauses, notes, sessions). Consume the actual CorrectionMeaning including replacement IDs, exclusions and adopted=true/false/null.
3. Validate every returned original note/slot ID against the actual note inventory. An ambiguous target does not authorize replacing all notes about the same suffix.
4. Keep an accepted surface attestation intact when only its proposed translation is corrected. A scope=gloss correction cannot become a surface rejection by propagation.
5. If two actual adopted corrections compete for one slot, retain their authority/order relation and unresolved conflict; do not choose whichever makes the grammar frontier smaller.
6. Bind exact original witness identity and preserve original correction units even when unadopted. Effective-note resolution receives these actual decisions, not an already overwritten note table.
7. Return the complete actual correction index with missing replacement/access/adoption premises explicit.
