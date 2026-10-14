---
description: Judge a scoped gloss replacement that preserves actual supported uncertainty.
args:
  request: RepairRequest
  originalNote: NoteReading
  source: CorpusSource
  graph: EvidenceGraph
  grammar: GrammarReport
  translations: TranslationReport
returns: PatchMeaning
---
1. Directly invoke a typed scoped NL PatchMeaning literal over this actual original note/slot, source correction/authorization rules and actual grammar/translation results. Exclude authored canonical patches or preferred gloss labels.
2. Determine whether the source permits editing this slot and whether an adopted correction governs it. A grammatically better gloss without authorized scope returns preserve or unresolved, not an executable edit.
3. Identify each surviving hypothesis contribution relevant to this slot and each actual rejection proof that defeats the copied gloss. Retain original target identities, not generalized rejection of the whole notebook.
4. When survivors agree on a corrected role/boundary, propose that common supported meaning. When they differ on a consequential prospective/temporal contribution, state the alternatives or uncertainty rather than selecting one by plausibility.
5. Preserve source-established transcription, lexical stems and untargeted clauses. An explicitly gloss-only correction cannot authorize changing accepted surface forms.
6. Return replacement text only for status replace, with actual supported/rejected hypothesis IDs and nonempty exact source quotes. Descriptive wording is free; named source identity and authorized slot remain exact.
7. Missing scope, authority or decisive meaning remains unresolved with no repair text. Parent applies this actual judgment through exact span plumbing and later challenge inspects entailment separately.
