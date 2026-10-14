---
description: Build actual prompt-response and note-lineage constraints.
args:
  source: CorpusSource
returns: EvidenceGraph
---
1. Bind effective to resolveNoteVersion(source.utteranceIndex.notes, source.corrections, source.rules). Preserve actual original-to-effective entries and disputed slots.
2. For each original utterance, including a declared held-out target with no consultant judgment, call alignElicitation(utterance, source, effective). Pass its actual original prompt, session and situation, not a combined generic scene.
3. Consume actual returned assertedFields/excludedFields and judgmentNoteIds. Construct one EvidenceUnit with the actual utterance, actual situation, actual alignment and the selected effective notes.
4. An utterance with only a transcription still has source identity but may have no usable semantic constraint. Retain scope.status unresolved/unusable and its reasons; do not infer a negative judgment from absence.
5. Add exact-ID graph links for utterance→prompt/situation, note→utterance and correction→target/replacement. Their relationship meaning comes from actual NL stages; set insertion is exact mechanism.
6. Check that every original elicitation unit is represented and all note IDs resolve to the returned effective-note entries. Preserve unknown authority as an evidence blocker rather than silently selecting the uncorrected proposal.
7. Return actual units, effective versions, links and unresolved premises. Grammar inference consumes these explicit applicability scopes before testing candidate predictions.
