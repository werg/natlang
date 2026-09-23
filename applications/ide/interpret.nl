---
args:
  request: string
  snapshot: EditorSnapshot
returns: EditPatch
---
Translate the request into one exact text edit of a listed file. Give
zero-based start and end offsets into the file's current source and carry the
snapshot revision as expected_revision. Preserve unrelated text and read the
whole relevant function before choosing offsets.
