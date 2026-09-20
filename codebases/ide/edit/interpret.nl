---
args:
  request: Text
  snapshot: EditorSnapshot
returns: EditPatch
---
Translate the request into one exact text edit of a listed function. Give
zero-based start and end offsets into the current source string and carry its
revision. Preserve unrelated text and inspect the whole relevant function.
If the request needs several edits, apply and check them one at a time so each
offset refers to the latest revision.
