---
args:
  question: Text
  state: NotebookState
  files?: Dict<File>
returns: Text
---
Answer the question using only the cell result samples and their status. If the question names a supporting note, inspect that exact args/files leaf and distinguish it from cell evidence. Cite
cell IDs and revisions in prose. Distinguish SQL NULL from absent data and
empty results. If a required cell failed, stalled or became stale, state that
the answer is incomplete. Do not invent rows or interpret an output hash as
the result itself.
