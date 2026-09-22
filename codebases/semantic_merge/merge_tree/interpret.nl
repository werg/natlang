---
description: Semantically reconcile edits to a parent-linked tree.
args:
  base: State
  updates: Update[]
  policy: string
returns: Draft
---
Treat stable node IDs as identity. Interpret subtree moves, independent renames, deletion
of a parent while another author edits a child, and claims that two headings are the same
topic. The returned nodes must keep one parent per node and no parent cycle. Explain any
unresolved move or deletion as an alternative with its source update IDs. Never hide a
subtree simply because one author deleted its former parent.
