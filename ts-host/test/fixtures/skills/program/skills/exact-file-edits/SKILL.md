---
name: exact-file-edits
description: Applies exact find-and-replace edits to files without disturbing whitespace or unrelated files. Use when a change request lists literal text to replace.
natlang:
  exports:
    applyEdits: "(edits: { path: string, find: string, replace_with: string }[]) => number"
  requires:
    skills: [decision-finality]
  tests:
    - name: replaces only the first literal match
      input: [{ path: "a.txt", find: "draft", replace_with: "final" }]
      check: "the file a.txt contains final once and no other file changed"
  provenance:
    author: fixture
    families: [exact-multi-edit]
---

Use the `applyEdits` helper for literal edits. It matches text exactly, including leading tabs and blank lines,
and reports how many edits it applied. Do not normalise whitespace.
