---
description: Read the editorial context file that a brief names.
args:
  brief: string
  files: Folder
returns: string
---
Return the text of the editorial context file that brief names.

1. Find the file name in the brief: a word that ends in a file extension.
2. When there is one, read that file from files and return its text exactly.
3. When the brief names no file, return the empty string.
