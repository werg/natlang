---
description: "Read the supporting file a request names, if it names one."
args:
  request: string
  files?: Folder
returns: Untrusted<string>
---
Find the supporting file that request names, and give back its text.

1. Look in request for a file name or path (for example "notes.md" or "docs/brief.txt").
2. When request names no file, or files is not provided, return an empty string.
3. Otherwise read exactly that file from files and return its full text.
4. When the named file does not exist in files, return an empty string.
