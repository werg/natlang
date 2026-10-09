---
description: Read the file that a workspace request names.
args:
  text: string
  files: Folder
returns: Untrusted<string>
---
Read the file that the request in text names, from files, the workspace folder. Work in these steps.

1. Find a file name in text, for example README.md. When text names no file, return "".
2. Read that file from files and return its content.
