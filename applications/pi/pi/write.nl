---
description: Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories. intent says what the new content is for.
args:
  path: string
  content: string
  intent: string
returns: string
model: small
---
Write content to path exactly as given (files.write creates missing directories), replacing what was there, and
answer `Wrote N bytes to <path>`, N being content's length in UTF-8 bytes. When path held a file before, end with
` (replaced the previous N lines)`.
