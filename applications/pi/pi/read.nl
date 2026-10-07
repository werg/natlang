---
description: Read the contents of a file. Output is truncated to 2000 lines or 50KB; use offset/limit for large files.
args:
  path: string
  offset?: number
  limit?: number
returns: string
model: small
---
Read the file at path (files.read). When there is none, answer `Error: no file at <path>`.

Take its lines from offset (counted from 1; from the first line when not given), and only limit of them when limit
is given. Of those, keep at most 2000 lines and at most 51200 bytes, from the start. When that leaves some out, add a
blank line and `[Truncated: use offset=N to continue]`, N being the number of the first line not shown. Answer the
kept text exactly as it is in the file.
