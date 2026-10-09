---
description: Runbook lookup. Read the local runbook or log file an event names.
args:
  item: LogEvent
  files: Folder
returns: string
---
When item's message names a local runbook or log file (a relative path), read that file from files and return the part
of its text that bears on the event, at most 2000 characters. Return an empty string when the message names no file or
the file does not exist.
