---
args:
  checked: CheckReport
returns: string
---
Write a reading of the check result for a person who edits the source.

1. When checked.status is "checked", return "no errors".
2. Otherwise group the diagnostics in checked.detail by file and message.
3. Write one line per group, at most five lines: the file, what is wrong, and where.
