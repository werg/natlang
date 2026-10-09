---
args:
  request: string
  names: string[]
returns: string
---
Choose the file a request concerns. names lists the editable files.

1. Read the request and list every file name or module name it mentions.
2. When one of names matches a mention, return that name.
3. Otherwise return the name whose role, read from its name and extension, fits the request best.

Return one entry of names, spelled exactly as listed.
