---
args:
  text: string
  catalog: Recipe[]
  files?: Folder
returns: string
---
Choose one recipe ID from catalog whose description fits the user's request.
If the request concerns repository contents, read only the named file from
files (for example README.md) before deciding. Use the exact ID. If none fits,
return "unsupported". Do not compose shell text or invent a recipe. Prefer a
specific build or media recipe over a generic status recipe when that is what
the user requested.
