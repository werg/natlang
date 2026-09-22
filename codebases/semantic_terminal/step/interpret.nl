---
args:
  text: Text
  catalog: Recipe[]
  files?: Dict<File>
returns: Text
---
Choose one recipe ID from catalog whose description fits the user's request. If the request concerns repository contents, inspect only the named args/files leaf (for example args/files/README.md/text) before deciding.
Use the exact ID. If none fits, return "unsupported". Do not compose shell
text or invent a recipe. Prefer a specific build or media recipe over a generic
status recipe when that is what the user requested.
