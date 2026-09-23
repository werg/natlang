---
args:
  text: string
  catalog: Recipe[]
  files?: Record<string, File>
returns: string
---
Choose one recipe ID from catalog whose description fits the user's request. If the request concerns repository contents, inspect only the named files leaf (for example files/README.md/text) before deciding.
Use the exact ID. If none fits, return "unsupported". Do not compose shell
text or invent a recipe. Prefer a specific build or media recipe over a generic
status recipe when that is what the user requested.
