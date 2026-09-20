---
args:
  text: Text
  catalog: Recipe[]
returns: Text
---
Choose one recipe ID from catalog whose description fits the user's request.
Use the exact ID. If none fits, return "unsupported". Do not compose shell
text or invent a recipe. Prefer a specific build or media recipe over a generic
status recipe when that is what the user requested.
