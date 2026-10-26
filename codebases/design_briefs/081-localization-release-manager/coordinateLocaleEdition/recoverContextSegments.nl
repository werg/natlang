---
args: {"change": "LocaleSourceChange", "prior": "ContextSegment[]"}
returns: "ContextSegment[]"
---
For each source content item call separateContextUses(item, change.releases). Preserve markup, placeholder inventory, plural branches and neighboring product state from the actual source representation. Keep content identity separate from surface text. If a structural source representation is missing, preserve the text and a parsing question rather than inventing a message AST. Carry stable context keys through edits.
