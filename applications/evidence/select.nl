---
args:
  question: string
  found: SearchResult
returns: string[]
---
Choose IDs of the offered hits that can answer the question, including any
conflicting or qualifying evidence. Return distinct exact IDs. If the search
was truncated or no hit is relevant, return what there is; compose reports the
uncertainty.
