---
args:
  question: Text
  found: SearchResult
returns: Text[]
---
Choose IDs of the offered hits that can answer the question, including any
conflicting or qualifying evidence. Return distinct exact IDs. If the search
was truncated or no hit is relevant, preserve that uncertainty for compose.
