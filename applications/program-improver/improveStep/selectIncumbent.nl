---
args:
  candidates: IncumbentChoice[]
  incumbent: string
  objective: string
returns: string
---
Choose which candidate becomes the incumbent. Every candidate already has the highest validation quality, and the objective's measure (source size for `source-size`, model requests for `model-calls`) already ties between them.

1. When one candidate has the id `incumbent`, return that id.
2. Otherwise compare the candidates by sourceBytes and modelCalls, smaller first, and return the id of the smallest.
3. When several remain, return the smallest id.
