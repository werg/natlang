---
args: { concepts: ReuseConceptOption[], costs: CostScope[] }
returns: ID[]
---
For each concept compare its required enablingWorks to included cost scopes. Return missing work IDs; do not infer their price or treat a subtotal with excluded enabling works as comparable.
