---
args:
  brief: Text
  outline: Outline
  passages: Passage[]
  collection_revision: Text
  table_ids: Text[]
  asset_ids: Text[]
returns: Document
---
Compose one portable document from the outline. For each factual claim write a
short claim text, the exact span_id and revision of its passage, and a literal
quote copied from that passage. The quoted words must occur in the passage.
Use table_id only for an offered table. Put offered asset IDs in assets if used.
Carry collection_revision exactly into evidence_revision. Body text can be
adapted to the brief, but never calculate or invent exact table values.
