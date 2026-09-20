---
args:
  brief: Text
  passages: Passage[]
  table_ids: Text[]
  asset_ids: Text[]
returns: Outline
---
Plan a short document for the brief. Choose a title, section headings and which
offered tables or assets belong to each section. Use only source passages and
identifiers offered here. Explain uncertainty in the prose rather than
inventing a fact. Return a structured outline.
