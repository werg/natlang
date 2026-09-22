---
args:
  brief: string
  passages: Passage[]
  table_ids: string[]
  asset_ids: string[]
  files?: Record<string, File>
returns: Outline
---
Plan a short document for the brief. If the brief names local editorial context, inspect only that args/files leaf. Choose a title, section headings and which
offered tables or assets belong to each section. Use only source passages and
identifiers offered here. Explain uncertainty in the prose rather than
inventing a fact. Return a structured outline.
