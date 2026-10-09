---
description: Plan a document as titled sections, each with its passages, table and assets.
args:
  brief: string
  passages: Passage[]
  table_ids: string[]
  asset_ids: string[]
  note: Untrusted<string>
  problem?: string
returns: Outline
---
Plan the document that brief asks for.

1. Read brief and note; state in one phrase what the document is for.
2. Group the passages by what they say; each group becomes one section. Order the sections so that each builds on the
   one before.
3. Give each section a heading and one sentence of purpose.
4. Assign to each section the ids of the passages it draws on. Assign a table or an asset to the section it supports,
   using ids from table_ids and asset_ids; give each table and each asset to one section. Leave table_id null when no
   offered table fits.
5. Write the title.
6. problem, when given, says why an earlier outline was refused. Plan again so that every id is one that was offered.
