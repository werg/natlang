---
description: Write one section of the document from its passages, with a claim for each fact it states.
args:
  brief: string
  heading: string
  purpose: string
  passages: Passage[]
  table: Table | null
  note: Untrusted<string>
  problem?: string
returns: SectionDraft
---
Write the section headed heading, whose purpose is purpose.

1. Read the passages for this section.
2. Write one claim for each fact the section states: a short claim text, the passage id (span_id), and a quote copied
   from that passage.
3. Write the section body from the claims, adapting the wording to brief and note.
4. When the evidence is uncertain or partial, say so in the body.
5. When table is given, refer to its values as the table presents them; the renderer prints the rows.
6. problem, when given, says why an earlier section was refused. Write the section again so that each span_id is the id
   of one of passages, each quote is copied from the passage with that id, and each number in the body is one that the
   passages or the table give.
