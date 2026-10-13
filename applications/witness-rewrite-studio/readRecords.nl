---
description: Inventory source records by declared narrative ownership.
args:
  snapshot: SourceSnapshot
returns: RecordInventory
---
1. Initialize the seven RecordInventory arrays empty. Visit each SourceRecord exactly once.
2. Read its declared ID, folder ownership, opening scope statement and text. Treat nested act/take
   paths as organizational evidence; explicit selection and memory labels settle scope when present.
3. Put a director change request or preservation policy in instructions. Put stable world or prop
   definitions in canon. Put aliases and explicit distinctions in identityRecords.
4. Put the selected scene take in selectedScenes. Put a character's recollection in recollections,
   retaining the recollector and quoted frame boundaries in the original text for frame interpretation.
5. Put candidate repair recipes in proposals. A proposal can contain old text and replacement text;
   those quoted fragments remain proposed changes, never extra selected events.
6. A record can own distinct explicitly labeled sections. Preserve that record in every relevant
   inventory category only when its text declares those roles; later witnesses must cite the section.
7. Put a record with genuinely unresolved ownership in unclassified. Do not use filename similarity
   alone to equate a selected take with an abandoned draft or a memory of that take.
8. Return the inventory. Preserve original SourceRecord objects, IDs, paths and hash values.
   The finite source array is the traversal frontier; no network search or fabricated scene is needed.
