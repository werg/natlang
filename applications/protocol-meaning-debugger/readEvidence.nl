---
description: Interpret nested protocol documents without merging same-named revisions.
args:
  documents: ProtocolRecord[]
returns: EvidenceIndex
---
1. Visit each finite ProtocolRecord once. Read explicit source ID, owner, revision, scope and clause IDs.
   A nested archive path and an execution binding have different evidential roles.
2. Extract DefinitionClauses for question, population, anchor, window, metric, alias, provenance, scope
   and probe authorization. Keep the source's named owner and exact quotation for each interpretation.
3. A clause describing a historical definition is evidence about that definition, not an active runtime
   policy. Preserve its wording and ownership so lineage can bind or exclude it explicitly.
4. Treat implementation manifests as claims about selected artifacts. Keep their referenced clause IDs
   and revision/hash distinctions; do not infer selection from the shortest filename.
5. Distinguish raw observation coding from displayed metric labels. The word pass in a table has its
   stated operational meaning, while a report labeled success can still count failures underneath.
6. Keep unsupported or conflicting owner/role readings in unresolved. A numeric literal in a note
   does not establish a universal window or population cutoff without the clause defining its scope.
7. Return documents, definitions and unresolved. Derive no new trial readings or repair edits here.
8. The source records themselves remain immutable. Exact hashes check bytes; semantic clause extraction
   requires this NL interpretation and later source-grounded review.
