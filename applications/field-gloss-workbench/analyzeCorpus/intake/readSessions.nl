---
description: Separate elicitation situations from utterance meaning and actual consultant access.
args:
  clauses: SourceClause[]
returns: SessionIndex
---
1. Inventory actual session IDs, original consultant identities, prompt IDs and source event order. Keep a copied note's writing time separate from the elicitation it describes.
2. For each actual session use a scoped NL SessionReading judgment over its original clauses. Determine which consultant saw the depicted situation and what contrast set the prompt established, without interpreting suffixes or verb roles.
3. Collect every source situation mentioned in that session and call readSituation(sessionReading, situationClauses). Pass the actual session's context clauses, including exclusions such as an unasked future horizon.
4. Consume actual SituationReading results. If a scene contains several events but the prompt names only one, retain the scene inventory and the named elicitation event separately; do not silently merge their participants.
5. A later correction session may share the same surface string while changing the contrast set. Preserve separate original session/situation identities and actual order so later evidence cannot inherit the earlier prompt by spelling alone.
6. Check original prompt/situation/context clause IDs and exact quoted support. Preserve unknown consultant access or session order as unresolved; a familiar name is not evidence of attendance.
7. Return actual sessions and situations in source-established order, retaining unorderable rows with explicit unresolved reasons rather than dropping them.
