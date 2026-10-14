---
description: Read the original score archive and consume distinct source-role branches.
kind: directory-reducer
args:
  request: EditionRequest
returns: Archive
---
1. Read manifest.json through folder.readText and parse SourceManifest. Require its original caseId
   to match request.caseId. Its listed original documents are the immutable closed source universe.
2. Visit the actual nested directory groups, read each listed document and retain id/path/role/text.
   Extract visibly labeled SourceClause entries with nonempty original Witness quotations. Role labels
   organize the inventory; their factual meaning remains subject to source-role interpretation.
3. Validate unique original document and clause identities and exact text coverage. Preserve conflicting
   repeated IDs as unresolved rather than silently keeping the last file. Bind sourceId/path exactly.
4. Call classifyWitnesses(documents, clauses) for source roles and copying relationships. Pass that
   actual RoleIndex into readEditorialClaims and readRehearsal with the original clauses.
5. Call readConventions(clauses) and readPartbooks(clauses). These branches own supplied notation policy
   and diplomatic source readings; the archive reader invents neither pitches nor target assignments.
6. Return Archive with actual reader results, complete original documents/clauses and per-source
   coverage. Accumulate their unresolved findings. Later arbitration, performance and ensemble checking
   own the requested edition; inventory completion does not require their conclusions.
