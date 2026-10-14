---
description: Distinguish original, derivative, editorial and observational source roles.
args:
  documents: SourceDocument[]
  clauses: SourceClause[]
returns: RoleIndex
---
1. Bind originalDocuments to the immutable supplied documents. Inventory every original source ID and
   the provenance clauses actually mentioning its production, issuer, exemplar or observational method.
   Include control/convention documents so final source-role coverage is complete.
2. Initialize roles and unresolved empty. For each actual document bind relatedHistory to its original
   provenance clauses and any source-established chain explaining an indirect exemplar reference.
3. Declare WitnessRole. Create and directly await one scoped role judgment on that actual source:
   const judgeRole = nl<WitnessRole>`1. Read document and relatedHistory. Determine whether this source supplies original notation, copied notation/report, an editorial act, firsthand rehearsal observation, a governing convention or a control contract.
   2. Establish any named exemplar and the precise copied component. A notation copy can predate later editorial slips; copying a rehearsal report reproduces testimony rather than adding a new observation. Preserve a source that independently reports one component while copying another as a scoped account.
   3. Assign independent=true only for established independent production in the described scope, false for established derivation, null for missing or contradictory lineage. Keep the original sourceId; copyOf is the actual source-named exemplar or null when none is established.
   4. Return role, copyOf, explicit scope and nonempty exact original witnesses. Authentication/chronology is reported as source context, while the governing correction policy will decide authority later.`;
   const role = await judgeRole({document,relatedHistory});
4. Require role.sourceId to equal document.id. For nonnull copyOf require an original document ID;
   a missing exemplar remains an unresolved source link rather than a newly invented source.
5. Validate every role witness against the actual original document/history text and bind identity
   from that matched source. Append the actual role once. A rejected invocation retains the original
   document ID and concrete failure, preserving every earlier successful source-role judgment.
6. If the source history establishes distinct independence scopes that one role cannot faithfully
   summarize, retain that precise unresolved scope. Family construction must not count the whole source
   as an independent vote while this decisive relation is missing.
7. Compare returned source IDs to the original document universe exactly once and retain coverage gaps.
   Return roles and unresolved. The next dependency reducer connects these actual scoped claims;
   it does not reclassify the sources from filenames or document counts.
