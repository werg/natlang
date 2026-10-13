---
description: Apply replacement targets without erasing preserved historical clauses or unrelated duties.
args:
  clauses: SourceClause[]
  authority: AuthorityDecision[]
  exceptions: SourceRecord[]
returns: EditionSelection
---
1. Match each adopted competent replacement to its explicit target clause IDs and subject scope.
   Read whether it replaces an entire procedure or only the named prerequisite or effect.
2. For a supported replacement effective at time t, preserve the older governing clause before t
   and the replacement from t. Attach exact target/effect witnesses to each temporal interpretation.
3. Keep untouched clauses active across that boundary. A new queue exemption is a scoped exception
   and cannot delete custody, operator identity or safety acknowledgment merely because nearby text changes.
4. When target identity is ambiguous, retain both plausible readings as unresolved; do not select a
   version by matching filenames or force all scenario histories onto the latest published wording.
5. Join compatible adoption and effect intervals. If two same-authority replacements conflict without
   a source priority rule, expose that conflict for model/guard stages instead of choosing the longer text.
6. Return selectedClauses, replacedClauseIds, preservedClauseIds, authorityDecisions and unresolved.
   These retained historical intervals are consumed by actual scenario event-time guards.
