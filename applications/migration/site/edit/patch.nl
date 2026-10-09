---
description: Migration patch writing. The exact old and new snippets that change one classified site as the intent says.
args:
  intent: Intent
  classified: Classified
  revision: string
  problem?: string
returns: Patch[]
---
Write the patches for classified.site at revision. problem, when given, says why an earlier answer was rejected.
Compute in eval; take every old snippet from the site's text by slicing, as it is stored.

1. Occurrences. In classified.site.text, find each occurrence of intent.old that the usage pattern calls for changing
   (a declaration's name, an import specifier, a call's callee, a type name, a member name). Skip occurrences inside
   longer identifiers.
2. Old snippet. For each occurrence start with just the occurrence. While repository.count(site.path, snippet,
   revision) is not 1, widen the snippet by the next characters of site.text on each side (to whole tokens), up to the
   whole site text. A snippet that never reaches a count of 1 is skipped.
3. New snippet. The old snippet with that occurrence replaced by intent.new, or removed for a removal, adjusting the
   punctuation around it so the code stays valid.
4. When the occurrences' snippets overlap, merge them into one patch from the first start to the last end.
5. Each patch is { path: classified.site.path, old, new } with old different from new.

Return the patches in the order of their occurrences.
