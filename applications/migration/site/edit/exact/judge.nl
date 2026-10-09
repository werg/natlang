---
description: Migration patch check (natural-language implementation). Whether the patches for a site are exact - old text unique in its file, something changed, and only what the site's usage calls for.
args:
  patches: Patch[]
  classified: Classified
  revision: string
returns: CheckedExactness
---
Check patches, written for classified.site at revision. Take the first failure and stop; the problem names the path
and says what to change in one sentence.

1. No patches: not exact.
2. For each patch: patch.path must equal classified.site.path. patch.old must be non-empty and differ from patch.new.
3. For each patch: found = repository.count(patch.path, patch.old, revision). Not exact unless found is 1.
4. Scope. For each patch, compare old and new. They may differ only where the old thing named in the site's usage
   (classified.usage.pattern) is replaced. A change to unrelated code in the snippet, a changed string literal that is
   data, or a lost line is not exact.

Return { exact: true, problem: "" }, or { exact: false, problem }.
