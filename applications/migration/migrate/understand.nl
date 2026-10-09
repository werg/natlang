---
description: Migration planning, intent. Restate a migration request as the old thing, the new thing, the strings to search for and the behaviors that must stay, from the request, the file manifest and the check ids.
args:
  request: string
  snapshot: RepoSnapshot
  checks: string[]
  seeds: string[]
returns: Intent
---
Restate request for the later stages. snapshot.files are the files the migration may change (path, lines); checks are
the ids of the checks that will run on the result; seeds are strings the requester wants searched.

1. old: the identifier, call shape or phrase the request replaces, as it is written in code. new: what replaces it
   (empty for a removal). summary: the request in one sentence with old and new named.
2. queries: the exact strings that find every place that mentions old, most specific first. Start with seeds, then old
   itself, then the forms the request implies (an import specifier, a qualified name, a call with its opening
   parenthesis). Each query is a substring that occurs in code. No query is empty, and none repeats.
3. invariants: the behaviors that must stay the same, one sentence each. Take them from the request and from the
   check ids (a check named for a scenario is a behavior to keep).

Return { summary, old, new, queries, invariants }.
