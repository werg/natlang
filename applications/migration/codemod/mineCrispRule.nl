---
description: Synthesize a reusable deterministic crisp find-and-replace rule from verified semantic file migrations.
args:
  spec: CodemodSpec
  pairs: VerifiedMigrationPair[]
returns: MinedRuleCandidate
---
Inspect the verified before and after code pairs that were successfully migrated for spec.
Identify the common mechanical transformation repeated across those pairs and express it as a single reusable find-and-replace rule.
Return an object with description, find, replace, and optional regex (true for a regular expression pattern, false or omitted for an exact literal replacement) and flags.
Make the pattern specific enough that it will not alter already-migrated or unrelated code.
