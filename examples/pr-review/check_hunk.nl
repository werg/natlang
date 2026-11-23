---
description: Check one pull-request diff hunk against the bound team review rule.
args:
  hunk: Hunk
returns: HunkReview
ensures:
  - check: "quotesMatch(result, hunk)"
    message: every finding's quote must appear verbatim inside hunk.patch
  - predicate: every finding identifies a genuine violation of rule in hunk with an actionable fix
    against: [hunk]
---
Inspect hunk.patch against rule (the team review rule bound in this context). Report only genuine violations of rule
introduced in the added lines of hunk.patch. For each finding, copy the offending line or expression verbatim into
quote, set file to hunk.file, rule to the rule name or summary, choose severity "blocker" or "suggestion", and write a
concrete fix. Return { findings: [] } when hunk follows rule.
