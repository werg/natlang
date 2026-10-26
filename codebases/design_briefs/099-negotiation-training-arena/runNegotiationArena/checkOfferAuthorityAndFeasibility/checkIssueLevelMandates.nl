---
description: Check every offered issue and condition against the exact role mandate and scenario bounds.
args:{scenario:ArenaScenario;role:ArenaRole;offer:ParsedArenaOffer}
returns:{authorizedIssueIds:string[];authorityFailures:string[];unknownIssueIds:string[];boundViolations:string[];conditionQuestions:string[]}
---
Resolve each offer term key to exactly one scenario issue. Compare its value to the issue's supplied unit and minimum/maximum bounds, preserving malformed or incomparable values as questions. Check that the acting role's authority names that exact issue and that each condition has a named satisfaction rule. Report unauthorized issues, unknown IDs, out-of-bounds values and conditions with no verifier separately. Do not infer authority from a role title or treat an exploratory term as binding.
