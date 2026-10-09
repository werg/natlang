---
description: Migration failure triage. From a failing candidate's check output, or the message of a patch set that could not be applied, find each cause with its kind, location and evidence.
args:
  intent: Intent
  snapshot: RepoSnapshot
  validation: Validation | null
  rejected: string
returns: Finding[]
---
Find the causes of the failure of the candidate at snapshot.revision. validation holds its checks (null when none ran),
rejected is the message of a patch set that could not be applied (empty when none). Work in eval over the data.

1. A rejected message: one finding with check "apply", kind "wrong-edit", path the file the message names (empty when
   none), line 0, evidence the message, repairable true.
2. For every check of validation.checks whose status is "failed": read check.output (its last 4000 characters; truncated
   says more came before). Find each distinct failure in it and make one finding per failure, with:
   - check: check.id.
   - path and line: the file and line the output points at (a stack frame, an assertion location, a compiler
     message) in a file of snapshot.files; "" and 0 when none.
   - evidence: the output lines that show it, copied from check.output.
   - kind. Take the first that fits: "environment" when the check could not run (missing command, permission, timeout,
     check.detail is "timeout"); "missed-site" when the output reports a name, import or property that does not exist
     and it is intent.old or close to it, or the old behavior is still used; "test-expectation" when a test asserts
     the old name, value or message; "wrong-edit" when the output shows broken syntax or behavior at a place the
     migration edited; "unrelated" otherwise.
   - repairable: true for "missed-site", "wrong-edit" and "test-expectation", false for the others.
3. A failed check with no output gets one finding of kind "environment".

Return the findings in the order of the checks. When the candidate has no failure, return [].
