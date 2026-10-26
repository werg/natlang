---
args: { release: CompatibilityReleaseRequest }
returns: ReleaseBehaviorDelta[]
---
For each changed code path compare baseline and candidate behavior under the stated preconditions. Treat default value, order, retry, and error changes as behavioral changes even with an unchanged schema. Link each claimed delta to a diff, observed trace, or targeted verification request.
