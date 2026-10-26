---
args: { candidates: FollowupPrediction[], snapshot: AttemptArchiveCorpus }
returns: FollowupPrediction[]
---
Check each proposed change against approved safety controls, resource limits, and existing protocol authority. Remove only candidates that violate a hard constraint and preserve the exact reason. If authorization for a changed procedure is absent, return it as blocked pending approval rather than labeling it executable.
