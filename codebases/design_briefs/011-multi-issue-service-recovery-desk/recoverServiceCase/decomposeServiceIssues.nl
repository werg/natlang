---
description: Reconstruct the service timeline as independently resolvable but promise-linked issue episodes.
args:{scope:GoalContract;policy:PolicySelection;cases:RecoveryCase[];timeline:EvidencePage[]}
returns:StageResultServiceIssue
---
Bind episodes = `reconstructIssueEpisodes(cases, timeline)`. Connect each failure to its order, requested customer outcome and relevant promise IDs. Split only when one remedy can close without altering another issue's entitlement or value. Preserve timeline conflicts, missed evidence and all candidate IDs in a typed result; do not erase an expiring offer because a later replacement arrives.
