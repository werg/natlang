---
args: { delta: PolicyDeltaSet, pair: PolicyReplayPair }
returns: PolicyOutcomePair
---
Compare this verified baseline/candidate pair on its exact frozen case input. Identify condition-linked differences in duties, permissions, timing and escalation, preserving which policy change caused each difference. Keep the source policy hashes and run identities intact. If provenance or execution configuration differs, return the pair as unresolved rather than attributing the difference to policy.
