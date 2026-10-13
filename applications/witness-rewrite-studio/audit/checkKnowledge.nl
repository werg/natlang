---
description: Validate changed information routes and source-provenance claims.
args:
  original: Analysis
  replay: Replay
  obligations: Obligation[]
returns: SemanticFinding[]
---
1. Select knowledge and motivation obligations that require actor information at a specific event.
   Find the corresponding replay.knowledge states and dialogue decisions.
2. Check the complete asserted conjunction. Destination knowledge can survive oral testimony while
   a line claiming to have read the original letter becomes unsupported.
3. Require the supporting acquisition path to start at an informed source and reach the speaker
   before the assertion/action. A mutual report circle does not supply an originating observation.
4. Check the receiver actually heard or read the relevant content. Seeing the seal, carrying a closed
   envelope and hearing an unrelated bell each grant different information access.
5. Preserve allowed beliefs separately from actual-world truth. An explicit misleading report can
   explain a character's belief, but cannot establish the real destination under a canon obligation.
6. Return a finding per owned obligation with its actual acquisition path referenced by source IDs,
   status, reason and exact witnesses. Use unresolved when a necessary order/reception premise is open.
7. A candidate need not retain the original information route if the director allows another. Judge
   whether the new route meets protected meaning and provenance constraints rather than byte equality.
8. Do not infer information from narrative convenience or the known oracle. Only returned replay
   states, selected event witnesses and controlling rules license the proposition.
