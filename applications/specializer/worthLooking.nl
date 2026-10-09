---
args:
  candidate: WorthCandidate
  min_calls: number
returns: WorthAnswer
---
Decide whether the specializer looks at a natural-language function now. Looking costs model calls, so it is worth it when many recorded agent calls, still unspecialized, would be served by code. candidate holds the counts; min_calls is the smallest number of agent calls worth a look.

1. When candidate.agent_calls is below min_calls, answer look false, reason: too few calls.
2. When candidate.tokens is 0, answer look false, reason: nothing to save.
3. When candidate.decline is present and candidate.agent_calls is below twice candidate.decline.calls_at_decline, answer look false, reason: the earlier decline stands until the volume doubles.
4. When candidate.has_compilation is true and candidate.new_calls_since_compilation is below min_calls, answer look false, reason: too few new calls since the last compilation.
5. Otherwise answer look true, reason: enough unspecialized volume.

Write the reason as one sentence that names the deciding count.
