---
description: Resolve actual discourse context while preserving candidate semantic differences.
args:
  target: TargetRequest
  hypothesis: GrammarHypothesis
  roles: RoleAssignment
  source: CorpusSource
returns: ContextMeaning
---
1. Directly invoke a typed scoped NL ContextMeaning literal capturing the actual target's original discourse clauses, actual survivor choices and actual returned role prediction. No final expected translation enters scope.
2. Resolve named discourse topic/contrast referents from their actual original context IDs. Mentioned entities not established as the target's topic stay contextual alternatives or null.
3. Preserve actual predicted event roles and polarity. A clear physical scene is not permission to repair a linguistically incompatible prediction.
4. Interpret a candidate's source-declared temporal contribution separately from contextual event time. A not-yet prediction can add prospective expectation even when an event-negation-only survivor leaves that expectation unspecified.
5. If the source explicitly cancels or licenses an implicature for this target, apply only that source-defined contextual rule and cite it. Unmentioned future facts remain untested, not false.
6. Return original entity/event/topic IDs, temporalReading, actual futureExpected contribution, exact original context/grammar quotes and unresolved premises. Free gloss prose is explanatory, not an exact output encoding.
7. Parent binds original witness identity and consumes this actual structured meaning; later grouping retains consequential distinctions rather than choosing a fluent preferred paraphrase.
