---
description: Judge a claimed grant against the acts, territory and onward power of its supported parents.
args:
  claim: CompetenceClaim
  parents: AuthorityProof[]
  archive: Archive
returns: AuthorityAssessment
---
1. Gather the claim's exact clauses and the parent grants' scope witnesses. Capture their act-kind,
   territory, time and onward-delegation constraints; do not capture a presumed yes/no result.
2. In eval declare a local GrantJudgment matching AuthorityAssessment. Create nl.with<GrantJudgment>
   from that context and directly await it on the changing claim and proposed parent proof references.
3. Ask whether these parents authorize this precise grant. Require separate consideration of a power
   to act and a power to delegate that act. Preserve any missing decisive premise as permitted:null.
4. Consume true only with a supported narrowed interval and actual supportingProofIds. False keeps
   the exceeded act/territory/delegation scope in reason; null keeps the unresolved scope question.
5. Return the literal's witnessed assessment. A parent's unrelated valid power cannot substitute for
   the required power, and a later date does not enlarge the granted competence.
