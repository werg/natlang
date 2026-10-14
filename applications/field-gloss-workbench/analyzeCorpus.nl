---
description: Infer source-grounded grammar alternatives, target meanings and an authorized note repair.
args:
  request: LinguisticRequest
  snapshot: CorpusSnapshot
returns: LinguisticReport
---
1. Bind actualSource to intake(snapshot, request). Preserve original corpus identities and any unresolved session/correction premises; a later stage cannot replace them with a preferred grammar.
2. Call buildEvidence(actualSource) and bind actualGraph. Its effective-note and response-scope decisions determine which comparisons are licensed, separately from which hypothesis eventually survives.
3. Call inferGrammar(actualSource, actualGraph). Bind actualGrammar, retaining every actual attempt and first contradiction rather than only the surviving labels.
4. Call translateHeldOut(request.targets, actualSource, actualGraph, actualGrammar). Bind actualTranslations. Every target must consume the actual survivor interpretations, including consequential ambiguity.
5. Call repairGlossNotes(request.repair, actualSource, actualGraph, actualGrammar, actualTranslations). Bind actualPatch. Only the requested original gloss slot can change; uncertain distinctions remain explicit in the replacement.
6. Pass these actual typed objects to auditReport(request, actualSource, actualGraph, actualGrammar, actualTranslations, actualPatch). Return its report without reconstructing a cleaner ledger or substituting an authored answer.
7. Distinguish a rejected linguistic hypothesis from an unusable elicitation or a failed child return. Preserve those separate reasons in the actual report so missing premises cannot masquerade as negative consultant judgments.
