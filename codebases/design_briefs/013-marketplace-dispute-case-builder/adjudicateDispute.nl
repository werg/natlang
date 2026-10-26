---
description: Assemble a transaction-bound dispute review from attributed chronology, decisive questions, evidence, and conditional rules.
args: { request: DisputeBuildRequest; records: DisputeRecordSet }
returns: DisputeProgramResult
---
Bind scope = `bindTransactionScope(request, records.caseRecords)`. Require a unique match on case, transaction and both parties; keep identity or version mismatch blocked. Bind rules = `selectTransactionRules(request, records.policyCandidates)` using authority, jurisdiction and the transaction-time edition, never review-date recency.

Bind chronology = `reconstructPurchaseChronology(scope, records.evidencePages)`. Keep listing-at-purchase distinct from later edits; attribute each party statement to its source version. A record timestamp says when it was logged, not that its described event happened. Bind questions = `deriveDecisiveQuestions(chronology, rules)`, retaining immaterial disagreements separately and asking only facts that can change burden or outcome.

Bind evidence = `evaluateQuestionEvidence(questions, records.evidencePages)`. Separate authenticity from semantic support, collapse duplicate copies for weight, and preserve independently corroborating sources. Bind disposition = `applyConditionalDisposition(scope, rules, evidence, records.evidencePages)`. For every unresolved dispositive premise show each plausible outcome; incomplete search never becomes adverse proof. Require current ledger evidence before an added payment remedy and configured outage overlap before extending a response deadline. Route beyond-delegation interpretation for authorized review.

Bind dossier = `assembleContestableDossier(scope, questions, evidence, disposition, records.evidencePages)`. Represent both parties and response windows, restrict public addresses without deleting provenance, and reopen only findings materially affected by appeal evidence. Keep reviewer disposition separate from recommendation. Return commitAllowed false; this is a balanced review packet, not an adjudication or payment effect.
