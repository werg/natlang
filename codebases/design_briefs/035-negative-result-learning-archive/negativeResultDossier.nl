---
args: { archiveRequest: ArchiveRequest, attemptStore: AttemptArchiveCorpus }
returns: NegativeResultDossier
---
1. Bind attempts = reconstructAttemptHistories(archiveRequest, attemptStore). Reconcile intended protocol steps with actual log events and notes in event order; do not substitute the plan for an execution record.
2. Bind inferences = adjudicateInferenceValidity(attempts, attemptStore). Classify each attempt separately as valid-negative, measurement-invalid, incomplete, or uncertain. Only a completed valid measurement may update evidence against its hypothesis.
3. Bind sharedProcessIssues = connectSharedProcessFailures(attempts, inferences, attemptStore). One broken loader or instrument is one linked process explanation, not duplicate scientific evidence across attempts.
4. Bind rivalMechanisms = compareRivalMechanisms(attempts, inferences, attemptStore). Keep observed causes distinct from plausible mechanisms and attach contradictory evidence to the exact rival it challenges.
5. If a finding is valid-negative or its causal status is uncertain, bind followupPredictions = designDiscriminatingFollowups(rivalMechanisms, inferences, attemptStore). Every proposal changes a bounded variable set, declares predictions before execution, and respects protocol, safety, and resource limits. If the attempt was invalid or unfinished, first repair that process defect.
6. Bind lessonRevisions = reviseScopedLessons(attempts, inferences, rivalMechanisms, attemptStore). Preserve prior versions, narrow applicability after a later success under different conditions, and link contradictory or superseding evidence without hindsight rewriting.
7. Bind openQuestions = requestRestrictedMethodReview(attempts, attemptStore). When attribution is sensitive, expose the method to an authorized reviewer without revealing a restricted researcher identity to general readers.
8. Return { caseId: archiveRequest.caseId, attempts, inferences, rivalMechanisms, followupPredictions, lessonRevisions, sharedProcessIssues, openQuestions, commitAllowed: false }. This is an epistemic proposal; it does not publish, rewrite, or execute a follow-up.
