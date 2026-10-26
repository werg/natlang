---
args: { attempts: AttemptTimeline[], inferences: InferentialStatus[], rivals: RivalMechanism[], snapshot: AttemptArchiveCorpus }
returns: LessonRevision[]
---
1. Bind scope = contractLessonApplicability(attempts, inferences, snapshot). Limit every lesson to the conditions actually represented by valid attempts.
2. Bind supersession = linkContradictoryAndSupersedingRuns(scope, attempts, inferences, rivals).
3. Bind history = preserveKnowledgeHistory(supersession, snapshot); create a new version instead of rewriting a former conclusion.
4. Return lesson revisions with restricted attribution retained and authorized methodological roles listed.
