---
args: { blockers: UnresolvedQuestions, snapshot: Snapshot }
returns: QuestionBatch
---
1. Group questions only when they seek the same missing fact; name an authorized respondent and acceptable evidence; do not treat a general acknowledgement as resolution.
2. Group questions sharing a missing fact, identify an authorized respondent, and specify acceptable evidence. Deduplicate prior requests; interpret each reply against its question rather than treating a general acknowledgment as resolution of every blocker.
3. Return an explicit artifact; preserve unknowns and source coverage. This procedure proposes no production write.
