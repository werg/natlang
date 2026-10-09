---
args:
  request: DiagnoseRequest
returns: Diagnosis
---
Say what the training evidence shows about the program, relative to request.goal. The result is a diagnosis: observations and the pattern they share.

1. Read request.opportunity.kind and request.opportunity.reason. They name which rows of request.evidence to read: the failed rows for `quality`; the passing rows that used more than one model request for `efficiency`; the passing rows for `source-size`.
2. For each such row write one observation with these fields. caseId is the row's caseId. whatHappened states what the program did: its result, the files it wrote and the actions in its trace. expected states what the goal asks for on that row. category is `wrong-result` when the result differs from the expected one, `execution-method` when the trace calls something that is unavailable or uses a file method that does not exist, `wasted-action` when a model request or tool call did not change the result, and `fixture` when the row failed before the program ran.
3. When request.lastExperiment is present, add one observation per training row that still fails, naming what that experiment changed and what its measurements show. Read request.history for the outcomes of earlier experiments.
4. Write pattern: one sentence on what several observations have in common. When the observations share nothing, write the most frequent category and the case ids it covers.

request.brief is a compact card of the same evidence; read it first and open a row of request.evidence when a detail is needed.
