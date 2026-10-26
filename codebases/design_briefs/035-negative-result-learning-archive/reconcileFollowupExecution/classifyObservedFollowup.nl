---
args: { prediction: FollowupPrediction, predictionRef: Ref, receipt: FollowupExecutionReceipt }
returns: FollowupOutcome
---
1. Require receipt.predictionRef to equal predictionRef and receipt.explanationId to equal prediction.explanationId. A different registered prediction cannot lend its forecast to this run.
2. Classify measurementValid "invalid" as invalid-run, "unknown" as uncertain, and a documented incomplete run as incomplete. None of these outcomes supports either forecast.
3. For a valid complete run, compare the actual outcome with both predeclared predictions, using the supplied observed conditions and deviations. Favor "if-true" or "if-false" only when the result fits one prediction materially better under the observed conditions. If both, neither, or neither can be compared, return "neither" or "unresolved" and state the inference limit.
4. Preserve the exact prediction and run evidence references. A follow-up result narrows a rival only under its observed conditions; it does not prove universal causation or erase prior attempts.
