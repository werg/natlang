---
args: { rivals: RivalMechanism[], inferences: InferentialStatus[], snapshot: AttemptArchiveCorpus }
returns: FollowupPrediction[]
---
Only propose a new trial when at least two consequential rival mechanisms make different predictions. Bind change = chooseMinimalChange(rivals, snapshot), predictions = declarePredictionsBeforeExecution(change, rivals), and authorized = verifyFollowupConstraints(predictions, snapshot). If the prior attempt was invalid, repair the process first and do not vary several scientific factors at once. Return only predeclared, safe predictions; any missing approval remains a blocker.
