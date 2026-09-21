---
description: Assess each observation and retain its reason alongside exact counts.
args:
  observations: Text[]
  criterion: Text
returns: Report
---
function review(observations, criterion) -> Report
  assessments = for each observation in observations: assess(observation, criterion)
  report = summarize(assessments)
  return report
