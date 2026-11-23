---
description: Assess observations against a criterion, tally verdicts in code, and synthesize an overall conclusion.
args:
  observations: Untrusted<string>[]
  criterion: string
returns: Report
---
Call summarize(observations, criterion) to run the per-observation assessments concurrently and tally the verdicts.
Weigh the returned assessments and counts to write a one-sentence conclusion stating whether criterion holds overall
(naming any contradicting observation, or noting when no evidence supports it), and return the tally together with
conclusion.
