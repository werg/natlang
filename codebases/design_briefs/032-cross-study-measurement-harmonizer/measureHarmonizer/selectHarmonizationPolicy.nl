---
args: { request: HarmonizationRequest, corpus: MeasurementCorpus }
returns: PolicySelection
---
Select only a policy record effective for the requested review version, exact data owner, dataset scope, and protocol. Keep every original policy identifier and clause available to the review. If the policy collection is absent, no rule covers the named datasets, or owner authority conflicts, return no selected authority and name the precedence question; do not infer permission from a case description.
