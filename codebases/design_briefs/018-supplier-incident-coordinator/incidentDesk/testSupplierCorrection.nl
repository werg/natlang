---
args: { request: SupplierIncidentRequest, clusters: StageResultIncidentCluster, exposure: StageResultExposureFinding, containment: StageResultContainmentPlan, evidence: EvidencePage[] }
returns: StageResultCorrectiveAction
---
Compare the supplier's causal explanation with the cluster's rival mechanisms and evidence from affected lots. Separate immediate correction from recurrence prevention. A training-only change is not sufficient unless evidence ties the mechanism to a training failure and a verification task tests that causal claim.
