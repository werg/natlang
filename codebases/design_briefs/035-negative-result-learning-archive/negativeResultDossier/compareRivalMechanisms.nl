---
args: { attempts: AttemptTimeline[], inferences: InferentialStatus[], snapshot: AttemptArchiveCorpus }
returns: RivalMechanism[]
---
1. Bind causes = extractDemonstratedMechanisms(attempts, inferences, snapshot); admit a demonstrated cause only with a source-backed process deviation.
2. Bind alternatives = retainPlausibleRivals(attempts, inferences, snapshot); mark each untested explanation plausible, never demonstrated.
3. Bind challenges = attachContradictoryObservations(causes, alternatives, snapshot). Keep each challenge tied to the exact mechanism and conditions it bears on.
4. Return the surviving rival set without collapsing different mechanisms into a single generic “failure” label.
