---
args: { attempts: AttemptTimeline[], inferences: InferentialStatus[], snapshot: AttemptArchiveCorpus }
returns: RivalMechanism[]
---
Create demonstrated mechanisms only from an observed deviation, instrument fault, data-loading error, or other directly evidenced causal event. A valid negative result that conflicts with a hypothesis is not itself a demonstrated explanation for the hypothesis being wrong.
