---
args: { attempts: AttemptTimeline[], inferences: InferentialStatus[], snapshot: AttemptArchiveCorpus }
returns: string[]
---
Cluster only attempts that cite the same exact instrument, loader, reagent lot, or protocol-process record. A shared broken process explains why multiple outcomes are invalid but does not multiply the number of independent scientific replications. Keep unrelated failed attempts separate even when their outcomes look similar.
