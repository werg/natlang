---
description: Decides whether one large file under the run and data roots is kept, in use, deleted automatically or listed for the owner, from its computed facts and the storage policy's day thresholds.
args:
  facts: RetentionFacts
  thresholds: RetentionThresholds
returns: RetentionVerdict
---
facts describe one large file: what it is, how old it and its run are, and what refers to it. We keep the active lineage and what it needs, registered training and source data, held evaluation data and the records of runs; checkpoints of discarded runs, leftover partial writes and superseded intermediate checkpoints go. Go through the steps in order and stop at the first that applies.

1. If facts.registered is true: tier KEEP, reason "registered corpus or artifact file".
2. If facts.named_by_reference is true: tier KEEP, reason "named by a run script, recipe, registry, plan or certificate".
3. If facts.record_file is true: tier KEEP, reason "log, receipt, certificate or labels".
4. If facts.policy_keep_path is true: tier KEEP, reason "policy keep path".
5. If facts.in_use is a string: tier ACTIVE, reason facts.in_use.
6. If facts.run_age_days is less than thresholds.finished_after_days: tier ACTIVE, reason "its run changed recently".
7. If facts.partial_write is true and facts.age_days is at least thresholds.leftover_after_days: tier PRUNE-AUTO, reason "leftover partial write".
8. If facts.category is "checkpoint", "optimizer-state" or "weights-snapshot", facts.run_marked_stopped is true and facts.age_days is at least thresholds.aborted_after_days: when facts.directory_named_by_reference is true, tier PRUNE-ASK, reason "checkpoint of a stopped run whose directory a plan or script names"; otherwise tier PRUNE-AUTO, reason "checkpoint of an aborted or stopped run".
9. If facts.category is "checkpoint" or "optimizer-state", facts.step is a number, facts.step_files_in_directory is more than 1, facts.step is less than facts.latest_step_in_directory and facts.best_or_final is false: tier PRUNE-AUTO, reason "intermediate checkpoint".
10. Otherwise: tier PRUNE-ASK, reason "large " followed by facts.category and " file without a keep rule".
Return { tier, reason }.
