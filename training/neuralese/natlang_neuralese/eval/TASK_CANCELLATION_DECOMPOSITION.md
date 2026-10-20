# Operational interruption of task evaluation

Owner policy: a useful change may interrupt training at any time. Preserve earned
optimizer/RNG/exposure and unfinished task/recipe obligations. An interrupted
held evaluation grants no qualification and is not a model failure.

The existing task evaluator remains the exact verifier; cancellation is crisp
process ownership and durability, not a new scheduling policy.

1. Each trainer supplies its existing stop flag to the shared evaluator.
2. The shared collector dispatcher checks that flag before starting more work.
   On interruption, record the exact queued frontier, cooperatively terminate
   only its owned collector processes, and reap them without launching pending
   cases or resuming storage-paused ones.
3. Preserve actual completed-case outcomes, mark the evaluation unscored and
   operationally interrupted, and return to the trainer's existing checkpoint
   and deferred-qualification path. Do not count it as a quality failure or
   retry it until eligible while the owner is asking to stop.
4. Retain the existing evaluator behavior when no stop is requested. Production
   mathematical checks and subsequent genuine evaluation remain required.
