# Refill useful teacher capacity while preserving live tails

Named natural-language scheduling stage: **refill within the owned worker envelope**.
The supervisor's policy is: keep every started batch and assigned key with its
actual owner; when completed work leaves serving capacity unused, reserve the next
qualified TRAIN packet if the potential root workers across all owned unfinished
batches plus its pinned workers fit the existing 48-worker envelope. Do not wait
for an arbitrary number of whole batches to finish. Preserve ordinary/priority
fairness, current source exclusions, and every original runtime/actor receipt.

1. Observe each actual immutable queue and stable complete journal prefix. Count
   unfinished keys, including unstarted static-shard keys. Its conservative
   potential demand is `min(pinned_workers, unfinished_keys)`. No journal, torn
   observation, unknown owner, startup with no completed work, or changed source
   grants new capacity. A reserved batch that has not sealed its immutable source
   blocks another reservation until its full pinned work can be authenticated.
2. Use one ordered ahead-slot inventory for guardian, source sequence and publisher.
   Completed ahead slots retain their exact boundaries; active tails retain owner,
   token, source, workers, runtime, context and config snapshot in a distinct active
   inventory. Moving a live tail out of the newest slot never calls it completed.
3. If the exact observed aggregate demand plus the next batch fits the explicitly
   declared envelope, reserve through the existing source-selection policy and
   immutable intent protocol. Every earlier active or completed reservation counts
   as already served for the next scheduling decision; never select it twice.
4. Recover the exact token before spawning. Only one guardian owns the leases.
   Permit its authenticated active slots, reject foreign feeders, and preserve all
   original workers and engine processes. A dead materialized incomplete slot
   needs reconciliation, not a duplicate launch or capacity credit.
5. Seal completed slots independently, publish each live/completed slot through its
   own exact source/process/frontier checks, and adopt them into the serial cursor
   only after the genuine immediate predecessor closes. Preserve later slots and
   priority history through crashes before/after every state publication.
6. Keep absent-envelope behavior unchanged. Opt-in configuration declares the
   actual 48-worker bound; it introduces no arbitrary third-batch ceiling. Existing
   feeder, source oracle/qualification, protected split, runtime and teacher
   identity mechanisms remain authoritative.

Natural language owns the scheduling choice above. Crisp helpers implement only
exact queue/journal arithmetic, immutable identity, source intent/lease durability,
process recovery and publication verifiers. The scheduling stage has a named
interface and a deterministic verifier for its concrete observation and proposed
reservation; no model-facing prompts or inference semantics change.

Validation uses real recovered source/state/queue fixtures: 35 and 37 tails admit
38 only when the sum is bounded; unstarted keys remain counted; four or more
small live tails work without a batch-count cap; over-capacity/startup/torn/foreign
observations fail closed; source priority selection accounts for every live
reservation; owner death, boundary-before-archive interruption, serial adoption
and publisher discovery preserve all later source/frontier bindings.
