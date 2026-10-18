# Release completed collection capacity without losing source order

The scheduling policy is: retain every actual collection and its authority, seal a completed lookahead independently, and let the next qualified packet use released serving capacity while an older collection finishes. A completed collection consumes no worker capacity. Source cursor adoption still follows the original reserved order; completion order cannot select or relabel tasks.

1. Observe the original live tail using the existing exact queue/journal capacity verifier. Keep its owner, queue, unstarted jobs and runtime unchanged.
2. Authenticate completed lookahead source/start/phase/boundary and move its immutable slot into completed-ahead custody. Do not declare the older collection complete or advance its source cursor.
3. Reserve the next packet by replaying the existing source policy over the exact ordered pending/reserved chain. Each intervening completed slot must have an authenticated boundary. Priority service/fairness counts each reservation once.
4. Launch through the same single guardian and existing feeder with its pinned runtime/config/worker bindings. At most the configured live worker pools remain; completed slots do not impose a numeric batch-distance ceiling.
5. When the older collection finishes, adopt and close retained slots in order, preserving any later live lookahead. Export closed and live slots independently with exact per-batch authority and private frontiers.

Natural-language policy is stated above; crisp helpers authenticate source bindings, process births, exact ordering, durable reservations and atomic state transitions. No model-visible prompts, execution semantics, model allocation or engine configuration change. Regression checks cover repeated completed lookahead, wrong/missing boundaries, source mutation, crash/recovery, priority accounting, publication and eventual ordered adoption.
