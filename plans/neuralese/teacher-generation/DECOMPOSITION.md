# Continuous self-hosted teacher supervision

Policy: continue useful broad generation after a drained batch while retaining every unresolved case as explicit repair debt. Never turn an incomplete batch into a completed batch. Honor an operator drain before launching another batch or repair collector. Run at most one repair collector alongside the existing broad collector group; retain original source/index/seed, actor and runtime provenance. No admission follows from supervision.

1. Read the exact feeder identity and wait for its natural exit. This is a crisp process-identity service; it does not signal active inference.
2. Read the frozen batch queue, journal and score. Use the existing exact `teacher_batch_completion` verifier. The policy routes a fully closed batch forward, or records unresolved keys as repair debt and routes the next useful batch forward. An absent batch boundary or a still-live collector requires repair before routing.
3. Write the decision, original immutable input bindings and each observed finish atomically. Atomic writes, process tokens and a lock are crisp durability mechanisms. A prepared process token is recovered before any replacement launch, preventing duplicate feeders after a supervisor crash.
4. Run the existing feeder for one batch, using its existing task authoring, sampling and backpressure. Run exact debt entries once through the existing queue worker in distinct output directories, at request concurrency one. The supervisor never implements an inference or scoring alternative.
5. On operator drain, launch neither a feeder nor a repair collector. Let already active calls finish. Resume from the durable state when the drain file is removed.

The numbered policy above is the natural-language program. `scripts/coreweave/teacher_generation_guardian.py` is its exact outside-world/durability interpreter; completion and inference remain the existing shared mechanisms.
