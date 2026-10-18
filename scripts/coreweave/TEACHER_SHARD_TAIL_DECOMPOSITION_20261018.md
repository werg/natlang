# Keep the teacher supplied during static shard tails

The supervisor's policy is to use free serving capacity while retaining every original task. A batch can have a few live static-shard workers, several queued keys behind them, and many exited workers. Requiring every key to have started prevents the already bounded second batch from using that free capacity.

1. Authenticate the unchanged queue and complete worker-journal snapshot through the existing frontier reader.
2. Require actual completed work, an unfinished batch, and at most the configured number of started-but-unfinished requests. Count unstarted keys separately, rather than as active requests.
3. Preserve every remaining and unstarted key in the capacity observation. Leave all original shard owners and tasks untouched. Reserve the next independent source packet through the existing two-slot mechanism; never admit a third batch.
4. Keep initial startup, complete batches, torn journal writes, conflicting events and foreign keys under the existing guards. This observation grants capacity only, never completion, quality or training credit.

The natural-language instruction above is the scheduling policy; exact set differences, durable source reservation and journal binding are crisp mechanisms. The executing model sees no prompt or task change.
