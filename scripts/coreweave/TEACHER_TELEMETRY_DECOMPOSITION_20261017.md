# Durable teacher load observation

The recovery instruction is natural language: preserve the last observable host,
process, GPU and engine state outside the generation container before loading.
This observer makes no restart, allocation, memory-admission or model decision.
Its crisp mechanisms sample Linux counters and bounded log deltas, transmit one
JSON record per interval over one SDK exec stream, and fsync each received record
on the owner host. A missing sample or reset is evidence of missing observation,
not an inferred OOM. Clean file cache is not asserted reclaimable when a CUDA
driver may have registered pages.

1. Bind the expected container boot, actual sampler identity and selected log
   paths. Sample host and cgroup counters without constructing a model.
2. Include identities and memory counters of current engine/guard descendants;
   never emit their environment or complete command line.
3. Read bounded log deltas and GPU counters with a bounded subprocess call.
4. Flush each record. The owner appends it to a new, immutable-in-use JSONL and
   fsyncs it before acknowledging local readiness. One persistent SDK exec owns
   the sampler; it exits on signals, boot mismatch, broken output or the declared
   diagnostic duration. No detached children are started.
5. Preserve the entire received stream and terminal/error record if the provider
   destroys the container. A later boot starts a different observer, not a rewrite.

Actual third-role reset left a 45-second observer gap despite a ten-second interval. The cause remains unknown; synchronous `smaps_rollup` and GPU-driver queries can block. An explicit `--fast-host-only` mechanism omits those operations, while retaining host/cgroup counters, process status, birth checks and bounded engine-log tails. The existing detailed mode remains the default. Records name their sampling scope; terminal receipts retain a bounded stderr tail. This is observer plumbing, with no recovery/serving decision or model-facing change.
