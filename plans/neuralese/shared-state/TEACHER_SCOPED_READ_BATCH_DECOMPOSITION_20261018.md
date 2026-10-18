# Read completed teacher tranche members concurrently

The generation policy and completed-row selection stay in the existing guardian
and exporter. This change is crisp transport plumbing: collect the exact bytes
already selected for one immutable tranche with up to four scoped SDK reads in
flight. The observed 68 serial reads for batch41 took about ten minutes while
generation continued. There is no new model-facing instruction or source.

1. Select completed rows under the existing exclusive export lock and captured
   queue/journal snapshots. Resume a sealed tranche before collecting anything.
2. Enumerate the same context, output, manifest and held-error paths in their
   original order. Reuse exact duplicate paths within this collection only.
3. The scoped SDK reader supplies an ordered multi-read capability with four
   worker threads. Each read retains its existing timeout. Ordinary callable
   readers retain sequential execution. On error, cancel queued work and drain
   active reads before releasing the export lock; publish no tranche/frontier.
4. Reconstruct the same context/output structures in original order and use the
   unchanged source, accounting, archive and private-receipt verifiers. Advance
   the frontier and prune only after actual independent private verification.
5. Check overlapping bounded reads, identical sealed members and member order,
   failure before publication, and exact sealed-tranche continuation. Adopt a
   reviewed immutable publisher successor only at its normal idle boundary.
