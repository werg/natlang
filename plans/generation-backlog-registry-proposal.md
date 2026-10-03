# Generation backlog registry proposal

The read-only reporter at `scripts/report_generation_backlog.py` reads an
explicit registry file (`plans/generation-backlog-registry-20261003-v1.json`)
instead of embedding pool paths in code. That registry describes a historical
cohort: Qwen v9-v3 alternate compositions and reviewed Bunny/Luna v44
assignments. Since this registry was prepared, Qwen v9-v4 and Bunny v45 have
been approved and are running. Luna v45 is controller-approved and waiting for
its own v44 slots. Those newer assignments are intentionally not registered in
v1. The v9-v3 remainder is a historical selection remainder; v9-v4 preparation
already selected 1,024 from that older remainder. It must not be presented as
current unattempted work.

The registry and selected task artifacts define the historical cohort. The v44
worker journals and import ledger are append-only/current-state inputs, so a
later report may show more attempts or results than an earlier report from this
same registry. Each report pins the exact journal, ledger, result files, plan,
queue, and IR bytes it observed. It is a timestamped view of that cohort's
progress, not a claim that its journals never changed.

To report v9-v4, Bunny v45, or Luna v45, add their reviewed plan hashes, queue,
result/import ledger, journal, and approval linkage in a new immutable registry
JSON, then run the reporter with `--registry path/to/new-registry.json`. Keep
the v1 registry and prior report unchanged for reproducibility. The reporter's
cohort label and status should state whether it is a historical selection
snapshot or a registered progress cohort.

To extend it safely, register each pool with immutable source/selection proof,
selected IR, queue and runtime pins. Keep these dimensions separate:

1. **Targets**: exact task or payload identities available in a reviewed
   source universe, selected into a queue, excluded by a named gate, or left
   unselected. A composition over known questions is a different target from
   a new source question.
2. **Source coverage**: recursively joined source IDs, source groups, component
   program IDs, revisions, and aliases from the IR and its review proof.
   Report per-pool sets or counts; never sum overlaps as unique global
   coverage.
3. **Attempts**: queue keys plus journal start/finish events, with each key
   checked against the exact queue. Keep queued, active, terminal, duplicate,
   orphan, transport/resource-limited, and complete-export states distinct.
4. **Results**: count only artifacts reconciled by the pool's import ledger.
   Raw `accepted` fields or successful process exits do not establish current
   admission. If no admission ledger exists, report admission as unknown.
5. **Holds and failures**: source/policy holds, semantic wrong answers,
   format/admission rejection, incomplete export, and resource/infrastructure
   failure need separately pinned evidence and must not be collapsed into one
   rejection count.

For a later system-wide view, add registered static and repair universes with
their own reviewed identity and disposition ledgers. Reconcile their task
identities against all active/completed provider queues and canonical
admission records. The report should say global completeness is unknown until
every eligible universe, exclusion, alias rule, and attempt ledger is
registered and joined. Do not use catalog artifact totals as a target
denominator: they mix statuses, versions, sources, and derived artifacts.
