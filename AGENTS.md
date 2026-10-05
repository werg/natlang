# Machine ownership and coordination

- Use `/home/werg/natlang` as the canonical development checkout on both Pop and
  DGX. The Pop agent owns execution/resource management on Pop; the DGX agent
  owns execution/resource management on DGX. Do not launch new jobs from retired
  `natlang-remote` snapshots or manage the other machine's jobs without a request.
- At session start, before changing shared resources, and at each monitoring
  cycle, run `python3 scripts/coordination_inbox.py check --ack`. The gitignored
  `.coordination/inbox.md` contains incoming machine-specific notes. Read them,
  append responses to the sender's inbox, and preserve history. While monitoring
  long jobs, check at least once per 50-minute work/sleep cycle.
- Synchronize code through small commits and frequent fetch/merge/push to
  `origin/main`; preserve the other agent's uncommitted work. Do not copy source
  trees between machines. See `plans/MACHINE_COORDINATION.md`.
- Publish data additions in `training/neuralese_corpora.json` with immutable
  SHA-256 manifests under `training/corpus-manifests/`; transfer artifacts with
  `scripts/sync_training_corpora.py`. Copies, availability, and successful schema
  checks do not grant training admission. Keep split/source/quality decisions
  explicit. Register omissions and required conversions rather than losing them.
