# Machine ownership and coordination

- Use `/home/werg/natlang` as the canonical development checkout on both Pop and
  DGX. The Pop agent owns execution/resource management on Pop; the DGX agent
  owns execution/resource management on DGX. Do not launch new jobs from retired
  `natlang-remote` snapshots or manage the other machine's jobs without a request.
- Agents talk through `scripts/coord.py` (see "Messages" in
  `plans/MACHINE_COORDINATION.md`). At session start, before changing shared
  resources, and at each monitoring cycle (at least every 50 minutes), run
  `python3 scripts/coord.py inbox --ack`. Answer requests with `coord.py reply`
  or `close`; send with `coord.py send --to pop|dgx|all --subject ...`. Keep
  your machine's state in `coord.py status --set`, not in a stream of notes.
- Give each concurrently running agent its own coordination reader identity:
  use `coord.py --as <machine>-<agent-role>` or `COORD_AS`. Codex helpers must
  not acknowledge the owner's default `<machine>-codex` cursor. Forward
  decisions and requests outside your task to the owner; acknowledgement is
  not implementation or acceptance of a decision.
- Synchronize code through small commits and frequent fetch/merge/push to
  `origin/main`; preserve the other agent's uncommitted work. Do not copy source
  trees between machines. See `plans/MACHINE_COORDINATION.md`.
- Publish data additions in `training/neuralese_corpora.json` with immutable
  SHA-256 manifests under `training/corpus-manifests/`; transfer artifacts with
  `scripts/sync_training_corpora.py`. Copies, availability, and successful schema
  checks do not grant training admission. Keep split/source/quality decisions
  explicit. Register omissions and required conversions rather than losing them.

# Neuralese training foundation

- New neuralese training lineages use shared declared recipes under
  `training/neuralese/recipes/`; see `plans/neuralese/TRAINING_RECIPE.md`.
- Token-aligned identity and qualified causal output-state to raw next-token
  embedding distillation precede compression/recurrence. Finishing a warm-up's
  step count is not qualification. Preserve and diagnose failed gates.
- Runtime transport/gradient replay needs separate qualification against those
  exact weights. Legacy marker/RMS checkpoints do not inherit the new foundation
  certificate, and backbone changes require channel requalification.
