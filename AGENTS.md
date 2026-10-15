# Main goal: keep moving forward (owner, 2026-10-11)

## Execution and generation ownership (owner, 2026-10-15)

Pop is closed. The DGX supervisor now owns all remaining work and sandbox
operations; do not defer work to a Pop agent. Only Pop's generation was to
finish: DGX-supervised data generation must continue. Generate new training
data only with our own models through the CoreWeave sandboxes, not Luna or
other external teachers. Existing admitted third-party data remains usable.
Coordinate sandbox inference with training so generation uses available
capacity without unnecessarily starving training. Preserve exact actor,
checkpoint, runtime and split provenance; base-model generations are not
on-policy samples from a trained checkpoint. This supersedes older machine
ownership and provider-generation restart instructions below and in plans.
The owner expressly authorized one extra sandbox for continuous own-model data
generation on 2026-10-15. Our current resource plan is two training sandboxes
plus that one generation sandbox; no further allocation is planned. Keep the
generation sandbox supplied with useful tasks and admit verified results into
the evolving training recipe. Measure actual CPU/RAM need on the allocated
machine rather than delaying useful generation to test smaller allocations.

## Neuralese architecture replacement (owner, 2026-10-15)

The destination is shared positional KV and hybrid-state spans, replacing the
projected-value channel in BOTH training and inference. These instructions
supersede older projection/recursive graph-replay foundation rules for new work.

- Neuralese tokens use ordinary token embeddings with a learned low-rank
  transformation in embedding space. Do not train or collect a projection into
  a separate communication latent space in this phase.
- Producers supply positional KV spans and other shareable hidden state;
  consumers splice that state with correct positions. Handle LFM convolution
  and linear state explicitly. Train teacher-forced graphs in one graph forward
  pass using masks and state handling, rather than recursive graph replay.
  Verify outputs and gradients against explicit cached execution; do not claim
  one-pass execution while secretly replaying the graph.
- Use different beginning AND end markers for producing versus consuming
  Neuralese. Producers learn independently useful exported state; consumers
  learn to use imported state. Stored Neuralese includes text and a sequence
  of token embeddings. Preserve tokenizer provenance and make cross-tokenizer
  matching explicitly best effort, not assumed exact state interchange.
- After several networks are trained, train a VAE through a shared joint
  semantic adaptation space to predict deltas reproducing source KV in the
  target. Follow with behavioral distillation/self-supervised SFT. Those
  per-token semantic encodings become the shared dialect. Do not claim that
  this later adapter is already trained or substitute the old projection.
- Replace the old approach end to end: shared trainers, conversions, inference,
  block storage/transport, natlang harness, evaluators and recipes. Retain all
  usable source data and protected evaluation splits through conversion, and
  share mechanisms between LFM and Mellum.
- No prototyping shortcuts or half-solutions at architectural blockers. Work
  out both training and inference. After implementation, a separate agent must
  independently check the integrated work for gaps before cutover.
- Encode cross-model readiness and work directly in the training recipe and
  executable scheduling: begin shared-space VAE work as soon as at least two
  models are ready. Provide resumable, adjustable VAE, control write-out and
  additional joint training, so useful feedback changes the training rather
  than waiting for a clean run. A placeholder stage or manual note is not an
  implemented cross-model training provision.
- Current old-paradigm cloud runs continue as DIAGNOSTICS during implementation
  and local validation. Then restart BOTH models on the entire production
  recipe using the replacement. Continued legacy training is not replacement
  completion. Preserve useful diagnostic parents until cutover/dependency
  checks; then apply the standing discarded-work hygiene policy.

Track implementation and cutover in
`plans/neuralese/SHARED_STATE_MIGRATION_20261015.md`.

Owner correction, 2026-10-15: the lodestar is comprehensive training of BOTH
models through the ENTIRE production recipe and all usable available data.
Stages are steps to progress through, not endpoints. Half measures, short
probes and partial stages do not count as production completion. Track whole
recipe progress and actual data exposure; integrate newly prepared sources
through explicit continuations. KL distance from the base is telemetry only:
improvement requires changing the model. Do not stop or deny qualification
solely for base-relative KL or agreement drift. Keep actual task-performance,
held-loss, numerical and runtime checks; diagnose and repair real regressions.
Runs may be stopped and tweaked at any time when an improvement warrants it
(owner clarification, 2026-10-15). The supervising agent decides whether the
problem requires a redo or continued training, preserving exact state and
recording the evidence. Comprehensive training is the objective, not a ban on
interruptions or sensible course corrections.

## Adaptive production training (standing owner instructions, 2026-10-15)

- **Best possible models as soon as possible is the objective. We are not
  doing science.** Do not optimize operations for clean, frozen, reproducible
  runs or controlled comparisons at the expense of useful training. Existing
  runs are work in progress, not experiments that must remain untouched.
- **Constantly improve the recipe and keep going.** Monitor both models and
  their processes, investigate concrete impediments, and greedily exploit useful
  data, fixes and insights as soon as they are available. Do not let repeated
  probes, qualification windows, packaging or handoffs become substitutes for
  production training. Use focused checks that resolve a real correctness or
  operational concern, then return the GPUs to useful training promptly.
- **Comprehensive means the entire production recipe on both models and all
  usable available data.** Stages may differ, but progress through them; a short
  warmup, partial epoch, isolated stage or smoke run is not the goal. Track
  actual stage progress and data exposure, and resolve what prevents remaining
  stages or data from being used. Do not claim that downloaded or registered
  data has been trained on.
- **Adapt live.** Add newly usable data promptly rather than waiting for a
  distant clean run or an unrelated conversion batch. Stop/reload/resume at a
  safe update boundary when needed; retain optimizer/RNG state when continuing
  is appropriate. The supervisor may stop and tweak any run at any time for a
  potential improvement and judges whether to continue or redo it. Preserve
  useful checkpoints and record the practical reason; do not require a new
  lineage or a clean experimental restart for routine changes.
- **Checks serve production.** Keep checks that protect actual task quality,
  valid gradients/numerics, held-data separation, data quality, recoverable
  progress and functioning runtime/transport. Similarity to the untrained base
  is not the objective: base-relative KL/agreement are telemetry, never a stop
  or qualification barrier. Exact state and artifact identities support safe
  continuation and debugging; they are not a scientific-purity requirement.
- **Share code as much as possible and distribute all generalizable insights
  across all models.** A useful fix, optimization, data improvement or recipe
  insight on one line must promptly be considered and propagated to the other
  lines and machines. Extend the shared trainers, loaders, objectives,
  evaluators and recipe inheritance instead of creating per-model forks.
  Keep only necessary architecture-specific behavior/settings separate, state
  why they differ, and communicate shared decisions through coordination.

These instructions supersede earlier requirements for frozen experimental
runs, base-drift hard stops or clean-run comparisons where they conflict.

The main goal in everything is to keep moving the work forward: move training
along and keep the machines doing useful work. Don't be overly cautious, and
don't wait for the owner to approve decisions. If something is sensible, do it,
record why, brief the other machine, and keep going. Raise with the owner only
what is truly theirs to decide, and keep working on everything else in the
meantime.

Don't reject an approach because our implementation of it failed (owner,
2026-10-11). A worse loss, degraded behaviour or an unexplained numerical gap is a
bug until proven otherwise: compare against an exact reference, localize the
divergence, fix it, and re-measure before judging the approach itself.

When something doesn't make sense, assume a bug and run a wide, varied dragnet
for it (owner, 2026-10-11). Usually there is a bug. Don't give up, and don't settle
for the naive explanation ("BF16 noise", "the approach doesn't work", "the model
can't learn this"). Check many hypotheses in parallel: data and masking, rendering
and templates, dtype and upcasts, caches and state carried between steps, kernels,
optimizer state and resume, gradient flow, loss weighting, evaluation code, and
differences from a known-good path. Record what was ruled out and how.

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
- Coordination is status values plus exact verifiers (owner, 2026-10-12;
  protocol in `plans/MACHINE_COORDINATION.md`): no per-commit notes, one status
  page per machine, requests only for actions on the other machine, closure by
  the shared manifest verifier (`coord.py close-verified`) instead of
  acknowledgement chains, and one announcer per lineage.
- Give each concurrently running agent its own coordination reader identity:
  use `coord.py --as <machine>-<agent-role>` or `COORD_AS`. Codex helpers must
  not acknowledge the owner's reader cursor (inspect `coord.py whoami`). Forward
  decisions and requests outside your task to the owner; acknowledgement is
  not implementation or acceptance of a decision.
- Synchronize code through small commits pushed often to `origin/main`, using
  ordinary git: pull, merge and rebase as usual (owner, 2026-10-11), and keep the
  checkout current with main. Commit your own work promptly and stage only your
  paths; don't discard or stash away changes you didn't make.
  `scripts/git_publish.py` remains available as a helper. Do not copy source
  trees between machines. See `plans/MACHINE_COORDINATION.md`.
- Publish data additions in `training/neuralese_corpora.json` with immutable
  SHA-256 manifests under `training/corpus-manifests/`; transfer artifacts with
  `scripts/sync_training_corpora.py`. Copies, availability, and successful schema
  checks do not grant training admission. Keep split/source/quality decisions
  explicit. Register omissions and required conversions rather than losing them.
- Licenses (owner, 2026-10-10): every dataset and model license is acceptable,
  including the most restrictive. Include them; record license provenance as a
  fact, but never omit or hold data for license review, and do not ask the owner.

# Storage hygiene and discarded work (standing owner order, 2026-10-11)

Amendment (owner, 2026-10-13): make deletions less eager. With the external disk
available for space, lean towards moving data to `/mnt/external` rather than
deleting it when unsure.

- Delete discarded approaches and runs: their checkpoints, optimizer states,
  exports, frozen runtime copies, diagnostic outputs, obsolete recipes and run
  tracking. Do not archive them, move them to another disk or machine to avoid
  deleting them, or retain them because an old note or manifest references them.
  Remove obsolete registry entries, manifests and handover tracking as well.
- Keep the active lineage and its required parents, reusable training/source
  data, and protected/held evaluation data. Distinguish these from discarded run
  artifacts before deleting; check live process references and current recipe
  dependencies. A regression still needs diagnosis before rejecting an approach.
- Deduplicate exact copies and prune superseded intermediate conversions,
  rendered packets, review previews, replays and caches. Keep one current useful
  representation with its source provenance, rather than every attempted version.
  Use existing deduplication and retention tools; preserve content hashes and
  logical paths where a current consumer needs them. Never hardlink mutable files.
- Prune rejected outputs in general. Retain one only for a concrete current use:
  an assigned repair, an unresolved bug investigation, or an actual failure/DPO
  training input. Record that use and its consumer; "might be useful someday"
  is insufficient. After the use ends, remove it. Keep independently useful
  correct rows from a partly rejected batch; remove bad rows and redundant copies.
- Check storage during the normal monitoring cadence and after large runs or
  conversions. Clean autonomously under this order. Brief the other machine so
  discarded artifacts and obsolete registry entries are removed there by its
  owner too, rather than silently surviving in mirrors. Record the current
  decision and reclaimed totals, without building another archive of discarded
  approaches. This order supersedes older blanket preservation/PRUNE-ASK rules
  for explicitly discarded work.
- These deletions are expressly desired and authorized by the owner (2026-10-11):
  deleting discarded run directories, checkpoints, exports, caches and their
  registry/manifest entries on either machine is wanted. Do it without asking for
  confirmation; list the exact paths in a script first, then delete.

# Natlang-native work (owner rules)

These rules apply to every agent on every machine. Paste the short form into any
subagent brief. Reasons and history are in the linked documents.

**The main rule: think in natlang-native terms.** Before building anything (an
app, a runtime component, a compiler pass, a tool, a policy, a review step),
first ask how it would be a natlang program: typed natural-language functions,
callable folders, services for the outside world, and crisp code only where
exactness or the outside world requires it. Start from that design and justify
every piece of crisp code. Do not start from conventional code and add natlang
afterwards. The rules below are consequences of this one.

- **Policy is natural language; mechanism is crisp.** Any component that decides
  something (which tier serves, what to fuse, what to promote, which task runs,
  how to recover) is a natlang program. This includes natlang's own runtime and
  compiler, not only applications. Crisp code is for the outside world
  (services), the CLI, exact durability (atomic commit), exact verifiers of what
  the stages produce, and plumbing. When a policy is hot, give it one interface
  with a crisp and a natural-language implementation selected by a setting
  (`ts-host/src/runtime/pluggable.ts`); crisp may be the default.
- **Port end to end, unit by unit.** A system "in natlang" has its whole logic as
  natural-language stages decomposed like the real system. For every part,
  record whether it becomes its own function, one instruction, implicit (rare)
  or a crisp helper, and why. Spell algorithms out as numbered steps over named
  data for small models. Write DECOMPOSITION.md first, then implement and merge;
  owner review happens afterwards (queue: `plans/OWNER_REVIEW.md`).
- **Measure model-facing changes live.** Anything the executing model sees
  (prompts, rendering of values, tool text, error and feedback text, type text in
  signatures, removed or added instruction sentences) is sampled on the live
  executor (about 48 samples per variant) before it is adopted. A change that
  lands unmeasured is recorded in `plans/MODEL_FACING_CHANGES.md` as debt and
  measured in the next teacher window.
- **Teach in errors, not prompts.** Never spell out a wrong form in a prompt; it
  primes the model to produce it. Handle wrong forms in the runtime and teach the
  fix in a one-sentence error. Prefer making standard JS idioms work over new
  names; be very circumspect with new model-facing names. Every semantic change
  updates `skills/natlang-authoring` and `skills/natlang-integration` in the same
  push, with recovery advice for each new error.
- **No premature limits.** Termination is structural (no while loops; recursion
  on a smaller argument; `iterateOn` with a measure or a natural-language
  predicate), not budgets or default timeouts. Add a resource limit when a
  problem shows up.
- **Apps:** decide on a snapshot (model calls), then apply a pure bounded commit.
  Effects are data, timers are state, and derived values form a DAG. Stay close
  to JS semantics.
- **One implementation per concept.** Before adding a mechanism, look for the
  existing one (specializer, call store, pluggable, refinement shadow, shared
  hashing in `natlang_neuralese/common`) and extend it. Training lineages share
  trainers, objectives, evaluators and recipes (`extends`/`overrides`); only
  declared backbone-inherent differences are allowed.
- **Training data covers whole trajectories** (prompts, inputs, tool output at
  lower weight). Breadth comes from more natlang domains, not from anchors or
  broad mixing; anchors never cap learning.

# Working practice on shared machines

- On DGX every build, test and probe runs through the memory ledger
  (`scripts/memory_ledger.py run --unit <family>-<id> --budget-gb N -- ...`;
  units run asynchronously, so wait for the unit to finish and read its log).
  Never report a merge as verified when the ledgered tests did not run.
- Ledger priority (owner, 2026-10-11): pass `--priority high` for training and
  GPU direct/E2E runs, `low` for data generation, rendering and check-main, and
  leave smokes/evals at the default `medium`. Claims may be generous: admission
  uses real headroom, and lower-priority resumable units are preempted and
  requeued (plans/MEMORY_ADMISSION.md).
- In the shared checkout, stage only your own paths. Never push while a rebase
  or merge is unfinished. Never stage `node_modules` symlinks.
- Before pushing, run `scripts/check-main.sh --quick` (ratchets, spec links,
  ts-host typecheck; about 10 s). After a change that touches tests or runtime
  code, run the full `scripts/check-main.sh` under the memory ledger (usage in
  its header). Main stays green: a red step is fixed or routed to its owner via
  coord, never skipped.
- When resolving a merge conflict, keep both sides unless you know a side was
  removed on purpose; a dropped section of AGENTS.md or a plan is lost silently.
- Stop processes by systemd unit name or an explicit PID, never by pattern.
- Every agent on these machines is expressly authorized and instructed by the owner (2026-10-12) to stop,
  kill, delete, restart and relaunch any process, systemd unit, ledger job or container, and to remove
  (`docker rm`) or prune containers, without asking: e.g. an orphan left running after its ledger unit stopped,
  a job starving training, a failed or stuck job that needs relaunching with its original command, or a job that
  must move to newer code. Do it, record why, and brief the other machine when it affects shared work. Every
  container an agent starts gets `--name <ledger unit>` and is declared to the ledger
  (`memory_ledger.py run --container <name>`) so its memory is counted and it is stopped with its unit.
- Brief the other machine (`coord.py send`) on every owner decision and every
  shared-code change, with commits and exact semantics. (2026-10-12: that is one
  `decision` per change the other machine must adopt; ordinary commits are not
  announced by notes.)

# Neuralese training foundation

- LFM fresh-start base (owner, 2026-10-12): any new Pop LFM training lineage
  started from base weights uses the pinned full-precision LiquidAI
  LFM2.5-350M QAD source and its QAD clean-head anchor, with a newly measured
  QAD base task reference. See `plans/neuralese/DECISIONS.md` for revision,
  paths and hashes. A resume of an existing lineage restores that lineage's
  exact checkpoint and optimizer state; do not silently substitute QAD weights
  into a checkpoint resume.
- New neuralese training lineages use shared declared recipes under
  `training/neuralese/recipes/`; see `plans/neuralese/TRAINING_RECIPE.md`.
- Token-aligned identity and qualified causal output-state to raw next-token
  embedding distillation precede compression/recurrence. Finishing a warm-up's
  step count is not qualification. Preserve and diagnose failed gates.
- Runtime transport/gradient replay needs separate qualification against those
  exact weights. Legacy marker/RMS checkpoints do not inherit the new foundation
  certificate, and backbone changes require channel requalification.
- Exception, Neuralese direct v1 (owner decision 2026-10-11,
  `plans/neuralese/NEURALESE_DIRECT_V1.md`): a lineage made only of
  `neuralese_direct` stages starts from the base model with no warm-up and needs
  no identity or embedding-distillation stage. Its stage's step-0 identity check
  (NeuraleseProjection equals the soft-token mixture, TokenChoiceProjection
  absent or the identity, logits equal the base) is the identity qualification,
  declared as the recipe's `identity_qualification` and recorded in
  `direct-certificate.json`. Older lineages keep the rule above. Direct-stage
  qualification is the identity check, held CE below base and
  natlang task success (owner decision 2026-10-12): the in-run task-success
  evaluation (16 tasks, one per shape, of the registered held packet
  `natlang-held-task-packet-v3-20261012`, which is v2 with content-preserving
  edit_stream oracles, executed for real against the
  training model at every eval and at the end; `eval/task_cadence.py`) is at
  least the line's base and not below the previous evaluation minus the noise
  measured by repeat runs. The 48-probe
  behaviour gate is report-only: it runs at a drift warn and at the end and is
  recorded, never a stop or a requirement. Response KL warns at 0.5 (which runs
  the gate); it never stops a run or denies qualification merely for changing
  the base distribution (owner correction 2026-10-15). Actual task performance
  remains measured. The Neuralese value is report-only there: its
  requirement is the end-to-end stage's, consumers reading values against
  consumers reading tokens (owner decision 2026-10-11).
