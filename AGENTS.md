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
  not acknowledge the owner's reader cursor (inspect `coord.py whoami`). Forward
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
- Licenses (owner, 2026-10-10): every dataset and model license is acceptable,
  including the most restrictive. Include them; record license provenance as a
  fact, but never omit or hold data for license review, and do not ask the owner.

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
  data for small models. Write DECOMPOSITION.md first and get the owner's review
  before restructuring (queue: `plans/OWNER_REVIEW.md`).
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
- In the shared checkout, stage only your paths and commit in the same command.
  Push from a worktree when the checkout holds others' work. Never push while a
  rebase or merge is unfinished. Never stage `node_modules` symlinks.
- Before pushing, run `scripts/check-main.sh --quick` (ratchets, spec links,
  ts-host typecheck; about 10 s). After a change that touches tests or runtime
  code, run the full `scripts/check-main.sh` under the memory ledger (usage in
  its header). Main stays green: a red step is fixed or routed to its owner via
  coord, never skipped.
- When resolving a merge conflict, keep both sides unless you know a side was
  removed on purpose; a dropped section of AGENTS.md or a plan is lost silently.
- Stop processes by systemd unit name or an explicit PID, never by pattern.
- Brief the other machine (`coord.py send`) on every owner decision and every
  shared-code change, with commits and exact semantics.

# Neuralese training foundation

- New neuralese training lineages use shared declared recipes under
  `training/neuralese/recipes/`; see `plans/neuralese/TRAINING_RECIPE.md`.
- Token-aligned identity and qualified causal output-state to raw next-token
  embedding distillation precede compression/recurrence. Finishing a warm-up's
  step count is not qualification. Preserve and diagnose failed gates.
- Runtime transport/gradient replay needs separate qualification against those
  exact weights. Legacy marker/RMS checkpoints do not inherit the new foundation
  certificate, and backbone changes require channel requalification.
