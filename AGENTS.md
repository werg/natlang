## Corrected-mix regression: LR collapse and marker-weight spam (engineering, 2026-11-05)

Dev16 task_success fell 12/16 (bf95) -> 10/16 (step 93) -> 0/16 (129) -> 2/16 (140) -> 1/16 (204). Two causes:
1. Between 93 and 129, as the update-budget LR multiplier rose 0.4 -> 1.0 (backbone about 2e-5) on the corrected mix,
   the model fell into a "repeat the failing action, ignore the error feedback" attractor. On probe contexts it
   repeated the failing action 12/12 times, against 0/12 for bf95 and step 93. This predates the marker weight and
   prefix v4.
2. `--boundary-weight 32` (producer marker min_weight) concentrated 1-9% of each update's objective on 2-26 marker
   tokens. By step 204 the model answered `return_result` with a native_state (P(open marker) 0.71, against 0
   before), whatever the declared result type, and looped until the budget ran out.
Rules: do not pass `--boundary-weight` above a few tokens without per-boundary dev evidence. Raise backbone LR only
when paired dev16 at the new rate holds. The production restart (ops/bf95-lowlr-1105) uses backbone 4e-6, row LR
2e-4, the pre-s3 CE policy and 8-update windows, each evaluated on Pop. Steps 93-217 of corrected-mix/s3/solo are
diagnostic only, not parents. Evidence: decisions.log 2026-11-05 and the diagnosis probes it cites.

## BF16 SFT aliases exact raw prefixes physically (engineering, 2026-11-04)

Prefix arithmetic v4 (`native-prefix-physical-alias-v4-bf16-fanin`) supersedes
the 2026-11-02 occurrence-row rule below. Corrected-mix updates are dominated
by chain graphs. These re-render their whole history per invocation, so 82% of
projection rows were duplicate raw prefixes. v3 deduplicated only forward linear
primals, and every duplicate still ran attention/conv and full backward. v4
runs each exact raw prefix row once; its cotangent receives the FP32-reduced
fan-in of all consumers. Supervision, normalization, readout rows, exposure and
capacity geometry follow the actual aliased graph. Qualification is a declared
rounding-order transition on real complete graphs
(`plans/neuralese/native-prefix-arithmetic-qualification-20261104.json`):
- forward loss is equal;
- FP32 aliased vs unaliased has global gradient relative L2 <= 1e-4;
- BF16 gradients stay within the transition bound (relative L2 0.014-0.016,
  cosine >= 0.99987);
- BF16 is as close to the FP32 reference as v3 (0.066/0.082 vs 0.066/0.083).
It is not a same-arithmetic 1e-4 match; do not claim one. GB10 per-graph
fwd+bwd was 2.5-3.7x faster. Resumes adopt v4 through
`prefix_sharing_arithmetic_history`.

## Promoted packed groups retain forward-only source reads (engineering, 2026-11-03)

Focus reader identity must bind the first missing visit, matching one-time
promotion. Marking every future repetition as focused makes rows with different
earned counts share an invalid cursor (actual full-schedule failure2026-11-03).
A focus request can also promote whole mixed-source packed batches whose
constituent rows cross a reader's next index. Keep the exact batches, graph content, quotas
and order, but merge their minimally overlapping read windows and sort reads
within that buffer. Source cursors remain strictly monotone and distinct; do not
rescan compressed sources or suppress the guard. The default merged-window bound
is1048576 forward tokens, with an already declared larger single window retained
whole. A plan exceeding its bound must be corrected before SGD. Qualify the actual
pending schedule, not only small fixtures, before claiming the cutover works.

## Recipe review: scripted templates dominate the mixture (owner request, 2026-11-03)

Independent sampled audits (`plans/neuralese/RECIPE_REVIEW_20261103/`) found
that the `port_record_calls` wrapper and big2 template communication carry about 78%
of quota-aware loss. Genuine natlang teaching receives about 6%; multi-node Neuralese about 2%;
`iterateOn`/reducers/decisions/skills are rare. Only marker rows and the embedding
adapter train. RL tasks are 98% one wrapper generator, and the selector yields no
outcome spread. Tuning and metalearning are not recipe stages. Step counts and
loss on this mixture are not capability progress. Correct the mixture (one row
per wrapper task, template visit caps, withdrawals, diverse focus and admitted
diverse sources) and cut over at the next window. Add interleaved RL with a
spread-seeking selector and declared tuning/metalearner stages. Promote by paired
protected-development outcomes. Preserve earned exposure and historical custody.

Owner follow-up (2026-11-03): start training on the corrected mixture as soon as
its caps and quality checks pass, and fill measured per-construct gaps while it
trains. Never pad scarce behaviours by repetition. Fill gaps primarily with RL:
small-model (Haiku) subagents author task programs and exact oracles, prompted
for real depth and diversity, qualified by reference/negative/alias replay.
On-policy RL is preferred over more teacher demonstrations; verified successful
rollouts may self-distill into SFT. Teacher codebase generation is secondary.
Authoring is continuous, not a one-off batch: keep finding new impulses for
tasks and Natlang programs. Existing agentic and knowledge-reasoning datasets
seed program designs, call arguments and targets (inspiration with real oracles;
never protected/held inputs). Datasets are one source, not a limit: authors also
invent original programs, domains and problem types continuously. Rotate domains and problem types; measure
repetition against the existing pool before admitting each batch.

Meta-learning is the memetic path (owner correction, 2026-11-03; supersedes the same-day
"bank meta-learning main line", which was a miscommunication). The backbone, executing a natlang
improvement program, studies a program's invocations and failures and authors or revises its
guidance, skills, prompts and adaptation choices (including which LoRA bank components/mixtures to fit,
via the crisp bank fit as a service). It is trained by on-policy RL on "improve program P" episodes,
rewarded by held-world task gains measured by exact oracles, with JOLT+ privileged context, so
meta-improvement is a backbone skill that flows into general training. It is a declared recipe stage
interleaved with SFT/RL, not a side project. Reuse the bank machinery (PCA-initialized bank, coefficient
fits, held-family task-outcome evaluation) as mechanism and baselines. Successful improvements and the
improved programs' verified runs distil into SFT/RL with exact actor provenance. Promote by held-family
task outcomes, dev first and confirmation once.

## Interleaved RL is a declared stage; selector /2 seeks outcome spread (engineering, 2026-11-03)

Recipe kind `native_interleaved_rl` (`rl/interleave.py`) requires the joint SFT
stage and binds the RL task source, selector, round budget and JOLT+ loss. An RL
controller with `interleaved_rl_stage` alternates `rl_rounds_per_window` RL rounds
with one existing mandatory SFT window to that stage's `refresh_after_updates`
boundary; SFT quotas/exposure stay in the carried frames. Prepare the control with
`scripts/neuralese_data/prepare_interleaved_rl.py` under a runtime frozen from main.
Task selector `natlang.rl-task-curriculum/2` starts new families at their median
level, shifts practice by outcome-spread fraction (<0.3 moves one level per round)
and floors rounds at 16 tasks x 8 samples. Configs binding /1 adopt it through a
recorded controller transition; sealed selections keep their recorded policy.
Measure task-supply repetition with `scripts/rl_tasks/measure_repetition.py`.

Construct shares are enforced at selection time, never by dropping pool candidates
(supervisor, 2026-11-03: reaching the 30% effect/state bar by shrinking the pool cost
1,653 distinct programs). The pool assembler keeps every row that survives the generator
and family caps and tags rows whose oracle verifies a world `curriculum.rl.mix_class =
effect_state`; selector /2 carries `class_quota` (default effect_state 0.34) and fills at
least that share of every round from the tagged rows after the rotating breadth cursor
(rows never duplicated; untagged pools select as before; continuations adopt the field
through the recorded transition `task-curriculum-class-quota-20261103`). Measure the
sampled mixture with `scripts/rl_tasks/simulate_selection.py`, not the pool's own share.

## Selector /2 decays per-row visits (engineering, 2026-11-04)

A stable mixed outcome keeps the spread the selector seeks, so it re-picked the same rows (r7 pool latent simulation,
1,500 x 32: one row 372 visits, 245 rows 20+). The controller records each scored task's actual `visits`/`visit_rounds`;
`visit_penalty` (1.0 level per doubling above the pool's mean visits) and `visit_cap` 2 per `visit_cap_window` 50 rounds
rank over-visited rows behind fresh ones (a ranking, never a drop). Breadth, `class_quota`, frontiers and the 16x8 floor are
unchanged. Adopted through `task-curriculum-visit-decay-20261104`; sealed selections keep their rows and policy. Fresh rows
are mixed less often than re-picked mixed ones, so the sampled spread fraction falls (0.69 -> 0.54) while staying far above
`spread_target`: that is the price of not memorising single tasks.

## Neuralese compaction is the intended runtime design (owner, 2026-11-03)

Owner intent: Neuralese compacts and summarizes large values throughout the
natlang runtime, including parameter/argument listings, large inputs, tool and
service outputs, collections and documents rendered into model context. It also
compacts context between model runs, carrying learned compressed state rather than
full text. Training data includes bgkit-derived (and other) sources transformed
into a Neuralese bottleneck: a Neuralese-prompted summarizer/extractor producer
returns Neuralese to a parent lambda/consumer that must use it for the task outcome.
The compaction and summarization operators are directly learned operators,
trained parameters rather than only prose prompts. Exact KV/prefix reuse is a
separate performance cache, not compaction. Track remaining gaps in the fallow
register until each is live in runtime, data and training.

Directly learned operators (owner, 2026-11-03) start from text: initialize each
operator (compaction, summarization/extraction, guidance, skill and other
prompt-embedding operators) from its natural-language prompt, embedded in the
real rendered call, then let it morph through direct learning on sufficient
dedicated data. Each operator needs its own adequate training data and measured
use, never a nominal parameter with a handful of rows.

Neuralese span basis (owner, 2026-11-03): every Neuralese span, soft state or
learned operator output starts from a good token sequence: input-specific,
target-relevant content the consumer needs, never a constant cue, instruction or
placeholder shared across inputs. Use gold/checked summaries or supporting spans.
Where none exist, use a deterministic mechanical summary (structure outline,
question-selected extractive spans, exact identifiers and values, length scaled to
input), recording seed provenance. Then morph through end-to-end consumer-outcome
learning with annealed text supervision. A span whose content is the same for
every input, or that ablation cannot distinguish from shuffled state, is a defect.

## Built capabilities must not lie fallow (owner, 2026-11-03)

The SGLang actor was built and integrated, then left unused for days behind a
bespoke exactness bar while RL rollouts ran about 100x too slowly on the native engine.
When the slowness surfaced, the intended solution was not raised. Standard
practice (fast-engine sampling, trainer-recomputed logprobs, truncated
importance correction, divergence rejection) was never considered. Rules:
- Keep `plans/neuralese/FALLOW_REGISTER.md` current: every built capability that
  is not live, with its blocker, impact, next action and an active owner. Review it
  every supervision cycle; adding a capability without wiring it in creates a
  register entry, not a silent default-off flag.
- When any symptom appears (slow, failing, poor learning), first check what we
  already built for it (register, plans, AGENTS.md) before diagnosing from scratch,
  and state that option explicitly.
- A qualification bar may not block an otherwise-ready improvement indefinitely.
  If a blocker has had no active work for 48 hours, either resolve it, replace the
  bar with the standard-practice mechanism that keeps correctness safe, or record a
  deliberate exclusion with its reason. Strictness that prevents use is not safety.
- Deployment counts, not commits: a fix is done when the live job runs it.

## Platform and archived findings (owner, 2026-11-09)

CoreWeave quota was exhausted on all four accounts on 2026-11-06 and is not expected back: production RL runs on the
DGX (learner + rollouts) with Pop as a second rollout actor and the dev16 evaluator. Engineering findings from
2026-10-15 to 2026-11-02 (checkpoint/custody/capacity/numerics incidents, the Neuralese shared-state architecture,
reward-policy details, generation diversity rules) live in `plans/neuralese/ENGINEERING_FINDINGS.md`. Read the
relevant section before changing that code.

# Main goal: keep moving forward (owner, 2026-10-11)

## Act on current facts; nothing continues just because it started (owner, 2026-10-29)

Owner: don't leave potentially valuable work untouched, problems undiagnosed, or processes running merely because
they already started when the facts have changed or something better exists. Concretely:

- **A finding is not done until it is acted on.** Whoever finds a defect or a better setup affecting live work
  implements the fix and cuts over immediately: the change applies to the very next optimizer step, update,
  request or job that can take it, and running work is interrupted (preserving its state) when the change
  matters more than finishing the in-flight unit. A recommendation alone is an unfinished task. "Already
  running", "from the next round", "after the next milestone/merge/boundary" and "the other agent owns it" are
  not reasons to keep spending compute on a known-worse setup. Settings are read when they are used (e.g.
  learner parameters at each update start, not at round start), so an edit takes effect without waiting.
  Fresh runs start on the corrected setup (owner, 2026-10-30).
- **Re-justify running work every monitoring cycle.** For each running learner, job, conversion or eval, check
  that it is still the best use of its resources given today's facts: objective, data, code, newer findings.
  If not, stop or cut over, and record why.
- **Verify, don't assume.** Claims about state, ownership or other agents must come from a check made now:
  processes, coord activity, files, logs. Work whose author shows no activity (no process and no coord message in
  the last 24 h) is orphaned. The supervising agent owns it: review it and integrate or deliberately discard it.
  Don't address it to absent owners.
- **Diagnose surprises immediately.** An unexpected number, failure or regression gets a root-cause investigation
  in the same session, not a note for later.

## Work lands on main, not in worktrees (owner, 2026-10-29)

Shared-main commit hygiene: create new commits scoped to your own paths. Never
amend, reset or rebase shared main while another strand can commit; HEAD may
have changed since your last read. Reconcile divergent published history with
a normal merge, preserving both strands, rather than force-pushing.

No work may be lost or abandoned in worktrees: where possible, do work directly on main in the canonical checkout
(stage only your own paths). A worktree is a short-lived tool, not a home for work: merge its commits to main
promptly and remove it once its commits are on main and nothing live or recorded depends on it. Runs, tests and
jobs execute the code on main. On 2026-10-29, 69 worktrees held commits not on main, mostly from agents no longer running
(`/home/werg/data/dgxops/worktrees-with-unmerged-work-20261029.txt`). The supervisor reviews them and
integrates or discards each one with a recorded reason.

## Agents decide; nothing waits on the owner (owner, 2026-10-29)

Operational and technical decisions such as restarting learners, restarting a
lineage from base, changing the batch or objective, or redoing work are **not**
the owner's decisions. Every agent has the autonomy to do the work the best
possible way and must not block anything on the owner. Decide, record why,
brief the other agents through coordination, and keep going. Agents still
coordinate with each other on shared resources; that is not a reason to wait
for the owner.

## Early on-policy RL and explanatory prompts (owner, 2026-10-29)

Prepare early JOLT+ interleaved with useful SFT instead of requiring completion
of all SFT before outcome learning can begin. Collect from the actual trained
checkpoint being updated, preserving actor/checkpoint identity, sampled token
probabilities, complete execution graphs and unfinished SFT obligations. Historical
teacher continuations and a different checkpoint's attempts are not current-policy
rollouts. Actual graphs receive whole-graph updates under the standing
producer/consumer custody rules.

Early RL must use the full explanatory prompt and complete tool schemas so correct
idiomatic behavior is accessible. Include valid examples of awaited per-item `nl`,
exact sorting/filtering/counting in code, `iterateOn` with stopping/effect checks,
direct use of inputs/services without shadow bindings, typed concise child answers,
and `finish:true`. Bind the actual prompt, tokenizer/template and runtime in both
collection and replay, including the privileged JOLT+ context. Never put protected
answers or held trajectories in that guidance. Compact prompts are a later explicit
robustness target, not the initial RL default.

Sample diversity and loss weighting are separate from cohort service fairness.
Measure program/template/world repetition within and across actual packed groups;
missing identifiers are unknown coverage, not zero repetition. Preserve complete
graphs, exact row quotas and optimizer/RNG/exposure when improving order, with
bounded source buffering and qualified packed geometry. Do not repeatedly rescan
compressed sources to obtain random order.

Engineering finding, 2026-10-29: physical source-file boundaries are not task
families. The previous source-cohort CE weighting changed objective mass when
identical rows were rechunked and exceeded its declared scale cap after
renormalization. New objectives use explicit source/row-bound logical groups and
bounded mean-one normalization with a true cap, through recorded objective
continuations. Preserve exact historical objective/exposure history. Coarse
communication/coding/ordinary mechanism groups do not establish semantic task
taxonomy or the realized coding-token mixture. Lower aggregate loss across
changing samples, weights or QAT points is not a comparable learning curve.

## Program adaptation follows a trained backbone (owner, 2026-10-28 evening)

Collect program-tuning executions and metalearner targets once the backbone has
trained meaningfully and provides an applicable base. Do not spend hours fitting
the early step-750 checkpoint while joint SFT is unfinished. Prepare the full
tuning/metalearner machinery and existing oracle/source supply in parallel;
select the executing base from actual later SFT/RL progress and development task
outcomes. Preserve early diagnostic traces without calling them useful adapter
updates or capability lift. This supersedes running the early LFM3 tuning lane
alongside the initial backbone stage; the full intended order remains joint
SFT, whole-graph JOLT+ RL, program adaptation, then metalearner training.

## LFM3 focus: one joint SFT stage, no cross-model dependence, metalearner (owner, 2026-10-28)

The LFM3 line trains natlang execution and native Neuralese together in one
joint SFT stage; do not train natlang capability first and defer Neuralese.
Strike all dependence on and interaction with another model: no cross-model
VAE/shared-dialect stage and no waiting for Mellum. The objective after the
backbone stages is everything the tuning system needs (program tunes, native
soft states, LoRA mixtures/bases), the training corpus for the learned
optimizer (metalearner), and training the metalearner itself. No additional
teacher generation: prioritize the parts of existing data that fit, and use
exact oracles with RL/JOLT+ self-teaching where demonstrations are missing.
Optimize throughput for the target sandboxes and DGX in time-boxed passes, start
real training soon, and keep optimizing while training. Parallelize all
preparation (builds, data conversion, sandbox images, syncs). This supersedes
earlier cross-model VAE, Mellum-readiness and teacher-generation instructions
for the LFM3 line.

## Improvements ship on by default (owner, 2026-10-28)

Functionality that could move the needle is enabled by default in new
recipes, trainers, RL learners, serving and tools as soon as it works. Do not
implement an improvement and then leave it behind an opt-in flag, a
"pilot later" note or a report-only mode. Opt-out switches are fine; opt-in
defaults are not. "Default" means everything going forward, including
continuations and resumes of existing lineages: they adopt the improvement
through an explicit recorded transition at that step. Only replays or
re-verification of already-completed work, and a unit of work already in
flight (e.g. an RL round with a bound journal, which the next round then
upgrades), keep their recorded behaviour. When a default could be harmful,
measure and fix it; do not hide it behind a flag.

## Lean codebase and economical testing (owner, 2026-10-28)

Always eliminate legacy infrastructure outside the core direction: code, tests,
docs, recipes and registry entries. Git history is the archive, so keep a lean
codebase with clean abstractions. Preserve source/training data, protected held
data and live recovery dependencies; never keep code "just in case".

Test economically: run the targeted tests for a change, merge, and debug problems
when they appear. Do not rerun suites on a base commit to tell pre-existing
failures from new ones. Fix every failure you see whatever its origin; when a
failing test only covers legacy infrastructure, remove the test and that code.

## Individual-program tuning (owner, 2026-10-26)

Operate the tuning lane in `plans/TUNING_OPERATIONS_HANDOFF_20261026.md` and
`plans/PROGRAM_TUNING_RUNS_20261026.md` alongside production training. Measure
how far each program improves through hierarchical adapters, native soft states,
teacher SFT and whole-graph RL. Continue saved adapter/soft-state optimizer
histories when improving the recipe. Bind round journals to the actual recipe,
student base, initial parent and execution limits; a changed recipe must not
silently reuse completed rounds. Check reply/action learning separately from
loss on long repeated prompts, and adjust prompt supervision when it dominates.

Keep related invocation worlds on one split, select on development and confirm
on untouched confirmation worlds. Report actual base, teacher, SFT and RL task
outcomes; saved artifacts and zero-variance rollout groups are not capability
lift or policy updates. Program fits do not advance production lineage exposure.
Distill verified training-world executions through ordinary source admission to
the production recipe, preserving exact actor and split custody.

Source appends follow the declared recipe wiring (engineering finding,
2026-10-27): roles sharing one named input, such as the full training source
and its foundation source, move together. Preserve separately named predecessor
and held inputs exactly, resolve the complete future recipe, and bind its actual
resolved roles into the seed command. An already-applied source correction is
historical state, not an executable operation to repeat against an appended
dataset. Verify its committed proof in the earned full checkpoint, retain its
history and exact dependencies, and use the actual corrected train/held sources
as the next append's predecessors when retiring the consumed stage argument.

Checkpoint I/O is part of GPU supervision. Prefer the existing qualified
immutable background snapshots where measured memory permits, and track actual
commits and independent exports separately from enqueue acknowledgements. Keep
cold fixture preparation and cache offload from starving an earned checkpoint
write; reuse verified current sealed test fixtures and use internal disposable
scratch when its measured headroom permits.


## Learned LoRA mixtures and bases (owner, 2026-10-27)

Make direct component-mixture and native soft-state fitting an operational part
of program adaptation. Free hierarchical LoRA fits can supply bank growth and
behavior targets, but they do not establish that mixture training or execution
has occurred. Mixture labels must come from an actual fit in the named bank,
with task outcomes measured through the composed adaptation.

Train the shared LoRA bases and component bank alongside the mixture writer,
coefficient head and native guidance in the shared learning stage. Explicitly
enable `train_basis` and `train_bank` in its operational recipe rather than
inheriting their disabled code defaults. Individual program label fits may pin a
bank snapshot; after a shared bank update, publish its actual new snapshot and
refit/rebind affected coefficient labels. Verify actual basis/component changes
and downstream behavior, preserve optimizer history, and keep the backbone learners moving. Do not describe target extraction alone as training
the learned mixture system.

## Core memetic learning and legacy removal (owner, 2026-10-23)

Memetic learning and soft skills are a core training and user capability. Learn
stored native Neuralese guidance from multiple specific invocations of the same
program, measure actual task outcomes on distinct development and confirmation
invocations, and keep iterating the guidance and training recipe.
Keep related worlds on one side of the split. Do not claim capability lift from
stored artifacts, admission, source checks or gradient tests alone.

Use the current positional KV and hybrid-state architecture in training and
inference. Stored guidance must condition the real executing model. Training
exports retain the actual producer and every consumer import at their recorded
positions, with gradients through those dependencies. Retain text and tokenizer
provenance for re-encoding when the executing checkpoint changes; reuse compatible
state efficiently across repeated and concurrent calls. Guidance authoring keeps
exact actor provenance. The model being trained authors its own memetic guidance,
helped by JOLT+ privileged context (the same actor and checkpoint with verified
case facts); external teachers supply codebases and verified trajectories, not
guidance (owner, 2026-10-31).

The system has no deployed-user backward-compatibility requirement. Remove legacy
memetic adapters and flat soft-block fallbacks, migrate their current callers and
supported functionality to the native path, and remove obsolete public entrypoints,
docs, tests and discarded-run registry entries. A legacy adapter hiding missing
functionality is work to implement, not a reason to retain the fallback. Preserve
usable source data and protected evaluation data. Historical hashes and manifests
alone do not justify retaining a current legacy entrypoint; actual frozen recovery
and historical-reader dependencies retain their own exact sources as needed.

Apply this default across the system: do not keep obsolete adapters or compatibility
shims merely because an older API, test or recipe mentions them. Inventory current
callers, implement their useful behavior in the native architecture, migrate those
callers, and remove the obsolete entrypoints. Delegate concrete architectural gaps
for implementation and independently review the integrated result. An explicit
unsupported error is useful during migration but is not completion of missing
functionality. Keep actual live recovery dependencies until their replacement is
verified.


## Whole-graph RL and LFM curriculum (owner, 2026-10-16)

When an execution has a Neuralese graph, policy updates cover the entire actual
graph: preserve sampled and forced producer turns, consumer imports, exact
occurrence bindings and downstream outcomes, with gradients through the real
producer/consumer dependencies. Never turn an in-episode producer into a detached
external input or train a selected consumer/turn in place of the whole graph.
Missing or unsupported graph capture is repaired before that episode is used.

Owner amendment (2026-10-31): actively instill Neuralese use. Mine broad task
sources (with outcome oracles) for RL and self-distillation, prompt for Neuralese
where it helps (tasks and prompt variants declaring `Neuralese<...>` child results,
per-document/per-item views, fan-in to a parent), and reward its use: the task
outcome stays the primary reward, plus a bonus only for correct episodes whose
graph actually passes Neuralese state between calls (never on failures).
Self-distillation prefers successful Neuralese-using rollouts. Don't over-focus
on Neuralese to the detriment of task performance.

Prepare an executable LFM RL task curriculum, starting with tractable verified
natlang execution and growing into tool use, multi-step work and communication.
Use actual task outcomes to adapt difficulty; keep protected evaluation separate.
This does not relax
comprehensive training on the entire evolving production recipe.

## Adaptive production training (standing owner instructions, 2026-10-15)

- **Best possible models as soon as possible is the objective. We are not
  doing science.** Do not optimize operations for clean, frozen, reproducible
  runs or controlled comparisons at the expense of useful training. Existing
  runs are work in progress, not experiments that must remain untouched.
- **Constantly improve the recipe and keep going.** Monitor the models and
  their processes, investigate concrete impediments, and greedily exploit useful
  data, fixes and insights as soon as they are available. Do not let repeated
  probes, qualification windows, packaging or handoffs become substitutes for
  production training. Use focused checks that resolve a real correctness or
  operational concern, then return the GPUs to useful training promptly.
- **Comprehensive means the entire production recipe on the models and all
  usable available data.** Stages may differ, but progress through them; a short
  warmup, partial epoch, isolated stage or smoke run is not the goal. Track
  actual stage progress and data exposure, and resolve what prevents remaining
  stages or data from being used. Do not claim that downloaded or registered
  data has been trained on.
- **Adapt live.** Add newly usable data promptly rather than waiting for a
  distant clean run or an unrelated conversion batch. Stop/reload/resume
  immediately when needed; retain optimizer/RNG state when continuing
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
- **Qualify current foundations on an explicit shared-stage request.** Retain
  genuine completed producer/consumer exposure. Repeated synchronous zero-update
  qualification must not replace full-source SGD between ordinary SFT/DPO/RL
  continuations. Publish a genuine qualified zero-update boundary through the
  exact normal mechanism without epoch, source-visit or optimizer-update credit;
  then continue training. Preserve unfinished full-recipe obligations.
- **Keep launch checks nonmutating and necessary** (engineering lesson, 2026-10-20).
  Run normal metadata/argument preflight once; let the real entry perform its
  required state, numerical and capacity checks before updates. Do not execute
  an entire administrative entry in the canonical production directory first:
  it can leave strict-write launch records that block the real learner. Put
  diagnostics in an isolated namespace, preserve concrete failure evidence,
  and avoid repeated whole-entry barriers while useful training is stopped.
- **Avoid redundant supervision barriers** (engineering practice, 2026-10-20).
  Routine data append payloads that change only counts, paths and authenticated
  generations may proceed through the already reviewed mechanism with an
  independent operator review and the normal schema, source/held, custody,
  geometry and compare-and-swap checks. They do not need another round trip to
  the root agent. Review mechanism changes separately. Record the actual
  reviewer and standing authorization; never claim an individual review that
  did not occur. Preserve exact exposure and continue useful training.
- **Bring in deferred improvements** (owner, 2026-10-16). Keep a current
  inventory of deferred data, performance, training and inference work; actively
  integrate each ready improvement. An old handover, incomplete source census,
  clean-run boundary or another model's unrelated qualification is not a reason
  to leave useful work disabled. Preserve real numerical, task-quality and
  foundation prerequisites, and resolve their concrete blockers.
  Independent model improvements may advance independently; propagate shared
  fixes and retain exact optimizer/RNG/exposure through adaptive continuations.
- **Share code as much as possible and distribute all generalizable insights
  across all models.** A useful fix, optimization, data improvement or recipe
  insight on one line must promptly be considered and propagated to the other
  lines and machines. Extend the shared trainers, loaders, objectives,
  evaluators and recipe inheritance instead of creating per-model forks.
  Keep only necessary architecture-specific behavior/settings separate, state
  why they differ, and communicate shared decisions through coordination.
- **Verify the effective control code after loaders.** A fix in the installed
  package can be overwritten by a sealed controller or publisher source closure.
  Rebase every control layer onto shared fixes, preserve required operational
  behavior, update mechanism identities, and test after the complete loader has
  run. A passing default-module test does not prove that the production
  controller executes the fix. This applies to both model lines.
- **Rebind capacity declarations after runtime changes** (engineering finding,
  2026-10-27). Even a checkpoint-control-only backport changes the exact Python
  inventory. Update the affected recipe and seed memory requests to the actual
  runtime while retaining source, objective, geometry and reference checks; let
  the real restored entry perform qualification before updates. Retain failed
  operation records and reserve an unused operation namespace when continuing
  from the latest earned full state. Do not silently skip comprehensive SFT
  because a resolved failure remains memoized under an unchanged signature.
- **Keep recovered work ahead of historical operations.** Restarting the full
  mandatory schedule after an external optimizer continuation must reserve an
  unused operation namespace in the same lineage. Never let a completed bridge
  result from an older attempt replace the newer earned parent. Verify the
  actual checkpoint and cursor, and observe the real optimizer child advancing;
  a live coordinator or a launched request alone is not resumed training.
- **Reuse actual controller custody across continuations** (engineering finding,
  2026-10-21). Pass the coordinator's continuously held canonical lease descriptor
  through worker threads into completed/runtime configuration; opening the same
  lock again can self-contend. A normal resident publication changes bootstrap
  metadata: verify and refresh the actual adopted configuration rather than
  discarding its data or weakening custody checks. Preserve concrete configuration
  tracebacks and diagnose a blocked retry signature. Exercise this path after the
  complete sealed controller/publisher loaders; a previously shared guard fix can
  still be missing from their frozen source closures.
- **Publish completed resident training before adopting its next stage**
  (engineering finding, 2026-10-22). A supervised stage can finish after several
  live data appends while the coordinator still holds its bootstrap recipe.
  Use the real completed request and result through the normal publication
  helper to bind the earned checkpoint, actual resident configuration and next
  cursor. Updating bootstrap metadata alone does not publish the adopted recipe.
  Preserve optimizer/RNG/exposure and verify the next real optimizer child.

Historical reader replay (engineering finding, 2026-10-22): recipe inputs can
include the exact classification reader implementation. A changed reader is not
a byte-identical relocation. Replay historical completed adaptations and drained
handoffs using the package that actually prepared their metadata, with complete
code inventory and bound recipe checks before and after isolated CPU replay.
Both old and future plans in a completed adaptation were enumerated by that
adaptation's implementation; its old numerical package is not automatically the
original metadata reader. Preserve strict source/held, lineage and exposure
checks. Include committed resident histories and both effective sealed control
closures in validation before cutover. A live scheduler holding the canonical
lease can still be blocked: monitor its actual configuration_errors and retry
signature alongside real optimizer progress. When a normal publication advances
the earned parent, rebind a completed invocation to that actual state through
the existing continuation mechanism rather than reusing an obsolete invocation.

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

Owner, 2026-11-09: the coord inbox is retired. No Pop agent runs; the DGX supervisor drives Pop directly over ssh
(the RL actor and dev16 evals). Do not check `coord.py inbox` in monitoring cycles. The coordination rules below
apply again only if a second concurrent agent is started.

- Use `/home/werg/natlang` as the canonical development checkout on DGX and Pop.
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
  Reconcile coverage against every entry of the exact upstream inventory, with
  explicit member/index bindings; sequential directory labels are not a coverage
  proof. Keep conversion, admission, staging and actual learner exposure separate.
- Licenses (owner, 2026-10-10): every dataset and model license is acceptable,
  including the most restrictive. Include them; record license provenance as a
  fact, but never omit or hold data for license review, and do not ask the owner.

# Storage hygiene and discarded work (standing owner order, 2026-10-11)

Owner clarification (2026-10-16): delete pretty much everything except what is
necessary to carry on. This includes remote W&B transfer/export storage. Keep
the latest independently verified recovery state, explicitly active diagnostic
or handoff parents, useful source/training data and protected held data. Prune
superseded checkpoint exports continuously after replacement durability is
verified; do not accumulate every periodic full checkpoint. Write the exact
deletion list first and verify removal. Never delete an entire mixed data and
checkpoint run merely because most of its bytes are obsolete checkpoints.

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
- **Supervise the model's own outputs** (owner, 2026-10-28; retires the
  2026-10-07 whole-trajectory supervision rule). Assistant actions and typed
  answers are the targets; prompts, instructions, inputs, tool/service output
  and harness feedback are context with zero loss weight. Breadth comes from
  more natlang domains, not from anchors or broad mixing; anchors never cap
  learning.

# Working practice on shared machines

- Frozen runtime launches use that runtime as their actual ledger `--workdir`,
  as well as binding `NATLANG_REPO` and `PYTHONPATH`. Python's initial cwd can
  otherwise import the canonical checkout's regular `scripts` package ahead
  of the frozen helper closure. On2026-11-01 the collector correctly refused
  this mixed process-custody helper before any rollout; changing only runtime
  environment metadata was insufficient. Preserve the refusal and fix launch
  ownership rather than weakening exact helper checks.
- DGX launch environments are explicit: project/data tests use
  `/home/werg/natlang/.venv-neuralese/bin/python`; CoreWeave SDK orchestration
  uses `/home/werg/.local/share/natlang/cw-venv/bin/python`. Pass
  `PYTHONPATH=/home/werg/natlang/training/neuralese:/home/werg/natlang` inside
  the ledger command for project/data entrypoints. Node builds use the Node24
  bin directory under `/home/werg/.local/share/natlang-dev/`. Do not assume a
  caller's shell environment is forwarded into a ledger unit. Verify these
  environments before launching substantial jobs; repeated import/launcher
  failures are a shared operational bug to fix, not useful qualification.
- A ledger unit's explicit relative executable path resolves in its admitted
  `--workdir`. Bare executable names retain PATH lookup. Use an absolute script
  path for an intended checkout when inspecting or recovering older launches:
  older `systemd-run` invocation paths resolved against the caller's checkout
  before changing directory. Verify the tested checkout's path and revision in
  the actual check log; a worktree claim alone is not evidence of its tests.
- For DGX administrative ownership guards requiring `os.pidfd_open`, use the
  host `/usr/bin/python3`; the CoreWeave SDK venv lacks that API. Invoke SDK
  operations through its venv as a subprocess when both capabilities are needed.
  Compile the assembled remote Python body before transport, not only the
  outer launcher. Honor each receipt's binding schema: compare path/SHA fields
  explicitly and check declared byte counts against the actual file; a legacy
  two-field binding is not unequal evidence merely because another includes bytes.
- Reuse authenticated live parser/index caches for CPU preparation and isolated
  consumer checks. Cache isolation is not an objective. Verify exact source,
  tokenizer, role, parser-policy and entry identities before reuse; preserve
  cold work only when no valid completed cache exists. Locate artifacts through
  registries and bound manifests; do not leave unbounded content searches or
  directory crawls running across the external disk after the answer is known.
- On DGX every build, test and probe runs through the memory ledger
  (`scripts/memory_ledger.py run --unit <family>-<id> --budget-gb N -- ...`;
  units run asynchronously, so wait for the unit to finish and read its log).
  Never report a merge as verified when the ledgered tests did not run.
- `memory_ledger.py adopt` changes accounting; it does not resize an existing
  unit's kernel `MemoryMax`. When a lightweight waiter becomes a larger job,
  admit its real budget and explicitly set and verify the actual cgroup limit
  before allocating its working set, or launch a fresh correctly budgeted unit.
  Disposable RAM-backed readback files count toward that host budget too.
- Bound dirty file-cache memory during large artifact copies (engineering finding,
  2026-10-22): a small Python heap does not bound a ledger unit's memory. Flush
  and synchronize bounded write windows, release clean cache when appropriate,
  and resume an interrupted copy only after validating its actual prefix. Verify
  full content before replacing a logical path; preserve source and live readers.
- Do not add or remove hardlinks to source, manifest or index files while live
  native readers validate or stream them. Link-count changes alter ctime even
  when bytes remain unchanged and correctly invalidate strict reader checks.
  Copy for live backup staging or drain the affected readers first; keep the
  strict checks. Immutable checkpoint pinning has its own checked-stat protocol.
- Restart transient ledger services through the ledger with a fresh unit name
  (engineering finding, 2026-10-22). A stopped transient unit may already have
  been collected, so `systemctl start` can fail even after a clean stop. Preserve
  the existing state/lease namespace, verify the new actual process and update
  supervision bindings; never treat scheduling the wrapper as resumed work.
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
- Give standalone tests a disposable, unit-specific `NATLANG_HASH_CACHE`.
  `check-main.sh` supplies one by default. Test fixture identities accumulated
  millions of entries in the shared production cache and caused real metadata
  job memory failures; cache isolation preserves the same exact hash verifiers.
- Set `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` for standalone small numerical
  tests, as `check-main.sh` already does, unless a test needs explicit parallelism.
  On2026-11-02 the same35 update/norm checks took75.41s and691 CPU-seconds
  with default pools versus9.61s and10.93 CPU-seconds with one thread. This
  changes test resource use, not production trainer threading or arithmetic.
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

# Neuralese training foundation

The LFM3 line defines the training foundation (owner, 2026-10-30; replaces the
LFM2.5 QAD base, warm-up/identity-qualification and Neuralese direct v1 rules):
see `plans/neuralese/LFM3_RECIPE_20261028.md` and
`plans/neuralese/LFM3_COMMITTED_DECISIONS_20261028.md`. New lineages use the
shared declared recipes under `training/neuralese/recipes/`. A resume restores
its lineage's exact checkpoint and optimizer/RNG/exposure state; never substitute
base weights into a resume. Qualification is actual task behaviour, numerics and
recoverable state, not similarity to the base.
