> **Writing or running natlang programs?** None of the rules below apply to you: they govern the research
> project's model training on its own machines. Work in your own directory outside this repository
> (`natlang init my-program`), follow the `AGENTS.md` that `natlang init` writes, and use the natlang skills
> (`natlang skills install`). The language and CLI are documented in `README.md`, `docs/RUNNING_PROGRAMS.md` and
> `spec/SPEC.md`.

# Goal: a strong model, as fast as possible

The one real goal is to move toward a strong model. The backbone today is LFM3-1.7B (port, data curation, codecs:
`plans/neuralese/LFM3_RECIPE_20261028.md`). Every method is a means to that end: RL, SFT, rejection sampling, teacher
and stronger-model data (existing or newly generated) and distillation from it, curriculum and task authoring,
Neuralese, adapters and program tuning. Pick a method on its expected merit. Keep it while paired held-out results show
it helps, and drop it when they show it does not. Earlier strategy statements in plans and DECISIONS ("no additional
teacher generation", "no cross-model dependence", "fill gaps primarily with on-policy RL", a fixed order of additions)
were steps in thinking the problem through. They are not bans.

Agents decide and keep the work moving. Restarts, resets to base, objective or batch changes, redoing work and adopting
or changing a plan are agents' decisions. Decide, record why in `plans/neuralese/DECISIONS.md` and keep going. Raise
with the owner only what is truly theirs. Waiting for an OK on a technical course change is a defect.

- **A finding is not done until it is acted on.** A defect or a better setup becomes the change of the next
  experiment. A recommendation or a list of "proposed experiments" is unfinished work. Changes that are sound from
  first principles (harness and model-facing bug fixes, sampling, reward and loss-weighting fixes) ship as the new
  default with an opt-out and targeted tests. The next held eval shows any regression.
- **Fix the mechanism everywhere it lives.** A defect or design change found in one place is first generalised: what
  shared mechanism (format, protocol, allocator pattern, loss term, serving feature) produced it, and which components
  implement or consume that mechanism (all engines: managed llama-server, in-page wllama, Python native engine, SGLang
  actor; ts-host runtime; learner, held eval, probes, collectors). Before editing or briefing, write that inventory
  (`git grep` the concept, `docs/ENGINE_PARITY.md`), scope the change to all of it, and list each component left out
  with its reason (for engines only a current blocker counts: parity is the goal, see "One serving feature"). A "not applicable" cites the code (file and
  lines) showing the component does not use the mechanism; the supervisor checks those citations before accepting. A subagent brief carries the inventory or makes
  building it the first step; a brief naming only the component where the symptom appeared is incomplete.
- **Orphaned work.** Work whose author shows no activity in the last 24 h is orphaned. The supervising agent reviews it
  and integrates or deliberately discards it.
- **Diagnose surprises immediately.** An unexpected number, failure or regression gets a root-cause investigation in
  the same session.
- **Keep the machines doing useful work** for the experiments in flight (collection, training, held eval, diagnosis).

# Experiments: how results are made trustworthy

- **Bounded controlled experiments decide.** One experiment is one collect -> train -> eval cycle of hours, not days.
  It runs on one pinned commit: actor, harness, scorer, learner and evaluator all run that commit, and the SHA is
  recorded. It changes exactly one thing against its baseline and declares its success and failure criteria before it
  runs. Paired held-out eval judges it. Record each experiment and its result in `plans/neuralese/DECISIONS.md`. The
  latest entries there are the current state of the work.
- **Pinning.** `scripts/pin_checkout.py create COMMIT WORK/pin-<c12>` makes a detached worktree with ts-host rebuilt.
  Launch that copy's scripts with the pin as the ledger `--workdir` and `NATLANG_REPO`/`PYTHONPATH` bound to it.
  `grpo.py` refuses a live checkout unless `--unpinned`. No commits are made in a pin.
  `pin_checkout.py remove DEST` deletes it when the experiment ends.
- **Improvements enter at the next pin, never mid-measurement.** A measured win becomes the baseline of the following
  experiments, with an opt-out switch. Sizing changes at a resume (slots, memory, KV) leave the treatment unchanged.
  Anything that changes what the model sees or how it is trained is a new pin and a re-measured start policy. A
  correctness bug found in a running experiment invalidates it: stop it, fix it, rerun it.
- **A significant unexplained held regression ends the experiment at that eval.** If a paired held result falls below
  the declared failure threshold with p < 0.05 and its cause is unknown, stop the run and start the bug hunt on the
  freed machine. The declared failure rule is a last resort, not permission to keep running. When a learner sentinel
  fires (`grad_spike`, `behaviour_shift`, `reasoning_shift`, a KL jump), diagnose it at that step instead of watching
  it. "Don't touch a pinned experiment" protects valid measurements; a regressing run with a suspected bug is not one.
- **Comparable curves only.** Lower aggregate loss across changing samples, weights or quantization points is not
  progress. Only task outcomes on the pinned held set count. dev16 is a smoke test at most, never a gate.
- **Checks protect results:** task quality, numerics, held-data separation, data quality, recoverability, runtime
  function. Base-relative KL is telemetry, never a stop. Launch checks are nonmutating; never run an administrative
  entry in a run directory first (it can leave records that block the real run).
- **Code identity is provenance.** Every run, checkpoint and actor records its git commit and dirty flag
  (`common/code_provenance.py`). Refuse a wrong policy SHA, model/tokenizer identity or artifact schema. A resume
  restores its run's exact checkpoint, optimizer and RNG state; never substitute base weights into a resume. Outside an
  experiment, services run from `/home/werg/natlang` (Pop: `git pull`).
- **Precision.** Fine-tuning keeps FP32 master weights and FP32 optimizer state and computes in BF16
  (`exp0_sft.load_model`, `lfm3_fast_train`). Pure-BF16 parameters lost LFM3 updates to rounding (APPROACH_REVIEW 1.1).

## Held-out evaluation

- **The held set is held-v3:** `ops/held-v3/manifest.json` under `/home/werg/data/lfm3-rl-dgx-20261029/` (sha
  45471060, 256 tasks; `plans/neuralese/HELD_V3_20261113.md`), the default `grpo.HELD`. It is held-v2 with 14 bad or
  leaking tasks replaced and the same `engine.json`. Never train on held ids or their source-group siblings. Training
  excludes them through `held_eval.exclusion_ids()`, which also covers held-v2 (`also_excludes`). Before relying on
  the set for a new data source, run `held_eval verify` against `ops/held-v3/exposure/collect_exposure.py`.
- **Tooling** (`rl/held_eval.py`): `measure` (resumable, sample-major episodes against `--endpoint` or
  `--checkpoint`), `serve` (the pinned engine as a ledger unit) and `compare A/result.json B/result.json`. The primary
  statistic is the mean per-task pass-rate change with a within-task permutation p and a task-bootstrap CI, timeouts
  counted as failures; McNemar, the task sign test and the ran-out/loop rate stand beside it.
- **What compares.** `compare` pairs only one code commit and scoring series, one sampling law and one attested engine
  (SGLang, vendor projection, FP8 KV), so the start policy is re-measured at each new pin and on each machine. Read
  results at 2-4 samples per task: one-sample differences under ~8 pts are noise. Held-v2 notes and the A/A history:
  ENGINEERING_FINDINGS.md ("Moved from AGENTS.md 2026-11-15") and `ops/held-v2/runs/`.

## Diagnosis: degradation means bug

A worse loss, degraded behaviour or an unexplained numerical gap is a bug until proven otherwise. Don't reject an
approach because our implementation of it failed. Compare against an exact reference, localize the divergence, fix it
and re-measure before judging the approach. When something doesn't make sense, run a wide, varied dragnet: data and
masking, rendering and templates, dtype and upcasts, caches and carried state, kernels, optimizer state and resume,
gradient flow, loss weighting, evaluation code, and differences from a known-good path. Record what was ruled out.

Closing a degradation as "not a bug" requires all of the following:
- The explanation is a mechanism that predicts the observed direction and size. "Noise", "learning dynamics" and
  "drift" are descriptions, not explanations. A consistent direction that the objective does not select points at a
  biased gradient, i.e. a learner bug, until shown otherwise.
- Every dragnet item is marked checked (with evidence), ruled out, or not checked. "Found nothing in scope X" is
  recorded as exactly that, never as "no bug". The entry stays open while any plausible item is unchecked.
- Investigators are briefed with the observations, not a hypothesis. Never give them a menu whose benign answer ends
  the hunt ("bug vs learning dynamics"). Ask them to find the cause and to report what they did not check.
- A second, independently briefed investigator tries to find the bug before a "not a bug" verdict is recorded or
  reported. Until then, report it as "agent X found no bug in scope Y". The supervisor checks every verdict against the
  observations, above all the direction of the change, before relaying it.

Diagnostics take the fastest route to a trustworthy answer:
- Before launch, the brief states the question, each candidate design's wall time (steps x seconds per step x arms)
  and whether it reaches the regime it tests (warmup vs step count, the LR actually reached, sample size). A design
  that cannot answer is not run.
- It also states the expected result and what would change if it came out otherwise. A check whose answer is
  already near-certain (stock library code used the standard way, a property earlier measurements imply) or whose
  outcome changes no decision is not run: answer from the evidence and say so. A question is not a request to measure.
- Choose by time to a trustworthy answer, not by cost: offline statistics on archived rollouts; forward-only
  evaluation of existing weights (scaling an update `base + a*(theta_k - base)` as an LR stand-in, grafting,
  interpolation, sign-randomized nulls); short learner-bench replays; full-size runs. Run the slow method when it is
  likely decisive and the fast one is not; when either may be needed, run both and stop the slow one once the fast one
  answers.
- Start static code analysis as soon as a defect is suspected, in parallel: it needs no GPU and serves as the
  independent second investigation. Feed it each new result.
- One question per ledger unit, the fewest steps and groups, results after each step. Runs already provide the
  per-step behaviour probe, branchable states (`--branch-states`, `--keep-checkpoints-until`, `--resume --resume-state
  FILE --replay-schedule`) and the learner bench; full-size runs answer only what these cannot.

When a symptom appears (slow, failing, poor learning), first check what is already built for it:
`plans/neuralese/FALLOW_REGISTER.md` (built capabilities not in use, with blocker and next action; review it between
experiments), the plans and this file. A qualification bar may not block a ready improvement for more than 48 idle
hours: resolve it, replace it with the standard-practice mechanism, or record a deliberate exclusion.

## Heartbeats and machine worth

A monitoring cycle checks that the pinned experiments run correctly and the machines are healthy, and it fixes
breakage. It does not re-plan; plans change between experiments, from their results. Each cycle also asks what the
machines are worth. Is any running job's result already decided or uninformative? Is anything (a diagnosis, the next
run) waiting on memory or GPU that a job holds? Then stop that job. Every run leaves >= 15 GB of DGX ledger headroom for
diagnostics and probes. Stopping a run also stops its Pop eval actor. The heartbeat prompt names the experiments
actually in flight; fix a stale list in the same cycle.

# Current approach (revisable; each item stands on its rationale)

**State.** The start policy is the untouched base (5ff07638): no w5 checkpoint beat it on held-out. The method is
plain GRPO (`scripts/neuralese_data/grpo.py train`). grpo-v2-run1 and run2 both regressed significantly on held-out; the 2026-11-15
hunt points at fresh AdamW taking near-full-LR noise steps at LR 1e-5 with ~24 groups a step, and an LR sweep chooses
grpo-v3's step size. The latest DECISIONS.md entries are authoritative; `plans/neuralese/APPROACH_REVIEW_20261111.md`
is the plan this approach came from.

- **Standard tooling, minimal crisp code** (an HF training loop with `lfm3_fast_train`, the SGLang server). Earlier
  controllers, custody schemas and transitions consumed the effort and confounded measurements. Build infrastructure
  only when the next experiment needs it.
- **Per-call sequences; the model's own outputs are the targets.** Each model call is one (prompt, completion)
  sequence. Prompts, inputs, tool output and harness feedback are zero-weight context. The useful learning is in
  actions and answers; prompt CE on long repeated prompts dominated loss without teaching execution.
- **On-policy collection with exact identity.** Collect from the checkpoint being updated with its identity, exact
  prompt, sampled ids and probabilities: ratios and KL are only correct on what was actually sampled. RL uses the full
  explanatory prompt and complete tool schemas so idiomatic behaviour is reachable; never put held answers or
  trajectories in it.
- **Data from any source that helps:** teacher and stronger-model demonstrations (existing or newly generated),
  distillation, self-distillation of verified successes, SFT, rejection-sampling fine-tunes, RL. Choose by expected
  held gain per machine hour and measure it like anything else. Measure repetition and diversity in what is actually
  trained: in 2026-11 scripted templates carried ~78% of SFT loss and RL tasks were 98% one wrapper generator
  (`plans/neuralese/RECIPE_REVIEW_20261103/`); step counts on such a mixture are not capability.
- **Task supply.** Subagents author task programs with exact oracles, qualified by reference, negative and alias
  replay, continuously and across rotating domains, seeded from existing agentic and reasoning datasets (never held
  inputs) and invented ones. Measure repetition against the pool before admitting a batch
  (`scripts/rl_tasks/measure_repetition.py`, `simulate_selection.py`); never pad scarce behaviours by repetition.
- **Breadth from more natlang domains** (chat and writing included) rather than anchor KL or broad mixing; forgetting
  general ability outside natlang is acceptable.
- **Additions enter one at a time with a paired measurement.** Off by default today: Neuralese-declaring task variants
  (`--neuralese-variants`), whole-graph Neuralese gradients and use bonus, the JOLT+ teacher and privileged twin,
  memetic guidance programs, interleaved SFT and QAT. An addition stacked on a baseline that does not improve cannot be
  measured, so the base loop comes first; the order after that is a judgement call (Neuralese task variants are cheap:
  they are data).
- **Never upweight a few tokens without per-token evidence**, and check an update's direction against its gradient
  before blaming the objective (a boundary weight of 32 and stale Muon momentum each once dominated updates).

## Neuralese, meta-learning and adaptation (product intent)

- **Neuralese compaction is the intended runtime design:** learned operators compact and summarize large values
  (parameter listings, inputs, tool outputs, documents in context) and carry compressed state between model runs. Exact
  KV/prefix reuse is a separate cache, not compaction. Training data includes sources rewritten into a Neuralese
  bottleneck whose consumer must use it for the task outcome.
- **Learned operators start from text:** each is initialized from its natural-language prompt in the real rendered
  call, and every Neuralese span from an input-specific, target-relevant token sequence (gold/checked summaries, else a
  recorded mechanical summary). A span that ablation cannot tell from shuffled state is a defect.
- **Whole-graph Neuralese gradients** go through the real producer/consumer dependencies; never turn an in-episode
  producer into a detached input. GRPO today trains consumer calls with the imported state fixed. Task success comes
  before Neuralese use.
- **Memetic learning and soft skills are a core user capability:** guidance learned from several invocations of a
  program, judged on separate development and confirmation invocations, conditioning the real executing model with
  text/tokenizer provenance kept. The long-term meta-learning direction is the backbone running a natlang improvement
  program, trained on held-world task gains.
- **Program tuning and LoRA mixtures** (`plans/TUNING_OPERATIONS_HANDOFF_20261026.md`,
  `plans/PROGRAM_TUNING_RUNS_20261026.md`): mixture labels come from actual fits in the named bank, and outcomes are
  measured through the composed adaptation. Their targets are worth collecting once the backbone shows held gains.

## The GRPO loop as it runs on main

`grpo.py --help` documents every flag; history and measured numbers are in `plans/neuralese/ENGINEERING_FINDINGS.md`
("Moved from AGENTS.md 2026-11-15"). Defaults as of 2026-11-15:

- **Shape.** One ledger unit holds the learner and its child SGLang actor (held engine pin, FP8 KV). `--slots` (96)
  work-stealing episodes run continuously over shuffled task groups (`--group 8`) from `POOLS`, excluding held-v3/v2
  tasks and siblings, `ops/pool-errata` rows and `ext_codebase-*` families (`WORK/pool.json` records the pools). Each
  step trains on the groups finished since the last one (`--min-groups 24` to `--max-groups 48`, `--max-staleness 2`).
  A share of 1 - `--explore` (0.4) comes from tasks with pass rate in `--pass-band 0.1 0.9`, which a task's latest
  scored group updates. The rest are unmeasured tasks.
- **Episodes and reward.** 40 model calls per episode (a budget stop scores 0); the 3600 s wall clock is an unscored
  safety net; a signal-killed episode abandons its group. `--reward graded` (file partial credit) or `binary`; selection
  pass rates and held eval stay binary. Sampling is temperature 1, top-p 0.95 everywhere (`common/sampling.py`).
- **Loss and optimizer.** `-min(rho, 2) * A * logp + 0.02 * k2(start || policy)` on per-call sequences, `A`
  group-normalized, `--loss-norm token`, policy term under the learner's nucleus, `--tail-logp-floor -7` for A > 0,
  sampled decisions at most 8 token units, Neuralese consumer calls with the imported state as fixed context.
  FP32-master AdamW, `--lr 1e-5` (under review), `--warmup 2`, `--clip 1.0`, LR scaled by groups-with-spread / 12
  (`lr_data_scale`). The KL reference runs the policy's own graph from a reference-weights file.
- **Compute.** `lfm3_fast_train`; layers beyond `--activation-gb 12` recompute; shared prompt prefixes (>= 256 tokens)
  run once as a prefix tree; no new episodes start while the learner runs (`--learner-phase-slots 0`; the GB10
  time-slices the two, so overlap buys nothing: GRPO_STEP_THROUGHPUT_20261117.md).
- **Sync and state.** Each step's BF16 weights go to the actor's safetensors on /dev/shm and SGLang reloads them in
  place; the front attests the served sha. Native checkpoints every 2 steps (last 2 plus held-evaluated ones kept), the
  FP32 state every 4 to `/mnt/external/natlang-grpo-v1`. `--branch-states N` keeps the last N FP32 states (~21 GB
  each), `--keep-checkpoints-until N` every early checkpoint; `--resume [--resume-state FILE]`. Between optimizer
  steps the AdamW moments (13.6 GB) live in `WORK/adam-moments.offload` (`--keep-moments-resident` opts out).
- **Held evals.** By default on the training actor (collection pauses): the start policy at 4 samples, then a
  checkpoint at 2 samples every 4 h, compared with step 0 in `WORK/held/step-N/`. `--held-remote pop` keeps the DGX
  training: the start policy is measured once on Pop's engine, then the newest checkpoint back to back
  (`pop_rl_actor.py serve-checkpoint`, `--held-remote-slots 14`, FP8 KV and 16 running on Pop, `held-pop.log`).
- **Sentinels** (`train.jsonl`): `grad_spike` (> 5x the run's median grad_norm), `reasoning_shift`, k3 `kl_mean` with
  `ref_dlogp_max`, rho statistics, and the behaviour probe after every step (`behaviour_probe.py`,
  `ops/grpo-v2/behaviour-probe-v1.json`: logp change, entropy and `</think>` hazard on 48 start-policy calls, ~42 s,
  flagged as `behaviour_shift`). `cuda_peak_by_phase_gb` gives each phase's memory peak.
- **Learner bench** (minutes): `grpo.py train --replay ROLLOUTS --replay-max-group-tokens 16000 --min-groups 6
  --max-groups 6 --max-steps 6 --keep-checkpoints-until 6`; `--advantage-control shuffle|zero` are reward-free
  controls; `--replay-schedule` replays a run's exact batches. The 2026-11-15 hunt's runs are in `ops/grpo-v2/hunt/`.
- **Memory.** The ledger budget bounds GPU and host together. The learner's CUDA peak grows with chain length and
  `--activation-gb` does not bound it (read `cuda_peak_by_phase_gb`); add the actor (`--sglang-kv-gb` + front) and
  0.3-0.7 GB per slot. run2's last sizing: 16 slots, actor KV 6 GB, 32 running, 86 GB budget.

# Machines

- **DGX (GB10, 128 GB unified memory)** trains: the learner, the SGLang actor and diagnostics. **Pop** (RTX 4060
  8 GB, 14 GB host RAM) runs held evals of pinned runs first. When none is queued, it runs forward-only probes and
  checks of the 1.7B BF16 model (not the learner); a held eval preempts them. The DGX supervisor drives Pop over ssh.
  Cloud (CoreWeave) is unavailable. Both machines use `/home/werg/natlang` as the canonical checkout.
- **Memory ledger.** On the DGX every build, test and probe runs through
  `scripts/memory_ledger.py run --unit <family>-<id> --budget-gb N -- ...`. Units are asynchronous: wait and read the
  log. Priority is `--priority high` for training/GPU runs, `low` for data generation and check-main, and the default
  `medium` for smokes and evals. Never report tests as passed when the ledgered run did not happen. `memory_ledger.py
  adopt` changes accounting only, not a unit's `MemoryMax`. Restart a stopped transient service under a fresh unit
  name. Ledger units do not inherit the caller's shell environment. Containers get `--name <ledger unit>` and are
  declared with `memory_ledger.py run --container <name>`.
- **GB10 unified memory.** GPU memory is system RAM, and a unit's `MemoryMax` also bounds its CUDA allocations.
  `CUDA_ERROR_OUT_OF_MEMORY` from `cuDevicePrimaryCtxRetain` with tens of GB free means fragmentation (no free order
  >= 8 blocks in `/proc/buddyinfo`). Keep the `natlang-gb10-compactor` user service (`scripts/systemd/`) running.
  `common/cuda_context.py` `ensure_context()` and `start_sglang` retry after a compaction nudge. A device-to-host copy
  into fresh pageable memory runs at 0.05-0.1 GB/s, so prefault or pin host buffers before copying checkpoints.
- **SGLang actor.** Run one engine per GPU and give it all slots (two engines time-slice the GB10). Serve with
  `serve_native_checkpoint.py --engine sglang` defaults: 128 running, `--sglang-kv-gb 12`, the pool size derived from
  memory free at launch. Ledger budget = `--sglang-kv-gb` + 24 GB. Do not pass `--sglang-mem-fraction`. Decode is
  memory-bandwidth bound: a standalone actor wants >= 64 collector slots (~250 MB host each; episodes share node
  processes, 8 per process, `NATLANG_COLLECTOR_SHARED=0` opts out). Details are in
  `plans/neuralese/ACTOR_THROUGHPUT_20261112.md`.
- **Environments.** Tests and data use `/home/werg/natlang/.venv-neuralese/bin/python` with
  `PYTHONPATH=/home/werg/natlang/training/neuralese:/home/werg/natlang`. Node builds use the Node24 bin under
  `/home/werg/.local/share/natlang-dev/`. `os.pidfd_open` guards need host `/usr/bin/python3`. Standalone tests get a
  disposable `NATLANG_HASH_CACHE` and `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` (`check-main.sh` sets both).
- **Processes.** Any agent may stop, kill, restart, relaunch or remove any process, unit, ledger job or container
  without asking; record why. Stop them by systemd unit name or explicit PID, never by pattern.
- Do not add or remove hardlinks to source, manifest or index files while live readers stream them (ctime changes
  invalidate strict reader checks).

# Shared checkout, code and data

- **Work lands on main** in `/home/werg/natlang`. Commit small steps promptly and push often to `origin/main` with
  ordinary git (pull, merge, rebase). Stage only your own paths and commit them in the same command
  (`git commit -- PATHS`), because a bare `git commit` takes what others staged. Never amend, reset or rebase shared main
  while another strand can commit. Reconcile divergent history with a normal merge, never a force-push. Never stash or
  discard changes you didn't make, never push mid-rebase or mid-merge, and never stage `node_modules` symlinks. When
  resolving a conflict, keep both sides unless one was removed on purpose. Worktrees are short-lived: merge their
  commits promptly and remove them. Don't copy source trees between machines.
- **Push gate.** Run `scripts/check-main.sh --quick` before pushing, and the full `check-main.sh` under the ledger
  after runtime or test changes. Main stays green.
- **Lean codebase, economical tests.** Remove legacy infrastructure outside the current direction (code, tests, docs,
  recipes, registry entries, compatibility shims and fallbacks; there are no deployed users to stay compatible with).
  Git history is the archive. Keep source/training data, protected held data and live recovery dependencies. Run the
  targeted tests for a change. Fix every failure you see whatever its origin; don't rerun suites on a base commit to
  assign blame.
- **Data.** Publish data additions in `training/neuralese_corpora.json` with immutable SHA-256 manifests under
  `training/corpus-manifests/`, and transfer them with `scripts/sync_training_corpora.py`. Keep conversion, admission,
  staging and actual learner exposure separate and explicit. Every dataset and model license is acceptable: record
  license provenance as a fact, never hold data for license review.
- **Storage hygiene.** Free space continuously. Delete discarded runs and approaches (checkpoints, optimizer states,
  exports, frozen runtime copies, diagnostic outputs, obsolete recipes and registry entries) without asking: list the
  exact paths in a script first, then delete and verify. Keep the active lineage and its parents, the latest verified
  recovery state, active diagnostic parents (e.g. a hunt's branch states), the episodes an experiment trains on,
  reusable source/training data and held data. Before deleting, check live process references, current recipes and
  checkpoint history. When unsure, move to `/mnt/external` instead of deleting. Deduplicate exact copies, and never
  hardlink mutable files. Keep latency-critical data (scorer DBs, training inputs, active checkpoints) on NVMe.
- Engineering findings and incident history: `plans/neuralese/ENGINEERING_FINDINGS.md`. Read the relevant section before
  changing that code.

# Natlang-native work (product rules)

These rules apply to every agent on every machine for the product: the natlang language, runtime, compiler, tools and
apps. Paste the short form into any subagent brief. The training and RL stack is the exception: it uses standard
tooling and minimal crisp code.

**The main rule: think in natlang-native terms.** Before building anything in the product (an app, a runtime
component, a compiler pass, a tool, a policy, a review step), first ask how it would be a natlang program: typed
natural-language functions, callable folders, services for the outside world, and crisp code only where exactness or
the outside world requires it. Start from that design and justify every piece of crisp code. Do not start from
conventional code and add natlang afterwards. The rules below follow from this one.

- **Policy is natural language; mechanism is crisp.** Any product component that decides something (which tier serves,
  what to fuse, what to promote, which task runs, how to recover) is a natlang program, including natlang's own runtime
  and compiler. Crisp code is for the outside world (services), the CLI, exact durability (atomic commit), exact
  verifiers of what the stages produce, and plumbing. When a policy is hot, give it one interface with a crisp and a
  natural-language implementation selected by a setting (`ts-host/src/runtime/pluggable.ts`); crisp may be the default.
- **Port end to end, unit by unit.** A system "in natlang" has its whole logic as natural-language stages decomposed
  like the real system. For every part, record whether it becomes its own function, one instruction, implicit (rare) or
  a crisp helper, and why. Spell algorithms out as numbered steps over named data for small models. Write
  DECOMPOSITION.md first, then implement and merge; owner review happens afterwards (queue: `plans/OWNER_REVIEW.md`).
- **Decide model-facing changes on first principles.** Fix bugs and structural problems in what the model sees directly,
  never gated on an A/B: lost or erased actions, history that differs from what was sampled, needless prompt rewrites,
  limits below the model's capacity. Record each in `plans/MODEL_FACING_CHANGES.md`; held-out evaluation of the next
  checkpoint shows any regression.
- **Teach in errors, not prompts.** Never spell out a wrong form in a prompt; it primes the model to produce it. Handle
  wrong forms in the runtime and teach the fix in a one-sentence error. Prefer making standard JS idioms work over new
  names, and be very circumspect with new model-facing names. Every user-facing semantic change updates the covering
  reference of `skills/natlang-authoring` or `skills/natlang-integration` in the same push, with recovery advice for
  each new error. Native-training and training-application notes go to `docs/NATIVE_RUNTIME_NOTES.md` and
  `docs/SEMANTIC_APPLICATION_NOTES.md`, not the skills (`skills/README.md`).
- **One serving feature, every engine.** The managed llama-server, the in-page wllama build, the Python native engine
  and the SGLang actor serve one protocol, and parity is the goal. A serving change (request field, cache, scoring,
  sampling, capture, state handoff) lands on every engine in the same work stream, fork changes and releases included.
  A wider parity gap found along the way is closed too, not labelled "not applicable". A row in
  `docs/ENGINE_PARITY.md` records only a gap blocked right now, with the blocker named; it is never a deferral. Fork pins
  move only through `scripts/update_llama_runtime.mjs --fork`, which vendors the in-page build from the same release;
  never hand-build or hand-edit `ts-host/vendor/wllama`.
- **No premature limits.** Termination is structural (no while loops; recursion on a smaller argument; `iterateOn`
  with a measure or a natural-language predicate), not budgets or default timeouts. Add a resource limit when a problem
  shows up.
- **Apps:** decide on a snapshot (model calls), then apply a pure bounded commit. Effects are data, timers are state,
  and derived values form a DAG. Stay close to JS semantics.
- **One implementation per concept.** Before adding a mechanism, look for the existing one (specializer, call store,
  pluggable, refinement shadow, shared hashing in `natlang_neuralese/common`) and extend it.
