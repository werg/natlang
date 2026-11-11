# Approach review: why the LFM3 line is not improving, and what to do instead (2026-11-11)

Peer review, not anchored on the current plan. Everything below was checked against the files named; numbers are
quoted from them. `W` = `/home/werg/data/lfm3-rl-dgx-20261029/ops/bf95-lowlr-1105/w5`,
`T` = `training/neuralese/natlang_neuralese`. One small experiment was run (ledger unit `review-ckpt-delta-2`,
CPU, read-only on the checkpoints); nothing under `W` was modified.

## Summary

The lineage has never learned anything measurable, and the reason is not subtle: **the backbone weights cannot
move at the configured precision and learning rate.** All weights, optimizer state and the EMA are BF16 with
stochastic-rounding writes (owner rule "no full precision"), and the backbone LR has been 1.56e-6 (quartered by an
update-budget multiplier from a 2e-5 commitment, via 4e-6). At that LR the intended per-element update is about
1/200 of a BF16 ulp, so each update is realised as a sparse set of random one-ulp flips. I compared the production
checkpoints at steps 92, 99, 102, 110, 118, 126 and 134: successive stage deltas have cosine 0.00 +/- 0.03 with each
other, 80-88% of changed elements moved by exactly one ulp, and the total 92->134 movement equals the random-walk
sum of the stage movements to within 0.5% on all nine matrices tested. Forty-two updates of RL and SFT left the
network with zero coherent weight change. The held-out decline (52 -> 39) is either rounding-noise damage or eval
drift between code versions (both are live); either way **no update since step 92 carried a learning signal into the
weights the forward pass uses.** The team's own diagnosis (`ops/diag-sanity-1109/REPORT.md`) measured this on
2026-11-09 (noise energy 20-85x signal per update; 100x LR learns cleanly) and responded with a 4x LR change inside
the same noise regime instead of fixing the precision.

Around that core defect sits a loop that is roughly 100x too slow to learn even if the arithmetic were right
(16 tasks x 8 samples per 5-6 h round, 269 generated tokens/s across 20 collector slots, 165 supervised tokens/s in
the learner, SFT at 3.4% of one epoch after two weeks), an evaluation that cannot see a 10-point change (dev16,
n=16, reference re-measured 12 then 10; held evals run under four different code commits; the base model has never
been measured on the held set), and 133k lines of bespoke training infrastructure receiving 3,310 commits in 21
days. The owner rules that produce this are named in section 5.

## 1. Main reasons we are not improving (ranked)

### 1.1 Updates are BF16 rounding noise: the backbone has not moved coherently since step 92 (decisive)

Evidence:

- Config: `W/config.json` `trainer_parameters.learning_rate` 1.56e-6 (seed) -> 6.24e-6 from round 3;
  `native_ordinary_precision: bf16-stochastic-rounding-v1`, `dtype: bfloat16`; recipe
  `plans/neuralese/LFM3_RECIPE_20261028.md` "No full precision (owner): parameters and optimizer state are BF16
  everywhere ... with stochastic-rounding writes". Committed LR was 2e-5
  (`LFM3_COMMITTED_DECISIONS_20261028.md` #2); production ran 4e-6 (decisions.log 2026-11-05 02:59) and the
  update-budget multiplier quartered that to 1.56e-6 (`FALLOW_REGISTER.md` "Why these went fallow" #3). The SFT
  windows ran at the same 1.56e-6 (`W/sft-window-00000{0,1}/learner/train.jsonl`, `backbone_lr`).
- The team's own measurement (`ops/diag-sanity-1109/REPORT.md` test 4): intended update rms per element 2.6-3.0e-7
  against a median ulp of 6.1e-5 (1/200 ulp); visible BF16 weights change by 1-1.7% of elements per update with
  noise energy 70-86x the signal; AdamW-routed norms 100-1700x; `q_norm`/`k_norm` cannot move at all. Test 2: the
  directed in-sample effect at 1x LR is below the common-mode drift; at 100x it is clean and sign-correct.
- My checkpoint experiment (ledger `review-ckpt-delta-2`, script in the scratchpad, output reproduced here).
  Nine matrices (conv in/out_proj and FFN w1 in layers 0, 11, 23), pairs of consecutive lineage checkpoints:

  | stage | kind | fraction of elements changed | relative delta | changes that are exactly 1 ulp | cos with previous stage delta |
  |---|---|---|---|---|---|
  | 92->99 | RL round 0 (7 updates) | 0.03-0.05 | 2.2-3.2e-4 | 0.85-0.87 | - |
  | 99->102 | RL round 1 (3) | 0.015-0.025 | 1.4-2.1e-4 | 0.88-0.89 | +0.001 |
  | 102->110 | SFT window 0 (8) | 0.03-0.05 | 2.1-3.2e-4 | 0.85-0.87 | -0.014..-0.030 |
  | 110->118 | RL round 2 (9) | 0.04-0.06 | 2.6-3.9e-4 | 0.84-0.86 | -0.004..-0.012 |
  | 118->126 | RL round 3 at 4x LR (7) | 0.09-0.15 | 4.7-7.3e-4 | 0.79-0.82 | +0.008..+0.016 |
  | 126->134 | SFT window 1 (8) | 0.02-0.04 | 2.1-3.1e-4 | 0.86-0.88 | -0.006..-0.035 |
  | 92->134 | whole w5 | 0.15-0.23 | 6.6e-4..1.0e-3 | | |

  For every matrix the observed 92->134 relative delta equals sqrt(sum of squared stage deltas) to within 0.5%
  (e.g. L0.w1: predicted 1.01e-3, observed 1.01e-3; L23.conv.in_proj: 7.81e-4 vs 7.77e-4). Independent random steps
  add exactly that way; a coherent drift would exceed it. The 4x-LR round moved three times as many elements but is
  no more correlated with its neighbours (cos +0.01). Conclusion: across RL and SFT alike the realised weight change
  has no coherent component at this resolution. The Kahan compensation buffer may hold an intended update, but the
  forward pass never sees it (it needs ~200 coherent updates per element to surface one ulp; the lineage has 42).
- Learned-row telemetry agrees: `W/round-000000/learner/train.jsonl` records `gradient_update_cosine` of -0.069,
  +0.009, +0.059 for the Neuralese rows at LR 7.8e-5, i.e. even the rows that "train" move almost orthogonally to
  their gradient (stale momentum through Muon, see 2.3).

Consequence: every held/dev16 comparison between w5 checkpoints compares noise realisations of one model. The
held-out trend 52 -> 47 -> 44 -> 39 is consistent with (a) cumulative one-ulp noise on ~20% of elements and/or (b)
evaluation drift across four code versions (1.3); the step-92 baseline eval now running under current code
(`heldeval-driver-92.service`, 92/128 samples done) separates them. Neither reading involves learning.

### 1.2 Even with correct arithmetic the loop is ~100x too slow to learn

- Per RL round: 16 tasks x 8 samples + 16 teacher attempts = 144 rollouts (`W/round-*/score.json`), of which 5-14
  groups have any outcome spread (`groups_with_spread`: 10, 5, 14, 8, 8). About 60-100 informative episodes per
  round. The learner then runs 7-9 updates of 16 episodes each (train.jsonl `objective_details.episodes`).
- Wall time: collection 2.6 h at 16 workers (decisions.log 2026-11-11 15:40; was 5.7-6.9 h at 4), learner 152 min
  (AGENTS.md "RL learner wall time"). Round 4 generated 1.86M tokens over 4,180 model calls in 1.9 h: **269 generated
  tokens/s aggregate** for a 1.7B model on a GB10 plus an RTX 4060 (`W/round-000004/generations.jsonl`,
  `rewards.jsonl`). A plain SGLang/vLLM server for a 1.7B model does several thousand tokens/s at this concurrency.
  The harness re-renders the whole history on each of ~24-30 calls per episode, so prefill dominates.
- Learner: 2,600-3,000 forward tokens/s, 165 supervised tokens/s (train.jsonl `supervised_tokens_per_second`):
  5.2M forward tokens per update for 331k sampled tokens (16x), because whole-graph replay re-renders every call's
  history and packs a stop-gradient privileged twin (2x) that has zero weight with `beta_s = 0`
  (`diag-sanity-1109` "Other observations"). Plus 11 min policy-capacity backward passes, 7 min init, 5 min
  objective preparation, 3 min save per round.
- SFT: window 0 trained 8 updates on 27k-95k supervised tokens each; window 1 on 24k-445k
  (`sft-window-*/learner/train.jsonl`). The joint SFT stage stands at 4,049 of 119,399 required graph visits
  (3.4%) (`sft-window-000001/learner/report.json` `active_stage_completion`). At 8 updates per 6-hour cycle one
  epoch takes months.
- Net: about 500 informative RL episodes and about 1M supervised SFT tokens per day. GRPO on a 1.7B model normally
  consumes that in minutes.

### 1.3 The evaluation cannot see what the training does

- dev16 is n=16. The reference checkpoint (step 67, `71c3eb45`) scored 12/16 and then 10/16 on re-measurement
  (`ops/pop-behavior-eval/comparisons/raw-*-vs-71c3eb45594d.json`, reference `task_success` 12 vs 10). Every w5
  checkpoint from step 75 to 134 lies in 8-13/16; the "STOP" verdicts, rollbacks and keep decisions of 11-10/11-11
  (decisions.log 347-352) were taken on this noise. 30 dev16 evaluations were spent this way.
- The held set (64 x 2) is the right idea but its four measurements ran under four code commits (e5d1bec, bd1e539,
  83ff767, f6c8b49; `W/held-eval/policy-*/result.json` `code.git_commit`), with a context-overflow fix (a7fce3f5b)
  landing between 110 and 118 (`unresolved` 1 -> 6 -> 0 -> 0) and 18 commits touching eval/actor paths. Forty of
  the 64 tasks have identical outcomes across all four policies (0/2 or 2/2); the decline sits in 15 tasks worse vs
  5 better. This is not a controlled comparison.
- **The base model has never been measured on the held set.** The only base comparison in the record is
  decisions.log 2026-10-31 02:41: "base 32/51 vs SFT step608 17/51, McNemar p=0.006 -> REGRESSED even in its own
  training format", and 03:08: RL vs base "4 vs 5 at 9 paired, lineages barely moved from fresh-init". After that,
  all comparisons were lineage-internal. Nobody knows whether step 92 is better than LFM3-1.7B out of the box.
- "Frugal eval" (owner, 10-30) plus a 16-case gate means the project has no instrument that could detect success.

### 1.4 Infrastructure complexity consumes the effort and confounds every measurement

- `T/` is 133,105 lines of Python (train/ 40k, rl/ 30k, tune/ 11.5k); `rl/production.py` alone is 2,927 lines;
  ts-host 74.8k lines of TypeScript. 3,310 commits since 2026-10-21 (1.75M insertions, 318k deletions; 426 on
  10-28 alone). `W/` holds config versions v67-v70 from one day plus six `config.json.pre-*` backups and four
  restart-freeze logs from one afternoon (11-09 15:09-17:57). Round 3's learner was launched five times
  (r25-r29) for capacity, objective-preparation and inventory refusals (decisions.log 349-353) before it trained.
- The mechanisms that fail are mostly custody/transition machinery, not learning code: recorded transitions,
  held-qualification scopes, capacity proofs, reservation locks, checkpoint offload floors, policy-source
  continuations, pre-collection staleness guards. `FALLOW_REGISTER.md` "Why these went fallow" is an honest
  description of a system whose knobs nobody can hold still.
- Because the controller "runs from the checkout" and "improvements ship on by default", the live job's code changes
  between rounds and between evaluations. Rounds 0-4 record four different `code.git_commit` values
  (`score.json`), one of them dirty. With this policy no two measurements are of the same system.

### 1.5 The recipe stacks unvalidated novelties with no positive control

Each of these is standard-practice-adjacent but bespoke, and none has been shown to work on this model in
isolation: Muon with BF16 stochastic-rounding weights and momentum; JOLT+ with privileged twins (`beta_s` now 0, so
the student KL is off and the twin is dead weight); whole-graph replay with custody rules; truncated IS at k = 128
over actor/learner mismatch; score-centred advantages; Neuralese marker rows and a rank-8 embedding adapter as the
only Neuralese operators; a Neuralese bonus (use rate 5-13% of episodes, `score.json` `neuralese.*`); expert
iteration into SFT windows with per-family shares; memetic guidance programs; selector /2 with visit decay; weight
EMA in BF16-SR; pre-collection with staleness import. The one measurement that would anchor all of this, "does a
plain SFT/RFT update at a working LR improve held-out success", has not been run. The committed-decisions file
forbids it: "YOLO run with our best foot forward, instead of budgeting time for lots of experiments."

## 2. Concrete bugs found, with proof

**2.1 BF16 stochastic-rounding weights at LR 1.56e-6 / 6.24e-6 produce a random walk, not training.**
Proof: section 1.1 table (nine production matrices, random-walk sum matches observed whole-lineage delta to 0.5%,
stage-to-stage cosine ~0, 80-88% one-ulp changes); `diag-sanity-1109` tests 2 and 4. The fix is standard: FP32
master weights (1.7B x 4 B = 6.8 GB, plus 13.6 GB of Adam state; trivial on a 121 GB machine) with a BF16 compute
copy, or at minimum an LR at which the update exceeds an ulp. The owner rule "no full precision" is the root cause
and should be withdrawn for this model size.

**2.2 The LR was quartered twice by machinery, not by decision.** Committed 2e-5 -> production 4e-6 (after the
11-05 collapse) -> 1.56e-6 (update-budget multiplier, `FALLOW_REGISTER` #1/#3; both train.jsonl families show
`backbone_lr 1.56e-06`). The 11-05 collapse at 2e-5 happened on the template-dominated corrected mix with
`--boundary-weight 32` (AGENTS.md "Corrected-mix regression") and is therefore not evidence that 2e-5 is too high
for clean data. Both SFT windows in w5 also ran at 1.56e-6: eight updates on 27k-445k supervised tokens at that LR
are no-ops by construction.

**2.3 Stale SFT momentum steered every w5 RL round (found by the other agent, commit 1d724c8f8).** Muon normalises
the update, so with momentum 0.95 carrying SFT gradients ~1000x the RL gradient, every RL update applied an
SFT-direction step. Confirmed by `gradient_update_cosine` -0.07..+0.06 on the learned rows in round 0
(`W/round-000000/learner/train.jsonl`). Fixed on main; no w5 round has run with the fix.

**2.4 Round 0 trained with uncapped importance ratios.** `W/round-000000/learner/train.jsonl`
`score_centering.rho_max` = 211, 100, 242, 213, 244, 203, 148 on the seven updates (policy_loss declares TIS with
k = 128); rounds 1-3 show 1.5-10. Round 0 (steps 93-99) was collected through the SGLang actor against a native
learner; whatever produced rho up to 244, that round's update was dominated by a handful of tokens. The Sonnet
dragnet should confirm whether `rho_max` is pre- or post-truncation and why the mismatch vanished in round 1.

**2.5 JOLT+ as configured is cost without signal.** `policy_loss.beta_s = 0`, teacher attempts are one per group
so their advantage is 0 (`diag-sanity-1109` test 1: "Teachers: advantage 0"); the only teacher contribution is the
`beta_t` KL term, and privileged episodes appear only in the first 1-2 of each round's 7-9 updates
(`jolt.privileged_episodes` 9,6,0,0,0,0,0 in round 0). Meanwhile the privileged twin is still packed into every
graph, doubling forward tokens. Sixteen of 144 rollouts per round and ~half of learner compute buy nothing.

**2.6 Evaluation drift is unmeasured.** Held-eval policies 110/118/126/134 were measured under four code commits
with a scoring-relevant fix between the first two (section 1.3); the dev16 reference moved 12 -> 10 on re-run.
No A/A measurement (same checkpoint twice under the two code versions) exists for the held set.

**2.7 The SFT windows ran on a mixture whose dominant families are scaled to 0.001.** Window-1 census
(`W/sft-window-000001/learner/supervision-objective-census.json`): `native-communication-template` 13,639 rows at
scale 0.001, `port-record-wrapper` 18,930 rows at 0.001, 101,326 rows total. These rows still cost forward compute
and consume the one-epoch visit budget. Not a correctness bug, but a 30% waste of the scarce SFT time on rows the
recipe has decided to ignore.

## 3. What to stop doing

1. **Stop training in pure BF16 with stochastic rounding at LR <= 1e-5.** Withdraw the "no full precision" rule for
   LFM3-1.7B. Until FP32 master weights are in, no update is worth its GPU time.
2. **Stop the w5 controller for good** (it is paused). Do not roll it back and resume: every checkpoint from 92 to
   134 is the same model plus noise, and the loop that produced them cannot learn.
3. **Stop gating on dev16.** Sixteen cases cannot distinguish 8 from 12. Keep it as a smoke test only.
4. **Stop changing code under a running measurement.** Pin one commit for a whole collect -> train -> eval cycle;
   record it; compare only within a pin. "Improvements ship on by default" and "cut over immediately" must not apply
   to the actor, harness or scorer during an experiment.
5. **Stop the JOLT+ teacher, privileged twins, Neuralese bonus, memetic programs, expert-iteration shares,
   pre-collection and interleaved SFT windows** until a plain update demonstrably improves held-out success. Each
   returns as a controlled addition with its own paired measurement.
6. **Stop whole-graph replay as the learner's data path for now.** Treat each model call as one (prompt,
   completion) sequence with the episode reward; this is what every working agentic-RL setup does, cuts learner
   forward tokens ~10x, and removes the custody machinery from the critical path. Whole-graph Neuralese gradients
   return when there is a Neuralese capability to train (use rate today 5-13%).
7. **Stop building infrastructure.** No new transitions, qualification scopes, controllers, custody schemas or
   register rows for two weeks. Measured learning is the only deliverable.
8. **Stop "YOLO, no experiments".** Two or three tiny controlled runs (hours) decide more than two weeks of
   production.

## 4. Plan for the next 48 hours

Principle: one pinned commit, one model, one metric (held 64 x 2, paired McNemar against the start policy), plain
supervised/RFT updates with FP32 master weights, nothing else changes. All learner work fits in ~30 GB.

**Hour 0-2: baselines under one pin.** Let `heldeval-driver-92` finish (~1 h). Then run the same held eval on the
base checkpoint (fresh-init `5ff07638`, `/mnt/external/natlang-development-data/archive/data/lfm3-rl-dgx-20261029/fresh-init/checkpoint.pt`)
under the same commit and series. This answers the open question of whether 92 updates of SFT did anything.
Start policy for everything below = the better of {base, step 92}.

**Hour 2-6: Experiment 0, learnability (the first experiment; success criterion below).** Use data that already
exists: the 58 + 52 verified-successful student episodes of rounds 3 and 4 (`W/round-00000{3,4}/rewards.jsonl`
`outcome == task_success`; the episode texts are in the row dirs / `rollouts.jsonl`; `rl/rollout_distillation.py`
already lowers them). Build per-call rows: for every model call in a successful episode, the rendered prompt as
context (loss weight 0) and the sampled assistant output as target. Expect ~2,500 rows. Fine-tune the start policy
with a plain HF/PEFT loop (`transformers` 5.18, `peft` 0.21 and the vendored LFM3 HF port `T/model/lfm3_port.py`
are in the venv): FP32 master weights, AdamW, LR 1e-5 full-parameter (or LoRA r=64, alpha=128, LR 2e-4 if memory is
tight), 3 epochs, bf16 autocast, assistant-token loss only. This is minutes of GPU time. Then re-run the same 31
training tasks x 8 samples through the pinned harness against the fine-tuned model served by SGLang.
*Success criterion: training-task success rises from the recorded 37-42% to >= 70%.* If it does not, the data
lowering or the serving path is broken and nothing else should run until that is fixed (this is the positive control
the project has never had). If it does, the model can learn from its own successes and the loop is sound.

**Hour 6-24: Experiment 1, one rejection-sampling fine-tune (RFT) round on held-out.** Collect from the start policy
on 256 training tasks (selected from the pool minus held split groups with the existing `select_tasks(exclude_ids=)`)
x 8 samples with the existing collector and SGLang actor, concurrency raised until the actor is GPU-bound (target
>= 1,500 generated tokens/s aggregate; enable radix prefix caching, raise slots to 48-64). Keep every successful
episode (expect 700-900), lower to per-call rows as in Experiment 0, fine-tune the start policy once (same
settings, 2 epochs), evaluate on the held 64 x 2 under the pin.
*Success criterion: >= +8 successes on the 128 paired samples vs the start policy (McNemar p < 0.05), with
loop_rate not above the start policy's.* Expected: RFT on a model at 40% success typically gives +5-15 points per
round on in-distribution tasks.

**Hour 24-48: second round or GRPO.** If Experiment 1 passes: iterate once more (collect from the new policy), then
switch to plain GRPO (group size 8, 64-128 tasks per step, per-call sequences, LR 1e-6 with FP32 master, KL to the
start policy 0.01-0.05, no teacher, no twin, no bonus), measuring held every ~4 h. If Experiment 1 fails while
Experiment 0 passed: the gap is generalisation across families, not learning; widen the task sample to 1,024 tasks
x 4 and lower the LR by 3x before anything else. Only after two consecutive held gains re-introduce one novelty at a
time (Neuralese-declaring task variants first, since they are data not machinery), each with its own paired eval.

Infrastructure to touch in these 48 hours: a ~150-line SFT script (HF loop, FP32 master, assistant-only loss), a
~100-line episode->rows lowering that reuses `rollout_distillation`'s reading code, a one-line concurrency change
to the collector, and a pinned-commit discipline. Nothing else. If the existing native trainer can be run with FP32
master weights (`auxiliary_rounding='kahan'` already keeps the intent; what is missing is the forward reading the
compensated weight), that is acceptable too, but only if it needs no new mechanism.

## 5. Which standing rules cause the circling (for the owner)

- **"No full precision" (recipe, 10-28).** Root cause of 1.1/2.1. Right for a 100B model on small cards; wrong for
  1.7B on 121 GB.
- **"YOLO run, no experiments; we are not doing science" (committed decisions; AGENTS "adaptive production").**
  Prevented the one controlled comparison (base vs trained, plain SFT at a working LR) that would have exposed 1.1
  on day one. Two bounded experiments per week are cheaper than two weeks of production that cannot learn.
- **"Improvements ship on by default; cut over immediately; deployment counts" + "run from the checkout".**
  Every measurement is of a different system (1.3, 1.4). Needed instead: pin per experiment, change one thing.
- **"Whole-graph updates, custody, exact occurrence bindings" (10-16).** Correct as an eventual Neuralese training
  requirement; as the only data path it costs 16x learner compute and most of the controller code before there is
  any Neuralese behaviour to protect (use rate 5-13%).
- **"Frugal eval" + dev16 as the gate.** The project has been flying on an instrument with +/-2 noise on 16 cases.
  Evaluation is where the compute should go until something moves.
- **"Act on current facts / nothing waits / hourly heartbeat re-justifies all work".** Produces the restart and
  reconfiguration churn in `W/` and decisions.log 347-360: seven config versions in two days, five learner
  launches for one round, rollbacks debated on noise. A heartbeat should ask "is the pinned experiment still
  running", not re-plan.
- **"Think natlang-native; policy is natural language; one implementation per concept".** Fine for the product;
  applied to the training stack it produced 133k lines of bespoke trainer/controller code where a 500-line HF loop
  plus an off-the-shelf server would do.
- Rules that are right and should stay: "degradation means bug" (it was a bug), "held-out set steers, not dev16"
  (11-10), "verify, don't assume", "fallow register" (its pattern list diagnosed this before I did), and the
  2026-10-31 finding "SFT regressed vs base" which was the correct alarm and was then lost.

## Appendix: files consulted

`W/config.json`, `W/state.json`, `W/round-00000{0..4}/score.json`, `W/round-00000{0..3}/learner/{train.jsonl,report.json}`,
`W/sft-window-00000{0,1}/learner/{train.jsonl,report.json,supervision-objective-census.json}`,
`W/held-eval/{manifest.json,policy-*/result.json}`, `W/round-00000{3,4}/{rewards.jsonl,generations.jsonl,packet.jsonl}`,
`ops/decisions.log` (lines 1-360), `ops/diag-sanity-1109/REPORT.md`, `ops/pop-behavior-eval/comparisons/*.json`,
`plans/neuralese/{LFM3_RECIPE_20261028.md,LFM3_COMMITTED_DECISIONS_20261028.md,FALLOW_REGISTER.md,ENGINEERING_FINDINGS.md,RECIPE_REVIEW_20261103/SYNTHESIS.md}`,
`T/train/optim.py`, `T/train/muon_kernels.py`, git log (commit 1d724c8f8, bfcb47464, e5d1bec5..f6c8b49e), `AGENTS.md`.
Checkpoint comparison: ledger unit `review-ckpt-delta-2` (CPU, 5.8 GB peak), checkpoints
`ops/bf95-lowlr-1105/rollback-step92/checkpoint.pt` and `W/{round-00000{0,1,2,3},sft-window-00000{0,1}}/learner/checkpoint.pt`.
