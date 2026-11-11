# SFT template excision and expert iteration (owner order, 2026-11-10)

Order: excise the template-dominated data from the production SFT windows and replace it with expert iteration, live
for the next window (w5 window 1). Production: `ops/bf95-lowlr-1105/w5` (CE policy `corrected-mix-full67-w1-v44b-1103`,
stage `native_joint_sft_v4`, 8 updates x 16 batches per window).

## 1. Measured composition of window 0 (`sft-window-000000`, updates 103-110)

Method: exposure credits of the stage's saved SFT frame, window 0 checkpoint minus the round-1 learner checkpoint (the
parent): 892 graph visits, one per row, 448,429 own-output supervised tokens (the trainer's own count). Rows are mapped to
their logical family with the CE policy's family map; own-output size is the `own_chars` of the core-focus scan cache
(`ops/bf95-lowlr-1105/core-focus-1105/scan-cache.jsonl.zst`, 834 of the 892 rows; agent-sessions (52 graphs) and
acquired-d13 (6) are not in it), scaled by each family's declared CE scale. Loss share = own-output mass x declared scale.

| family | graphs | share of graphs | declared scale | share of scaled own-output loss (covered rows) |
|---|---:|---:|---:|---:|
| natlang-inline | 401 | 45.0% | 7.49 | 38.5% |
| natlang | 268 | 30.0% | 8.00 | 36.9% |
| coding-swe | 24 | 2.7% | 1.12 | 9.8% |
| native-communication-rest27 | 129 | 14.5% | 4.07 | 7.3% |
| coding-sessions | 3 | 0.3% | 8.00 | 6.0% |
| natlang-chains | 2 | 0.2% | 3.32 | 1.3% |
| native-communication | 7 | 0.8% | 2.90 | 0.3% |
| agent-sessions / acquired-d13 | 52 / 6 | 6.5% | 4.89 / 1.04 | not covered |
| **port-record-wrapper** | **0** | **0%** | 0.015 | **0%** |
| **native-communication-template (big2)** | **0** | **0%** | 0.071 | **0%** |

856 of the 892 graphs came from the 7,786-row diverse core focus (`core-focus-1105`), which the trainer schedules first;
the other 36 are natlang rows. Window 0 trained no template graph at all.

The 2026-11-03 review figure (wrapper about 41% and big2 about 37% of loss, 78% together) described the recipe before the
corrected mix. The corrected mix (committed 2026-11-03 / 2026-11-05) already did the review's decisions:

- family reassignment: the wrapper rows are `port-record-wrapper` (18,930 rows), big2 template edges
  `native-communication-template` (13,639 rows);
- declared scales 0.015 and 0.071, planned remaining loss shares 0.5% each (`corrected_diverse_mix.family_balance` of the
  CE policy), against 8-10% for each genuine family;
- quota revision: template edges 1-2 visits instead of 7; withdrawals of 1,021 constant-instruction producers, 1,175 D12
  collapses, 2,833 duplicate-label targets and the near-duplicate targets (`corrected_diverse_mix.withdrawal_kinds`,
  source correction `source-correction.json`);
- the diverse focus (above) supersedes the template-heavy one.

What remains of the template families is compute, not loss: in the pool they are 17.3% of logical forward tokens
(wrapper 8.6%, big2 8.7%); after window 0 their remaining quotas are 18,841 wrapper visits and 27,265 big2 visits. They sit
behind about 5,600 unvisited focus rows (roughly six windows at the current pace).

## 2. Excision: what was done and what was not

Done (default on, `distillation_withdrawn_families: []` opts out): every memetic append window floors the declared CE scale
of `port-record-wrapper` and `native-communication-template` to 1e-3 (`min(current, 1e-3)`, never raising). A declared-scale
refresh is the one family change the trainer's resident CE continuation admits on an append (`native_resident_ce`: a
declared/1 refresh may change scales and extend the map; the objective transition records the predecessor). The rows keep
their earned exposure and stay in the source; no source file is touched.

Not done, with the reason: physical withdrawal of the rows. The resident CE extension is append-only
(`native_resident_ce.extension_policy` rejects a map with fewer rows), the trainer's ordinary availability deferral
(`native_availability`) refuses a CE-objective lineage, and `withdrawn_native_graphs` needs filtered leaf files, an own-index
continuation, a rebuilt family map, CE sources, packed-geometry sidecar (the capacity gate reuses only an exact prefix) and
a new quota basis, i.e. a static source correction like `corrected-mix-diverse-1103` (a multi-hour build with its own
CPU preflight). Measured need is nil for the next six windows (they are scheduled after the focus and their loss is
floored), so this stays a deliberate exclusion until the focus is consumed; the rows are then the first candidates of the
next static correction (deferral of the remaining wrapper/big2 rows, quota revision to the floor).

## 3. Replacement: expert iteration (`rl/rollout_distillation.py`, `rl/memetic_admission.py`)

- Every verified successful student episode of every scored round is a row (was: one per task). Filters unchanged:
  student episodes, train tasks, no memetic episode, closed whole graphs (no old external state), sampled outputs only,
  the current round's learner policy as actor. Neuralese-using episodes rank first and carry their return as weight. Exact
  duplicate sampled outputs of one task are dropped (never repeated); a row is visited once.
- Each row records its own sampling policy; a package names every actor, and admission still refuses a package whose actors
  are not checkpoints of the lineage.
- Admission gives each distilled family its own declared scale and priority: rollouts scale 8 (the cap: the same weight per
  sampled token as the genuine natlang families), priority 2; prepared statics scale 4, priority 1 (policy rows run first);
  memetic guidance 1. Override with `distillation_family_scales`.
- Window capacity: a window digests `refresh_after_updates x accumulate_batches` = 128 packed batches. A captured successful
  episode is one graph of 0.1-0.9M logical forward tokens (round 2 sample, 37 episodes: median 165k, mean 291k), so it fills a
  batch alone. Measured on the same sample, an episode's own-output weight is 1.9k on average (70.6k for 37) against 2.5k per
  batch of window 0 (315.6k weight over 128 batches); with `loss_scale = scale / token_reference(100)` one episode at scale
  8 carries about as much loss as an average batch of window 0 (152 against 160 mass units). The share of batches is
  therefore the share of loss. Default shares of the batches: rollouts 0.4, statics 0.2, memetic 0.1
  (`distillation_window_shares`); rows over a share stay deferred (never dropped), and admitted rows no update has visited
  yet are charged first, so a backlog never grows and no row is visited twice. `distillation_max_share` (row-mass cap of the
  source) is now 0.5.
- Default on (`rollout_distillation: false` opts out; continuations adopted it through the recorded transition
  `rollout-distillation-20261110`, effective from w5 round 2).

### Rounds already scored: what could be admitted

The order asked for round-000000 (76 successes) and round-000001 (27). Their student successes are 67 episodes in 15
tasks (4 Neuralese-using) and 18 in 6 tasks (1 Neuralese-using); the other successes are teacher episodes. They cannot be
admitted: lowering an episode to a whole graph replays its actor states from the round's durable collector bundle
(`round-N/blocks/manifest.json`), and the bundles of rounds 0 and 1 are gone (pruned after the updates; every lowering
ends in `native RL state requires its durable actor bundle`, all 85 episodes, also the non-Neuralese ones). The tool that does
the top-up (`scripts/neuralese_data/backfill_rollout_distillation.py CONTROL_OUT`, also run by the hook at every scored round)
works on rounds whose bundle exists: on the in-progress round 2 (bundle present, partial 138-rollout snapshot) it lowers
37 of 37 student successes (11 tasks, 6 Neuralese-using), 0 failed, 0 protected. Round 2 and later are therefore queued by
the hook at scoring time, while their bundle exists.

## 4. Why the appended rows needed the focus (found by the window-1 rehearsal)

The trainer schedules every row of an active pending focus before anything else (`native_pending_focus.promote`, a stable
partition of the packed batches), and appended rows rank only as the fresh-source frontier, one batch in five *behind* the
focus (`native_source_frontier.interleave_batches`: a batch holding a focus row stays in the priority lane). w5 has an active
7,786-row focus with 5,620 rows unvisited (about 800 batches against 128 per window). Simulated on the real window-1 schedule
(scratch copy of w5: real checkpoint frame, real queue plus the round-2 package), the first 128 batches of window 1 were 100%
old focus rows: the previous admission (priority and CE scale only) would have placed no appended row in window 1, and
the plan failed the trainer's monotone-window bound (`monotone source window exceeds its token buffer bound`, 1.08M > 1.05M
tokens) in the CPU preflight. Two changes follow:

- the append writes a successor pending-focus request (`_successor_focus`): it declares `supersedes` the active request's
  SHA-256 (the trainer verifies it against the checkpoint's focus history), lists the appended rows first and then every row of
  the superseded request (consumed ones are no longer pending), and is bound in the append and settled recipes; a stage whose
  focus is absent or finished is left alone (appended rows then lead through priority). `native_resident_ce.extension_policy`
  admits an append that changes `pending_focus` when `_pending_focus_continuation` authenticates the change;
- each rollout episode is its own source leaf (cohort). The trainer serves cohorts least-served first by token fraction
  (`balance_urgent_batches`), so one cohort of 37 giant episodes was served about 6 batches while 70 cohorts of ordinary
  rows were served first; single-episode cohorts are all unserved and priority 2, so they lead.

Simulation of the first 128 batches of window 1 with both changes: 37 rollout batches (29%), 8 prepared-static batches
(104 rows; the 20% static share is not reached because static cohorts are served fairly), 83 old-focus batches; no frontier
rows, monotone-window plan passes.

## 5. Window 1 rehearsal (scratch copy of w5; w5 state untouched)

`interleave.run_sft_window` for window 1 on `/mnt/external/claude-scratch-ceadmit/w1sim2` with the real queue plus the round-2
package: admitted 316 rows (37 rollout-distillation, 279 prepared-source; 3,800 static rows stay deferred), 0 refused, 0
excluded, no admission failure; template families floored; successor focus 316 added + 7,786 retained; trainer command built
(`--continue-from` the window-0 checkpoint). Expected window 1: the policy's own episodes about 29% of the batches and about 28% of the
loss (rollouts at scale 8 carry 152 mass units each against 160 per average batch), more as rounds 2 and 3 add episodes (up to the 40%
share), about 6% prepared statics, the rest the diverse focus.

CPU continuation preflight of that command (45 GB ledger budget; a 10.5 GB budget is OOM-killed before the schedule is planned,
as every earlier attempt was): see the final section.

## 6. Trainer CPU continuation preflight and deployment

Preflight of the window-1 command (`--continuation-preflight-only --device cpu`, real 18.5 GB window-0 checkpoint, real
optimizer/RNG, scratch `--out`; run outside the ledger under `systemd-run -p MemoryMax=56G` because the ledger queue was
jammed by two stale requeued units behind the controller's fixed 100 GB claim; peak about 33-48 GB, a 10-14 GB budget is
OOM-killed): **passed** (`continuation-preflight.json`, global step 110): 101,312 rows (100,996 + 316 appended), 5,986 focus
rows pending (5,670 old + 316 new), the superseded focus recorded (`superseded` by the successor, 8,102 rows active),
115,937 remaining graph visits, the monotone-window schedule planned. The first attempt failed on exactly the schedule problem of
section 4 (1.08M > 1.05M tokens), which the successor focus and the per-episode cohorts removed.

Deployment: code on main (`697fd1043`, `e3aa0684d`, `bc7edb977`, and the docs commit after them). The production controller is
started from the checkout: the rollout hook (`rl/rollout_distillation.py`) and the admission (`rl/memetic_admission.py`) are
imported by the running process at start/first use, so a controller restart in place picks up the newest code. The packages of rounds
2 and 3 are queued by the hook at their scoring (their collector bundles exist then); window 1 admits them with the five staged
static packages. Check after round 2 scores: `round-000002/rollout-distillation-all.json`, a `rollout-distillation` package in
`memetic/distill/admission-queue.jsonl`; after window 1 starts: `sft-window-000001/memetic-admission/admission.json`
(`share_cap`, `pending_focus`).

## 7. Physical withdrawal: audit and static source correction (2026-11-11, owner clarification)

Owner order: "excise" withdraws rows from the training schedule through a new corrected source; no existing data file is
touched (originals, `corrected-mix-diverse-1103`, CE policies stay as they are); withdraw only what is truly useless, never a family
blindly. Tool: `scripts/neuralese_data/prepare_template_excision.py` (`audit`, `build`, `preflight`, `commit-admission`); package of the
w5 window-0 boundary in `ops/template-excision-1111/package` (`withdrawal-plan.json` has the criteria, counts and decoded samples).

Audit of the 32,569 remaining rows (the 2026-11-03 corrected mix had already cut 105,245 wrapper rows to 18,930 and 122,700
template edges to 13,639): both families are mostly useful data, not scripts.

- `port-record-wrapper` (18,930): 18,930 distinct teacher programs, one row per task already (the review's rule); real tasks
  (competition maths, tool calls, web research). Only repeats of a kept row go: exact Jaccard >= 0.9 of the masked-target 4-gram sets
  (4 clusters): **9 withdrawn**, 18,921 kept (89 of them earned). Its quota is one visit per row, so there is nothing to cap.
- `native-communication-template` (13,639 edges): producers are input-specific (13,075 distinct producer replies; payload shared by
  8+ edges had been withdrawn in 1103), but the edges of one fan-in (1-4 per consumer answer, 4,851 answers) each re-train the same
  consumer answer, twice (quota 2). Keep every edge (genuine producer-to-consumer Neuralese state), withdraw an edge repeating a
  kept edge's producer reply into the same consumer answer (**1**), and cap the kept edges at **one visit** (`quota_revision`,
  13,638 rows; earned credit retained): remaining template visits 27,265 -> 13,627.

The correction (`natlang.native-source-correction/1`, `deferred` rows keep visit debt and credit in the pending ledger) is built
for the source a window's parent checkpoint trains: filtered leaves (new files), family map minus the withdrawn rows, CE policy
(`objective_transition`, source-bound quota basis, `history_settlement` over every historical first-row observation), packed-geometry
sidecar, capacity descriptor, successor pending focus (new leaf SHA-256s), append and settled recipes.

### The queue path (2026-11-11: replaces the manual stop/build/preflight/commit/restart cut-over)

Producer (checkpoint independent): `prepare_template_excision.py audit --controller-out OUT --out DIR --queue OUT` writes
`withdrawal-plan.json` (`natlang.source-withdrawal-plan/1`: `withdraw` {row id: family, criterion, binding}, `visit_cap` {row id:
`required_visits`}, reason, evidence, the audit's counts and decoded samples) and appends it to `OUT/memetic/distill/withdrawal-queue.jsonl`;
any plan is queued with `queue_source_withdrawal.py --controller-out OUT PLAN [--dry-run]`. The queue is a file of its own, not the package
queue: a controller whose code does not know it never reads it (an entry waits for the first boundary of a controller running
`rl/source_withdrawal.py`; no `restart_controller.py --in-place` ordering, unlike a new package family).

Consumer: `memetic_admission.prepare_window` at every interleaved SFT window boundary resolves the pending plans against the parent
checkpoint's saved frame (`source_withdrawal.resolve`), then, in one source change with the queued packages:

1. a plan row of an earned frame (saved credit >= quota) is never withdrawn (skipped, counted `earned_kept`); a row no longer active or with a
   different binding is skipped; a row whose family differs from the CE family map refuses the whole plan (`refused`, never retried); a visit cap
   lowers a saved quota only where it is higher (the trainer needs at least one deferred row per correction, so caps ride on a withdrawing plan);
   a plan with nothing left is `empty` (consumed, recorded);
2. every affected leaf is filtered (`withdraw_native_graph_targets.filter_source`: other bytes and order preserved, index projected; a leaf whose
   rows all go leaves the manifest; originals untouched) into `sft-window-N/memetic-admission/`, the manifest tree rewritten (`withdrawn-sources.json`),
   its packed-geometry sidecar written;
3. the admitted packages are appended after that filtered manifest (`train-sources.json` = filtered manifest + leaves, priorities as before);
4. one proof `natlang.native-source-correction/1` over the final source: `deferred` rows (withdrawn, visit debt and credit kept in the pending ledger),
   `added` rows (the appended ones, fresh quotas; `apply_source_correction` supports both in one proof), the `quota_revision` ledger;
5. the CE policy (one pass from the parent's: replaced leaf declarations, family map minus withdrawn plus appended rows, declared scales, objective
   transition from the parent's identity, quota-aware basis for the final source, `history_settlement` anchored on this proof), the packed
   geometry (filtered prefix extended over the appended rows), the successor pending focus (appended rows first, withdrawn rows dropped, others on
   their new leaf SHA-256s), the append/settled recipes (`source_correction` bound; the continuation is validated with
   `_source_correction_continuation`, the trainer's own `validate_admission` for the basis); committed with the ledger (`memetic_distillation.
   withdrawals[plan sha] = applied|empty|refused`) before the window config is written, so a retried boundary reuses it.

Failure: anything that may succeed later (index cache missing, a build check) raises inside `prepare_window`, which records it in
`interleave.memetic_distillation_failures` (`with_withdrawal: true`), then retries the boundary without the withdrawal, so the packages are still
appended; if that fails too the window runs on the existing source. The plan stays pending (nothing committed) and is retried at the next boundary.
The trainer applies the one source correction at the window's continuation (`source_correction_history`, superseded focus and quota revision land in the
checkpoint); the settled recipe carries it for later windows, which append on top of it as before.

Dry run on a scratch copy (live controller untouched): `prepare_template_excision.py rehearse --controller-out OUT --out SCRATCH [--plan PLAN]`
runs exactly the boundary's preparation on copies of config, state and queues; `preflight --controller-out OUT --out SCRATCH` is the trainer's CPU
continuation preflight of that window (about 48 GB; the controller's 100 GB ledger claim does not leave room: see below).

Preflight of the window-0 package (`--continuation-preflight-only --device cpu`, real 18.5 GB window-0 checkpoint, global step 110, scratch
`--out`): **passed** (`ops/template-excision-1111/package/preflight/learner/continuation-preflight.json`): 100,986 rows (10 deferred, 13,638 quota
revisions, `required_visits_removed` 13,638), the fifth recorded source correction, the pending focus superseded by the same request on the new
leaf SHA-256s (5,670 rows pending), 101,972 remaining graph visits (115,620 + the 316 window-1 rows before), monotone-window schedule planned.
The ledger cannot admit a 48 GB job while the controller holds its fixed 100 GB claim, so it ran outside admission
(`ops/template-excision-1111/guarded_preflight.py`: starts at 75 GB MemAvailable, MemoryMax 52G, OOMScoreAdjust 1000, own floor). A first attempt, adopted
into the ledger, was stopped by the ledger guard when the window-1 learner peaked; the second ran to the end. Stopping the controller first removes the problem.

A built correction is valid for exactly the checkpoint it was built against: the CE policy's quota-aware basis binds the parent frame digest (saved
quotas and credits), and a preflight of the window-1 package against the window-1 learner slot after another update was refused by the trainer
(`quota-aware CE parent quotas or exposure changed; recensus before training`). That is why the queue holds a checkpoint-independent plan and the
boundary builds the correction against `state['checkpoint']`, the exact parent its window continues (the same rule as the package appends).
