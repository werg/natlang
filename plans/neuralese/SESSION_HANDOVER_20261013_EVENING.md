# DGX coordinator handover, 2026-10-13 ~17:40 UTC

For any agent (any provider) taking over DGX coordination. Read `AGENTS.md` first. It holds the standing owner rules
(memory ledger for everything, both lines in lockstep, assume a bug, small commits to main, brief Pop via
`scripts/coord.py`). The previous handover is `plans/neuralese/SESSION_HANDOVER_20261013.md`. This file supersedes it
where they differ. No agents are running now (§8 lists the running machinery, §9 the unfinished work); ledger units, systemd units and containers keep running.

## Start of session

1. `python3 scripts/coord.py --as dgx-<role> inbox --ack`; answer Pop. Check `coord.py status`.
2. `systemctl --user list-units 'natlang-*'` and `python3 scripts/memory_ledger.py status`: v4, queued units.
3. Re-arm a failure watch: `runs/neuralese-direct-mellum-20261011/failwatch.sh` (logic) / `failwatch-once.sh`. It
   checks v4 unit inactive, guard stops, container exits, GPU idle >10 min, NVMe free+reserve < 60 GB, v4 median
   update > 31 s. **Nothing is watching v4 right now.**
4. Before every push: `scripts/check-main.sh --quick || exit 1` (gate on exit status). Main was green at 5bb9acb4.

## 1. Mellum v4 (critical path, running)

- Unit `natlang-neuralese-direct-mellum-v4-113152`, run dir `runs/neuralese-direct-mellum-20261011/run-v4/`, code
  snapshot natlang-345d32eb. It was at step ~3030: ~19 s/update alone, ~31 s with one e2e arm beside it.
- Next in-run task eval: step 3600, full system prompt, T=0.6. Compare with the base reference `reference_successes`
  11/16, which is **provisional** (greedy, one run, in `neuralese-direct-mellum-v4.json`).
- Rule: no GPU compute jobs beside v4, except one e2e diagnostic arm at a time (owner-approved, ≤ ~2x slowdown).

## 2. Combined v4 gap (planned, not triggered)

Trigger it when the **mixed direct+e2e recipe** is on main (see §3). The projection gate is ready: fc32044e,
`train/projection_align.py init`, measurement only. Union r4 can join if registered. Order:
1. Stop v4: `systemctl --user stop <unit>`; grace 1500 s; the checkpoint write is slow, ~105 MB/s.
2. Parent export, from the checkpoint's own snapshot, with full serving:
   `cd /home/werg/data/code-snapshots/natlang-345d32eb/training/neuralese && PYTHONPATH=$PWD python /home/werg/natlang/runs/neuralese-direct-mellum-20261011/export/export_direct.py <run-v4/direct_sft> checkpoint.pt <OUT> full`
   (dry run `dgxops-export-dry4` queued).
3. `projalign-mellum`: `memory_ledger.py run --unit projalign-mellum --budget-gb 34 --priority high -- sh -c 'cd /home/werg/natlang && PYTHONPATH=training/neuralese .venv-neuralese/bin/python -m natlang_neuralese.train.projection_align init --artifact <OUT> --base /home/werg/data/models/mellum21-12b-a2.5b-thinking --text-data runs/natlang-target-union-20261011-r3/program-header/mellum21/text-data.jsonl --device cuda --held-rows 150 --kl-rows 32 --out <OUT>-aligned-projection'`
   (exit 0 only if the gate passes: nearest ≥0.99, cos ≥0.98, KL ≤0.05).
4. Base reference at T=0.6, two runs: `/home/werg/data/dgxops-baseref-t06.sh` → `runs/heldv3-base-mellum-t06/`. Set
   `reference_successes` to the lower of the two runs.
5. Full vs program-header eval on the same stop checkpoint. This decides the prompt question in §5.
6. memetic: `memory_ledger.py run --unit memetic-q48task2-20261013 --budget-gb 9 --reserve-gb 2 --wait 10800 --detach --class experiment --priority medium --resumable --container memetic-student-20261012 --workdir /home/werg/natlang -- /home/werg/data/memetic-20261012/q48-then-task.sh`
7. Relaunch as the mixed run, binding `aligned_projection`. Run `--handoff --plan-only` first. The recovery data
   (`recovery_selection`, `cohort_repeats`) and the 16k context floor arrive as declared windowing continuations.
   Launcher: `runs/neuralese-direct-mellum-20261011/launch-v4.sh <sha8> --handoff` (untracked; adapt it for the mixed
   recipe).

## 3. E2E (owner decisions today; DECISIONS.md, NEURALESE_DIRECT_V1.md)

- **The soft-token mixture is removed.** The Neuralese value = E[chosen token] + C(h), C zero-initialised (fc32044e).
  A learned map h→embedding reached only cos 0.72–0.83 held. The value is exactly on the token manifold at entry.
- **No direct-stage readiness wait.** E2E entry = projection gate passed + a stable update recipe.
- **One mixed run per line:** direct SFT windows interleaved with e2e producer→consumer edges at a declared ratio.
  Neuralese-output spans are context-only in direct samples, trained by e2e (coverage within mined trajectories is
  99.96%). Uncovered ones take token CE through the TokenChoiceProjection with h detached.
- **Acceptance compares V against T_self**, a reader of the writer's own emitted tokens. Gold-T is reported as an upper
  reference only. Arm D showed the step-0 gap (0.105) is the writer's token errors against gold.
- **Rates:** the TCP at 3e-6 on Muon keeps crisp CE. A writer above ~3e-6 widens V−T (Muon 3e-4, AdamW 3e-5). At 3e-6,
  V−T stays flat but V doesn't improve. **Unexplained; treat it as a bug.** Check that consumer gradient reaches C (zero-init
  down matrix) and the backbone: per-group grad norms over the first 20 updates. Then sweep C's rate (3e-6, 3e-5,
  3e-4) on LFM u10006 with corpus v2 (arm `v2stream-b`, ~400 updates each).
- **Mixed-run code:** see the e2e lead's note on coord, "dgx-e2e handoff" (fork state, worktree path). Findings are in
  `plans/neuralese/E2E_COLLAPSE.md`.
- **Fixed today:** 47617d77 (string-typed handover writes); ab234f22 (piece kinds for held pairing); 715565b9
  (streaming loader skipped trajectories that reuse writer names); resume of direct-artifact e2e runs.
- **Data:** e2e corpus v3 `neuralese-e2e-cohort-20261013-v3` (106,380 edges) is registered and synced to Pop. Miner
  to-do: 20% of direct target tokens come from never-mined sources (`native-sft-v13-r2` 2.47M, teacher-campaign-evidence
  1.83M, semantic-inline-curriculum-s72 1.17M, static-lambda-rationalized 1.0M, bgkit-knowledge 0.4M). Extend
  `scripts/neuralese_data/e2e_mine.py`; also fix the ts-host call-store runId collision (`curriculum-reference`/`native-run`).

## 4. Data and union r4

- Recovery: `natlang-recovery-selection-r3-20261013-v1` (18,578 train) is in the shared direct-v3 recipe (Pop
  257d8a48, cohort repeat 2). Sources 2/3 (teacher-generated recoveries) are not started; design in
  plans/neuralese/RECOVERY_SUPERVISION.md.
- r4 renders waiting in the ledger: `render-r4-folder-edit-large` (Pop CoEdIT, 1,732 turns) and
  `render-r4-recovery-loop` (768 approved recovery turns). They auto-register via `pending-renders.json`. Owner input
  for r4: tool-loop families (pi, harness, SWE twins) get their own higher-weight group.

## 5. Open problem: LFM 0/16 with exact repeat loops (Pop owns the dragnet)

- Pop QAD LFM (v13 now, step ~9250): held CE 0.70 vs base 2.68, but task 0/16 at steps 8000 and 9000. 10/16 stops are
  exact repeats, even after visible error feedback. **0/418 eval turns contain `<think>`**, while 54% of actually-sampled
  training targets open with `<think>` (94% for workbench tickets).
- Ruled out: post-error supervision missing; the template suppressing think; collector stripping; the sampler; the
  repeat counter; context length. The synchronous eval uses the same engine object as training, so stale weights are
  unlikely. Remaining suspects: the QAT/quantised eval forward or train-vs-eval mode divergence, and the
  full-prompt-vs-training gap (LFM rows mostly have no system prompt; the full eval adds a 16k-character system message).
- DGX A/B (`ab-lfm-step8000-full` / `-program-header`, CPU, independent restore of the student tensors) was INVALID; rerun it per §9 item 4. Results → `runs/ab-lfm-step8000-{full,program-header}/task-eval.jsonl`, logs
  `/home/werg/data/dgxops/ab-lfm/`. Report successes, repeat stops and think share; Pop also asked for first-token
  P(<think>). Reply on coord thread pop-fed0. If the CPU restore emits <think> and the in-run eval doesn't, inspect the
  QAT/serving forward.
- Mellum does emit reasoning in eval (29% of turns at step 2000; the training target share is ~36%), and its prompt
  bytes match training.
- Owner decision today: generative evals and serving use the full system prompt for all models. Open recommendation,
  awaiting the owner: if the A/B confirms the gap, raise the full-prompt share in training to ~10–20% for both lines.
  Today it's 2.7% of direct SFT and 0% of trajectory rows.

## 6. Other queued or pending

- `dgx-parity-verifyplan-121620`: if `equal: true` in its log, remove `/home/werg/natlang-wt-parity2` and branch
  `dgx-parity-2`.
- `e2elead-align-cpu5` (alignment diagnostic) and the lfm25 r3 span sidecar.
- CoreWeave sandboxes: blocked on CoreWeave support (the owner is in contact). Don't create test sandboxes. Plans:
  plans/COREWEAVE_SANDBOXES.md, plans/COREWEAVE_HARDWARE_OPTIMIZATION.md.
- s7rl-72x8 RL run: off until a GPU gap.

## 8. Running machinery at handover (~18:30 UTC) and how to supervise it

**Long-running units** (`systemctl --user`). Stop them only by unit name or explicit PID, never by pattern:
| Unit | What | Supervision |
|---|---|---|
| `natlang-neuralese-direct-mellum-v4-113152` (+ container of same name) | Mellum v4 direct SFT, 98 GB ledger claim, priority high | `run-v4/direct_sft/train.jsonl` (updates), `eval.jsonl`, `task-eval.jsonl`, `run-v4.log`. Healthy = ~19 s/update alone. Stop grace 1500 s. |
| `natlang-e2e-dragnet-v2stream-c3e-5-193025` (+ container) | e2e arm on LFM u10006 + corpus v2, value E[chosen]+C, C rate 3e-5 (c3e-6 done; c3e-4 next if wanted) | logs `runs/e2e-collapse-dragnet-20261012/v2stream-c3e-*.log`; summarise with `runs/e2e-collapse-dragnet-20261012/summarize.py`. One arm at a time beside v4, ≤ ~2x v4 slowdown (owner-approved). |
| `natlang-task-eval-dispatch` | async task-eval dispatcher (`scripts/task_eval_dispatch.py --every 120`) | Leave running. |
| `natlang-memory-guard` | memory guard (kills unledgered memory hogs) | Leave running; everything that loads a model goes through the ledger. |
| `render-r4-e2e-musique4` (+ render container) | union r4 render | Auto-registers via `pending-renders.json` and the registration timer. |
| `swe-bulk-d13-followup` | SWE bulk data follow-up (low priority) | Check its log on finish; register outputs if it produced a corpus. |
| `mixed-land-tests` | tests for the mixed-run landing (§9) | Log `/home/werg/data/wt-mixed-tests.log` (`exit=` line at end). |
| `*-waiter` units (`render-r4-folder-edit-large-waiter`, `render-r4-recovery-loop-waiter`) | detached ledger waiters (043575ca); they launch the unit when memory frees | Nothing to do. |

**Ledger queue** (`python3 scripts/memory_ledger.py status`): `tself-land-tests`, `natlang-full-check-635a9adf46`,
`e2elead-gdiag-t` (the e2e gradient-flow diagnostic: does consumer gradient reach C and the backbone?), `tself-recur3`
(a leftover from the stopped helper; redundant with tself-land-tests, so cancel it), `render-r4-folder-edit-large`,
`render-r4-recovery-loop`, `swe-norm-twins-*`, `e2edata-mine-v3`, `dgxops-export-dry4` (v4 parent export rehearsal).
`dgx-parity-verifyplan-121620` may also still be queued. The queue drains as memory frees; v4 holds most of it.

**Timers (recurrent tasks; keep enabled):**
- `natlang-full-check-main.timer`: full `check-main` on new main commits; it posts failures to coord (dgx-full-check).
  Fix or route every red file.
- `natlang-render-registration.timer`: registers finished renders listed in `pending-renders.json`.
- `natlang-storage-retention.timer`, `natlang-archive-to-external.timer`: storage hygiene (AGENTS.md amendment: move to
  `/mnt/external` rather than delete). Keep NVMe free above ~60 GB; it was ~118 GB.
- `natlang-clef-pools-daily.timer`: daily Clef pool refresh (shared quota with Pop).
- root cron every 3 h: `/home/werg/bgkit/scripts/drop_host_caches.sh`.

**Recurring duties of the supervising agent:**
- At least every 50 min: `coord.py inbox --ack`, answer Pop, keep `coord.py status --set` current.
- Failure watch (§Start, step 3), re-armed every ~30 min if your tooling needs it.
- Watch v4's update time; if more than one GPU job is beside it, stop the extra one.
- Full-check results: main must stay green.
- When Pop pushes corpora: twin-render for both tokenizers, register, include in the next union.
- Brief Pop with one `decision` per shared change. Any setting added to one line goes to the shared recipe; the parity
  verifier enforces this.

## 9. Unfinished workstreams, exact state

1. **Mixed direct+e2e run (critical; it gates the v4 gap):** 4 commits rebased on main in the worktree
   `/home/werg/data/wt-mixed` (detached): EmbeddedE2E objective, `direct_sft` e2e_mix interleave, mixed launch recipes
   `neuralese-mixed-mellum-v4.json` / `neuralese-mixed-lfm-qad-broad-v1.json`, parent_readiness removed. Tests are
   running (`mixed-land-tests`). If `exit=0`: `cd /home/werg/data/wt-mixed && git fetch && git rebase origin/main &&
   scripts/check-main.sh --quick && git push origin HEAD:main`. If red: fix, don't park.
2. **T_self acceptance (V against the writer's own tokens):** 1 WIP commit stacked on the mixed commits in
   `/home/werg/data/wt-tself` (`train/self_text.py`, `e2e_acceptance.py`, `trajectories.py`, tests). Its tests have
   never passed; they're queued as `tself-land-tests` (log `/home/werg/data/wt-tself-tests.log`). Land after item 1.
3. **E2E learning dynamics:** V flat at writer 3e-6, worse above. Read `e2elead-gdiag-t` (gradient reaching C and the
   backbone) and the `v2stream-c3e-6` / `c3e-5` arms (V−T, crisp). Pick C's rate for the mixed recipe (TCP 3e-6 on
   Muon).
4. **LFM prompt A/B (invalid; rerun):** `ab-lfm-step8000-full` OOMed at 8 GB, and the program-header arm ran stale
   code (greedy, no temperature) and was stopped. Rerun on current main, sequentially, `--workers 2`, a larger ledger
   budget (~16 GB), T=0.6; script `/home/werg/data/dgxops/ab_lfm.py`, inputs
   `/mnt/external/natlang-development-data/pop-step8000-prompt-ab/`. Report successes, repeats, think share and
   first-token P(<think>); Pop also asks for a variant without the `<|neuralese|>` position-0 open marker that training
   windows prepend (dragnet §3b). Reply on coord thread pop-fed0.
5. **Combined v4 gap (§2):** after items 1 and 3.
6. **Miner coverage (§3), recovery sources 2/3, CoreWeave spec:** not started.
7. **The llama.cpp direct projector GGUF export** (8b6f9837) predates the mixture removal; update it to E[chosen]+C
   before the `mellum-deploy-gguf-ternary` job uses it.
8. **Old WIP branches on origin, not from this session:** `gpu-tests`, `infrastructure-implementation`,
   `storage/qat-convert-checkpoint-policy`, `wip/batched-execution`, `wip/build-migration`, `wip/eval-field-check`
   (superseded by main's eval field check), `wip/games`, `wip/refinement-types`, `wip/scheduling-workflow`,
   `wip/wiki-logs`. The owner wants nothing parked in branches: review each, then merge or delete.

## 7. Owner working rules from today

- Share every sensible setting across both lines at once. The line-parity verifier in `--quick` enforces it (283958ad).
- Don't park work in branches; merge to main promptly.
- Economise on agents: few, cheaper models for mechanical work.
- Deletions are less eager: prefer moving to `/mnt/external` (see AGENTS.md).
- The projection and backbone train jointly in e2e; there's no alignment loss after entry.
