# DGX coordinator handover, 2026-10-13 ~17:40 UTC

For any agent (any provider) taking over DGX coordination. Read `AGENTS.md` first. It holds the standing owner rules
(memory ledger for everything, both lines in lockstep, assume a bug, small commits to main, brief Pop via
`scripts/coord.py`). The previous handover is `plans/neuralese/SESSION_HANDOVER_20261013.md`. This file supersedes it
where they differ. No agents are running now; ledger units, systemd units and containers keep running.

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
- DGX A/B running on CPU: `ab-lfm-step8000-full` and `ab-lfm-step8000-program-header`, which restore the student
  tensors independently. Results → `runs/ab-lfm-step8000-{full,program-header}/task-eval.jsonl`, logs
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

## 7. Owner working rules from today

- Share every sensible setting across both lines at once. The line-parity verifier in `--quick` enforces it (283958ad).
- Don't park work in branches; merge to main promptly.
- Economise on agents: few, cheaper models for mechanical work.
- Deletions are less eager: prefer moving to `/mnt/external` (see AGENTS.md).
- The projection and backbone train jointly in e2e; there's no alignment loss after entry.
