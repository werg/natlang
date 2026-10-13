# DGX coordinator handover, 2026-10-13 ~08:10 UTC

The previous DGX coordinator session ends here. Its in-session subagents stop with it. Ledger units, systemd units,
containers and committed code keep running. Resume from this file plus `python3 scripts/coord.py inbox --ack`.

Owner's plan for the new session: one **Sonnet** subagent runs all ongoing work and monitoring (the list below). The
coordinator session itself stays light.

## 1. Critical path: Mellum v4 (do first)

- v4 lineage: `runs/neuralese-direct-mellum-20261011/run-v4/`, recipe `neuralese-direct-mellum-v4-from-v3-warmup.json`,
  union r3, program-header. It is **stopped** for the weekly NatLang teacher window (06:29:46Z–09:29:46Z,
  `runs/research-teacher-20261006/teacher-window.json`).
- The resume point is `direct_sft/checkpoint.pt`, step 2000 (the stop at update 2200 was SIGKILLed mid-save after
  600 s: the 47 GB write ran at ~105 MB/s; the partial file was deleted). `best-checkpoint.pt` is a symlink to
  `/mnt/external/natlang-active-lineage/.../best-checkpoint.pt`.
- **Relaunch at the window end (09:30Z), or as soon as the window items below are done:**
  1. stop the teacher: `systemctl --user stop natlang-qwen36-teacher.service`;
  2. `cd /home/werg/natlang && git fetch && bash runs/neuralese-direct-mellum-20261011/launch-v4.sh $(git rev-parse --short=8 origin/main) --handoff`
     (launch-v4.sh is untracked; its stop grace is now `docker stop -t 1500` / `--stop-seconds 1560`);
  3. confirm admission (`systemctl --user list-units 'natlang-neuralese-direct-mellum-v4*'`) and the first updates in
     `run-v4/direct_sft/train.jsonl`, at about 26 s/update. If it's much slower, check `nvidia-smi` for other GPU compute users.
- v4 now gets, via main: task eval synchronous every 1600 updates (`task_eval.async false`) with the new budgets
  (7200 s / 128 requests / 160 turns, ecff01ef); the checkpoint offload plus optimizer pruning of non-latest
  checkpoints (9fc35a2f, 76c875ff); held packet v3; the h_cut fix (65052501). If union r4 is registered with its v4
  override by then, the handoff takes it (check the plan output).
- In-run task success so far: step 1600 **4/16**, step 2000 **8/16** (packet v2 subset). Held CE 0.55 (base 1.48).
  Nested members are active.
- GPU rule: no GPU compute jobs beside v4 (they halve its speed; the ledger governs memory only). Exception: one
  LFM-350M e2e dragnet arm at a time. If v4 then slows by more than ~15%, move the arm to Pop or CoreWeave.

## 2. Teacher window items still in flight (the conductor agent stops with the session)

- Item 0, the Mellum base reference: unit `heldv3-base-mellum` → `runs/heldv3-base-mellum-full/task-eval.jsonl`. When it
  lands, set `task_eval.reference_successes` to the lower of the two runs in the Mellum direct recipe
  (plans/NATLANG_PILLAR.md item 0), then commit.
- Teacher up on :8082 (`scripts/ensure_teacher.sh`, `scripts/teacher/run-qwen36.sh`). Items 1–3 (specializer dry
  run `--profile dgx-qwen`, measurement batches, the MODEL_FACING_CHANGES debt via the A/B replay runner
  `ab-replay`, 3e07028a): record what ran in plans/NATLANG_PILLAR.md "Log".
- The full check-main: unit `natlang-full-check-687b8076c7` is queued, and it can't fit beside v4. Run it in gaps.

## 3. Queued ledger units (survive the restart; check results)

e2e tests14 and the dragnet round-3 arms (heads 3e-6 / 3e-5 with TCP 3e-6), the full check, `mellum-deploy-gguf-ternary`,
`s7rl-72x8-muon10006` (RL), `e2edata-assemble-v3`, `legacy-llamacpp-conformance`, `ckpt-write-bench2` (checkpoint
write speed), `memetic-q48task2-20261013`, `ckpt-offload-oneshot-073900`, `swe-norm-twins`, the `render-r4-*` units (union
r4), `repeat-dragnet-open-ab`. Running: `e2edata-mine-v3`, `swe-bulk-d13-followup`, the `natlang-task-eval-dispatch` service.

## 4. Workstreams whose agents stop now (pick up from the files and commits)

| Workstream | State | Where |
|---|---|---|
| Union r4 (both lines) | renders running; then register, sync to Pop, v4 `--handoff --plan-only`, data-continuation override. Owner input: give agentic tool-loop families (pi, harness, SWE twins) their own higher-weight sampling group; supervise turns after tool results | `render-r4-*` units, `runs/natlang-target-union-*-r4`, registry |
| E2E collapse | joint TCP at 3e-6 keeps crisp CE; writer rate is the next suspect (round 3 queued). Owner: projection and backbone train jointly; frozen is diagnostic only. Unpushed commits in `/tmp/wt-e2e2`: e8bae94e (family interleave, TCP rate), 6aab73b1 (layer staging plus donated_buffer, which is also uncommitted in the shared checkout), 7b78d0e2 (streaming all of e2e corpus v2) | coord dgx-e2e notes, `runs/e2e-collapse-dragnet-20261012/` |
| E2E data | corpus v2 registered (95,106 edges, 80 families); v3 mining running; call-store runId bug (`curriculum-reference`/`native-run` shared runIds) to fix in ts-host | plans/neuralese/E2E_DATA.md, `scripts/neuralese_data/e2e_mine.py` |
| Repetition dragnet | students loop (same action plus same observation ≥8 turns). Checking for looping training targets, missing observations, train/serve mismatch, decoding, runtime feedback | plans/neuralese/REPETITION_DRAGNET_20261013.md (if written), unit `repeat-dragnet-open-ab` |
| Eval/runtime fixes | signal forwarder robustness in train/recipe.py; why the checkpoint write is 105 MB/s (`ckpt-write-bench2`); error texts: adopt the measured-best wordings ("sentence" for return_result rejections; namespace error: the best of 3 variants, revert 4fcd23a7 if worse); StopSignal record plus conftest handler-restore fix (in /mnt/external/wt-twarm) | MODEL_FACING_CHANGES.md, branch errtext-20261012 / `/home/werg/data/wt-errtext` |
| Storage | moving non-latency-critical data NVMe → /mnt/external (target ≥300 GB free); plan in `/home/werg/data/storage-plan-20261013.txt`. Owner: deletions less eager | AGENTS.md storage amendment |
| Revival | memetic rerun (48 queries), RL scaling (72×8), llama.cpp Neuralese projector GGUF plus ternary experts | plans/REVIVAL_20261012.md |
| CoreWeave | controller built (`scripts/coreweave/`, W&B auth in `~/.config/natlang/coreweave.env`). The plain GPU sandbox runs (RTX PRO 6000, 97.9 GB, driver 595.58 / CUDA 13.2); sandboxes with our extras stay "creating". Issue report: `/tmp/coreweave-sandbox-issue-report.md`. Hardware plan, benchmark jobs (`hw-bench-probe` first) | plans/COREWEAVE_SANDBOXES.md, plans/COREWEAVE_HARDWARE_OPTIMIZATION.md |

## 5. Open owner decisions

- CoreWeave: email support with the issue report, and/or allow about 5 slow one-at-a-time sandbox creates to find the
  setting that blocks startup.

## 6. Monitoring the Sonnet subagent should keep

- `python3 scripts/coord.py inbox --ack` at least every 50 min. Answer Pop; brief Pop on shared changes.
- The failure watch: `runs/neuralese-direct-mellum-20261011/failwatch-once.sh` (wrapper; logic in `runs/neuralese-direct-mellum-20261011/failwatch.sh`) (v4 unit inactive, guard stops, container
  exits, GPU idle, NVMe free+reserve < 60 GB, v4 median update > 31 s). The job dir may not survive. If so, rebuild it
  from that logic.
- Agents' merges: quick check before push; full check in gaps; main must stay green.
