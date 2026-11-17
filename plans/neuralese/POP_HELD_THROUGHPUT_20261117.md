# Pop held-eval throughput (2026-11-17)

Question: where does the wall time of a held-v3 evaluation on Pop (RTX 4060 8 GB, SGLang fork, BF16 vendor
projection, FP8 KV) go, and what change gives more episodes per hour without changing what is measured? Measured
on grpo-v3-run1's Pop evals (pin 7e929f2ea, fork 1839b5071, 14 slots, 16 running, mem-fraction 0.8). Tool:
`scripts/neuralese_data/pop_held_profile.py` (`sglang ACTOR.log`, `episodes HELD_RUN_DIR`).

## Breakdown (step 8: 512 episodes, 16,004 s measure + 367 s serve)

| where | seconds | share | source |
|---|---|---|---|
| decode | ~14,880 | 93% | fit of the 14,815 decode-log intervals (40 steps each) on steps per batch size and the prefill work in them |
| prefill | ~1,070 | 7% | 5.58M new tokens at ~5.6k tok/s; prefix hit 93.4%; per-batch overhead fits at ~0 |
| harness/CPU between calls | ~0.4% of episode slot time | | traces: model calls cover 99.1% of an episode's wall time |
| serve start (rsync 265 s + engine ~95 s) | 367 | 2.2% | held-pop.log `copy_seconds`, `ready_seconds` |

- **Decode is the bottleneck, and the batch is small.** The step costs 18.5 ms at 1 running sequence and 24.7 ms at 14
  (fit), so tokens/s scale almost linearly with running sequences: 60 tok/s at 1, 428 at 8, 622 at 14 (p75 of the
  log's gen throughput). 14 slots give 13-14 running sequences (14: 69% of decode logs, 13: 14%). SGLang pads those to
  its bs-16 CUDA graph (captured sizes 1, 2, 4, 8, 12, 16), so 2-3 of 16 graph rows do nothing. `#queue-req` was 0 in
  every decode log, which means the engine was never full.
- **The KV pool is not the limit.** The 14-slot rule came from a ~180k-token FP8 estimate. The fork's hybrid SWA pool
  holds 318,297 full-layer tokens at mem-fraction 0.8 (~280k at the pin's 0.77) plus 31.8k SWA tokens. At 14 running,
  full usage was p50 0.35, p90 0.49, p99 0.63 and max 0.71 (8.6k tokens per running sequence at p50, 14.4k at p99).
  SWA usage was ≤ 0.31 and mamba usage 0.33 (42 of 128 states). There were no retractions.
- **Episodes.** Wall time p10 67 s, p50 311 s, p90 857 s, p99 1,596 s, max 2,631 s. Ran-out (`budget_exhaustion`,
  40 calls) episodes are 9% of episodes and 21% of slot time (step 7: 11% and 25%). That is what is measured, so it
  stays.
- **Tail.** Sample-major order never leaves slots idle at the boundary between samples: sample 1 starts while
  sample 0 is still finishing (step 8: sample 0's last finish at 8,484 s, sample 1's first start at 6,838 s). The
  waste is at the end. After the last dispatch, 1,356 s (step 8) and 1,064 s (step 7) went by with fewer than 14
  episodes running, and 4.2% of the span had a single episode running (step 8's last 16 episodes took 23.7 of 266.7
  min). Order was the sealed queue's: a long task dispatched late sets the end.
- **Between evaluations** (grpo `--held-remote pop`): the 60 s loop wait, the rsync of the 3.46 GB checkpoint (265 s)
  and the engine start (~95 s, of which CUDA graph capture is 24 s). The rest of the ~660 s gap from step 8 to step 10
  was the step-10 checkpoint not existing yet.

## Changes

1. **Longest-expected-first dispatch within each sample** (`held_eval.Dispatch`, the default; `--dispatch sealed`
   opts out; 770a08dab). Order stays sample-major, so a partial run stays balanced. Within a sample the next slot gets
   the task with the longest known episode: its episodes in this run (finished: wall time; running: time so far),
   else `--duration-prior` (earlier `result.json` files of held-v3, any policy), else 0. Ties keep the sealed order.
   The task, seed, budgets and engine of every episode are unchanged; only which episodes share a batch changes, as
   any slot timing already does. A resumed run seeds its estimates from its finished journals. grpo's Pop loop passes
   the start policy's result as the prior. `pop-held.sh` takes `PRIOR=<result.json>`.
   - Expected gain, from replaying run1's six Pop evals on 14 slots with their measured episode wall times (the replay
     reproduces step 8's span: 15,976 vs 15,999 s). Sealed order: 16,624 / 18,505 / 17,289 / 16,960 / 16,571 / 15,976 s.
     With an other-evals prior: 15,913 / 18,021 / 16,623 / 15,803 / 15,890 / 15,025 s, or **-500 to -1,160 s
     (mean -775 s, ~4.6%)**, within 0.3% of the oracle order. The in-run estimate alone (no prior) gives a mean of
     -540 s at 2 samples (one eval +81 s). On a 4-sample proxy (each eval's samples twice, out of ~31-36k s), the
     prior gives -490 to -1,160 s and the in-run estimate alone -740 to -1,360 s.
2. **20 slots and SGLang max-running 20 on Pop** (from 14 and 16), at the new pin only. This changes the engine.
   `max_running` is now attested in `rl_policy_binding.serving_engine` and in `held_eval.ENGINE_FIELDS` (770a08dab), so
   `compare` refuses to pair a 16-cap result with a 20-cap one. grpo's Pop base check moves a base measured under
   another cap aside. `pop_rl_actor start` reuses a running actor only under the same engine settings (a9c32c78a).
   Numerics: the decode graph for bs 20 exists (captured 1, 2, 4, 8, 12, 16, 20). Batch composition differs, as it
   already does between any two runs.
   - Pool (at 0.77, ~280k tokens): the 14-slot per-sequence usage scaled to ~19 running puts full usage at p50 ~0.59,
     with ~2% of time above 0.95 (24 slots: ~9%, so 24 is not chosen). Mamba states 8 x 20 = 160 (~0.05 GB);
     graph memory +~0.04 GB.
   - Expected gain: decode tokens/s at ~19 running versus 13.3 is about **+25-30%** (step time 18.5 ms + ~0.45 ms per
     sequence, padded to the bs-20 graph), so a 2-sample eval is ~16.0k s -> ~12.5k s before the order change.
     Measured in the smoke: **+28%** (below).
   - Defaults (grpo `--held-remote-slots 20 --pop-max-running 20`, `pop_rl_actor.POP_MAX_RUNNING = 20`,
     `ops/grpo-v4/pop-held.sh` SLOTS/MAX_RUNNING 20): `SLOTS=14 MAX_RUNNING=16` / the flags restore the old engine.
   - DGX side: 6 more collector slots cost ~1.6-2.5 GB (0.26-0.4 GB per slot). `pop-held.sh` claims 11 GB instead of 8.
     grpo-v4's unit holds its held collectors: +~2 GB against the 97 GiB sizing.
3. **Not changed:** chunked prefill (2048), overlap schedule (off for correctness, serve_native_checkpoint.py), mixed
   chunks, mem-fraction (0.77 already decided for OOM headroom), KV dtype, the case budget.

## Smoke (2026-11-17, Pop, after run1's step-10 verdict eval)

Engine: fork dbc51ddb72 (moved with `fork-checks.sh pop-move`; `pop-conv`: POP SM89 CONV BITWISE OK, EXIT 0), main
ee22da0b8, policy grpo-v3-run1 step 10 (9cd7aa1a), mem-fraction 0.77, FP8 KV, max-running 20, 20 slots, longest-first
dispatch with step 8's result as the prior. 50 min (`timeout 3000`), 80 episodes finished. Files:
`ops/grpo-v4/pop-smoke-20/` (`actor.log`, `actor-baseline-14.log`, `measure.log`, `run/`). The baseline is the same
policy measured by the step-10 verdict eval at 14 slots / 16 running / 0.8 on fork 1839b5071, sealed order.

| window (pop_held_profile.py sglang) | gen tok/s | mean running | decode / prefill s (fit) | prefix hit | full pool p50 / p99 / max | SWA max |
|---|---|---|---|---|---|---|
| 20/20, smoke, 12:48-13:38 | **605** | 19.2 | 2,699 / 290 | 91.2% | 0.59 / 0.90 / 0.93 (of 280,490) | 0.48 |
| 14/16, step-10 eval, first 50 min (08:42-09:32) | 472 | 13.6 | 2,755 / 186 | 93.7% | 0.39 / 0.56 / 0.59 (of 318,297) | 0.30 |
| 14/16, step-10 eval resumed, 09:32-12:40 | 452 | 12.2 | 10,446 / 835 | 93.2% | 0.31 / 0.58 / 0.66 | 0.30 |

- **+28% generated tokens/s** in the matched first 50 minutes (605 vs 472). The smoke's longest-first order put the long
  tasks first, which is the harder half for the pool and per-step attention, so this is not flattering. Decode step
  ~26.5 ms at 20 running (24.5-24.8 ms at 14). There were no queued requests and no retractions, and the prefix hit
  fell by 2.5 pts (more prefill: 290 vs 186 s per 50 min).
- **Episodes per hour** are not comparable over 50 minutes: longest-first starts the longest tasks first (80
  finished, 96/h). At ~14.4k generated tokens per episode (step 8: 7.37M / 512), 605 tok/s is ~151 episodes/h, against
  ~115/h for step 8's whole eval.
- **Memory on Pop:** GPU peak 7,195 of 8,188 MiB (the pool and graphs are fixed at launch; 1.00 GB was free after
  capturing graphs 1-20, 0.25 GB of graphs). Host peak 6.6 GB used, at least 7.2 GB available. Engine ready 81 s after
  launch (checkpoint already on Pop).
- **Pool:** the max of 0.93 of 280k shows that 20 is the right cap at 0.77. 24 would queue or evict in the long-task
  phase.
- **Measure hang found:** after `timeout` sent SIGTERM at 13:38:23, all 20 episode processes and the shared node
  collectors exited within a second (worker logs end in `KeyboardInterrupt: queue stopped`, collector sockets gone at
  13:38:24). The `held_eval measure` parent never wrote `journal.jsonl`, so it never reached `finalize` and was still
  inside `collect` when the supervisor stopped the unit 54 min later. **Root cause not identified:** the process was
  SIGKILLed before its stacks could be read, and a local SIGTERM repro of `collect` (40 entries, 6 slots, group
  signal through `timeout`) returned in 6 s. Fix (every held measurement runs this code: grpo's training-actor held
  eval, grpo's Pop loop, `pop-held.sh`, `held_eval measure`): after a stop, `collect` sends SIGTERM to running
  episodes, SIGKILL to any still alive at half of `STOP_DRAIN_SECONDS` (120 s), and returns at the full drain whatever
  is still blocked, logging each stuck slot thread's stack (`held eval stop`, `stuck_slots`, `stacks`). `run()`
  registers `faulthandler` on SIGUSR1 (`kill -USR1 PID` prints every thread's stack). The next occurrence shows the
  blocking frame, and the partial result is written regardless.

## Combined expectation

Per 2-sample eval: ~16.0k s -> ~12.5k s (slots) -> ~11.9k s (order), about -25%. Per 4-sample eval: ~9 h -> ~6.8 h.
A 2-sample read then lags training by ~3.5 h instead of 4.7 h.

## Not checked or not done

- **Checkpoint prefetch** (rsync to Pop while the previous eval runs, ~265 s per eval, ~2%): not built.
  `serve-checkpoint` already skips the copy when the file is on Pop. For the pin's base, the `start` path prepares a
  view instead.
- **Prefill CUDA graphs**: disabled by SGLang for LFM3 ("some layers do not apply Standard GQA"). Per-batch prefill
  overhead fit at ~0, so there is little to gain.
- **Mixed chunked prefill, overlap scheduler**: not tried (overlap gave wrong-distribution tokens on the GB10;
  mixed chunks are untested with the fork's conv kernels).
- **A full 512-episode eval at 20/20 was not run.** The +28% is from one 50-minute window. The full-eval expectation
  (~12.5k s before the order gain) is an extrapolation, and the pin's base re-measure will be its first measurement.
- **Root cause of the measure hang** (see Smoke): bounded and instrumented, not explained.
- **Ran-out episodes** (21-25% of slot time) are part of the measurement and were left alone.
