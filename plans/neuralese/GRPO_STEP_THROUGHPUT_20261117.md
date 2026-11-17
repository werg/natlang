# GRPO step wall time: where it goes and what changes (2026-11-17)

Question: one grpo-v3-run1 step (pin 7e929f2ea; 48 groups x 8 episodes) took ~3.25 h. Where does it go, and which
changes cut it without changing what is trained (objective, sampled data, numerics within BF16 tolerance)?

## Where the time goes (run1 `train.jsonl`, steps 1-10)

Per step (median-ish of steps 5-10, run1 sizing: 32 slots, max-running 48, KV 6 GB, `--learner-phase-slots 4`,
`--activation-gb 4`):

| phase | s | share | what runs on the GPU |
|---|---|---|---|
| wait_groups (collection) | 4,650-6,400 | ~46% | actor decode at 26-32 running, learner idle |
| lower | 45-80 | <1% | CPU |
| reference_forward | 790-1,390 | ~10% | learner forward + 4 actor slots |
| policy_forward_backward | 3,300-5,700 | ~40% | learner fwd/recompute/bwd + 4 actor slots |
| decision fwd/bwd, optimizer, probe | 100-340 | ~2% | learner |

Collection is bound by the number of slots, not the KV pool or prefill:
- The actor's decode lines (`actor.log`, 240k lines): with 25-32 running and the learner idle every sequence decodes at
  33-35 tok/s (a ~30 ms step: weights ~3.4 GB read, the FP8 KV of ~5.7k cached tokens per sequence, ~7 ms of scheduler
  CPU with the overlap scheduler off). KV usage was 0.15-0.19 of the 6 GB pool; radix prefix hits 94% of prompt
  tokens (43.6M prefilled vs 699M cached over the run), so prefill is ~5% of actor GPU time.
- The same engine at 64 running (the verdict held-eval actor, 2026-11-17 09:35, DGX otherwise nearly idle): ~1,780
  tok/s vs ~1,080 at 32. Decode is weight-read bound, so doubling the batch adds only the KV reads.

The learner phase shares the GPU with the 4 learner-phase slots (plus the in-flight episodes draining):
- Actor decode with <= 23 running fell to 14-16 tok/s per sequence (vs 34 alone): the GB10 time-slices the two
  processes. A 4-sequence decode step still reads all weights, so those slots took about a quarter of the GPU's time
  for ~130-250 tok/s, about an eighth of what the same GPU time yields at full batch.
- Microbenchmark (`memory_ledger` unit tput-mps-bench-1): a BF16 GEMM loop (83 TFLOPS alone) and a decode-like
  weight-read loop (209 GB/s alone) run together reach 42% + 44% of their solo rates (86% total: GPU work is conserved,
  minus ~14% for context switches). CUDA MPS (private pipe dir) gave the same split (43% + 43%), and capping the decode
  client at 40% of SMs made both slower (29% + 36%). So on the GB10 overlapping collection and learning buys nothing
  beyond filling idle gaps; the levers are doing each kind of work at its efficient batch and doing less work.
- The verdict actor at 64 running dropped from ~1,780 to ~840 tok/s while a learner profile ran beside it: the same
  50/50 split on the real engines.

## Changes

### 1. No new episodes while the learner computes (`--learner-phase-slots` default 8 -> 0)

What: the collector's slots take no new episode during the learner phase; in-flight episodes finish (they always did).
`train.jsonl` now logs `episodes_in_flight` at the learner's start and at the policy pass's start, so the memory the
learner shares with draining episodes is visible.
Why: the GB10 time-slices the actor and the learner (above). A 4-sequence decode step still reads all weights, so the
learner-phase slots took about a quarter of the GPU's time to produce 127-160 tok/s; the same GPU time spent at full
batch in the collection phase yields 6-8x the tokens.
Expected gain per run1 step: the in-flight episodes still drain into the learner phase (run1 step 10: 31 running at
the learner's start, 12 after 5 min, 8-9 for the next ~40 min of which 4 were the learner-phase slots, 3-5 after
that), so a few-sequence decode keeps a share for roughly the first 45 min instead of the whole ~105 min. Learner
phase about -600 to -900 s, collection +~0.5M tokens (+~350-450 s). Net about -300 to -500 s (3-4%). Not a treatment change: what the
learner trains on is the groups the collector finishes; the staleness mix shifts slightly toward fresh groups.
Opt out: `--learner-phase-slots N`.

### 2. Nucleus NLL forward on bf16 logits with linear_nll's kernel (`lfm3_fast_train`)

What: per 2048-row head chunk the policy pass's `_LinearNLLNucleus.forward` wrote the logits as an fp32 copy (1 GB)
and ran eager log-sum-exp (two passes), gather and top-k over it. Now the log-sum-exp and NLL come from the compiled
`_ce_forward` (the kernel `linear_nll` and the KL reference pass use) on the bf16 logits, and top-k reads the bf16
logits (`values.float()` is exact). The wide paths (rows whose nucleus reaches past the top 1024) upcast only those rows.
Measured (`scratchpad head/nuc.py`, 2048 rows x 128k vocab, peaked logits, GPU shared with an eval actor and a learner
profile): 223 ms -> 74 ms per chunk (the matmul alone 38 ms). Over run1's ~4.5M loss tokens per step: ~2,200 chunks,
-150 to -330 s per step (shared vs solo GPU). With flat logits most rows take the wide path and the gain shrinks
(669 -> 512 ms per chunk).
On real chains (3 run1 groups g520-522: 208 chains, 1.96M chain tokens, 1.03M computed after prefix sharing, 529k loss
tokens; `--activation-gb 4`; BF16 weights, same contention from the 64-slot verdict actor in both runs): policy pass
525.9 s -> 501.4 s (-4.7%), reference pass 125.4 -> 123.4 s (unchanged code; noise). Scaled to run1's ~4.8k s policy
phase: about -220 s per step.
Parity: the policy NLL now equals `linear_nll` bit for bit (test
`test_nucleus_nll_on_bf16_logits_matches_the_reference_pass_bitwise`): at equal weights the KL term's `d` is exactly 0,
where before the eager and compiled log-sum-exp differed by fp32 rounding (~2-4e-6 nats). Cut, nucleus log-sum-exp
and `outside` are bitwise the fp32 path's on all three top-k/wide-k settings; the backward is unchanged. On the real
chains `max |ref logp - policy logp|` at equal weights went from 3.8e-6 to 0.0. Old vs new head on 4096 rows (scratchpad
`head/oldnew.py`): dh relative L2 2.4e-7 (peaked logits) and 1.3e-5 (flat), dW 6.5e-8 / 6.3e-6, `outside` identical;
the nucleus NLL itself identical on peaked rows and within 3.8e-4 on flat rows, where an ulp of the log-sum-exp moves a
near-tie at the nucleus edge (the old code had the same sensitivity, against the other kernel). The profile harness's
summed BF16 `.grad` over 208 chains (no FP32 master accumulation) had grad norm 2.6824 (old) vs 2.6917 (new): within
the harness's own BF16 accumulation rounding (each add rounds at 2^-9 relative); the learner accumulates into FP32
masters. Not checked: an FP32-master gradient comparison on the learner bench (the GPU went to the verdict eval).


### 3. Recompute (`--activation-gb`): measured, not changed

Same chains: `--activation-gb 4` 525.9 s policy, `12` 495.5 s (-5.8%) for +7.5 GB of learner peak (19.9 -> 27.4 GB).
Most of the policy pass's FLOPs are the GEMMs it must do anyway; the recompute share at 4 GB is small because the long
leaves keep only a few layers either way and prefixes are always recomputed. Per GB, slots buy far more (below), so the
budget stays 4.

### 4. AdamW moments on disk between optimizer steps (`MomentOffload`, default on; `--keep-moments-resident` opts out)

What: exp_avg and exp_avg_sq (13.6 GB FP32) are read only by `optimizer.step` and `moment_cosine` right before it,
yet stayed resident through collection and the reference and policy passes, where they were 13.6 of the learner's
47-51 GB peak. After each step (and after the state save that reads them) a background thread streams them to
`WORK/adam-moments.offload` (NVMe) and frees their storage (`untyped_storage().resize_(0)`: the optimizer keeps the
same tensor objects); before the next optimizer step they are reallocated and read back. A failed write leaves them
resident (`moment_offload.put_error`). Both this and `StateSaver` now stream through one helper that flushes and drops
the written page cache every 512 MB (the cache is charged to the unit and a CUDA allocation does not reclaim it; the
per-step FP32 state save had been leaving up to ~21 GB of it).
Parity: bit-exact by construction; tests: round trip of NaN-containing tensors through several staging chunks (bitwise),
storage freed after `put`, the saver's `wait` first, a failed write keeps them resident, StateSaver's file loads back
equal. Learner bench (`tput-bench-pair-2`: run1 group g31 replayed 3 steps, offload vs `--keep-moments-resident`, same
seed): `bf16_changed_step_fraction` 0.001083 / 0.001083 (step 2) and 0.001622 / 0.001621 (step 3), `kl_mean` 5.04e-4 /
5.02e-4, `grad_moment_cosine` 0.9981 / 0.9980 (the read-back moments point where the resident ones do); grad norms
differ at the 3e-4 run-to-run level already present at step 1 (3.0579 vs 3.0589, before any offload).
Cost: put 46-132 s in the background (the shared NVMe wrote at ~100 MB/s while the verdict eval ran), get 15-18 s on
the critical path (~0.8 GB/s), once per step. `train.jsonl` logs `moment_offload` {put_s, cuda_reserved_after_put_gb,
get_wait_s, get_s}. Gain: memory, -13.6 GB at every phase but the optimizer step, which pays for the slots below.
The file (13.6 GB) lives in WORK and goes with the run's directory.

### 5. Sizing for grpo-v4 (`ops/grpo-v4/launch-grpo.sh`): 64 slots, 64 running, LPS 0

Collection is slot-bound. 64 running decode ~1.6-1.8k tok/s vs ~1.08k at 32 (the verdict actor's 64-running
collection at 2.7-4.8k cached tokens per sequence; run1's were ~5.7k, so expect the lower end). Memory: run1-r1's
107.4 GiB peak, minus the pin's fixes (~8.6), minus the moments (13.6), plus 32 more slots (0.26 GB each, ~8.3 while
running; they drain during the learner phase) gives ~87-92 GiB under the 97 budget. The KV pool (6 GB) was 15-19% used
at 32 running. Expected: collection 5,400 s -> ~3,700-4,000 s per 48 groups (-1,400 to -1,700 s).
Fallback if the step-1/2 ledger peak passes 97 GiB: `SLOTS=48` (sizing only). `episodes_in_flight` and
`moment_offload.cuda_reserved_after_put_gb` in `train.jsonl` show what the peak is made of.

## Expected step (run1's 48 groups, ~6.1M generated tokens)

| | run1 | grpo-v4 (expected) |
|---|---|---|
| collection | ~5,400 s | ~3,700-4,000 s (64 slots) + ~350-450 s (no learner-phase slots) |
| reference + policy + decisions | ~6,300 s | ~5,300-5,500 s (LPS 0: -600-900; nucleus: -220) |
| optimizer, probe, lowering, moments | ~140 s | ~160 s |
| step | ~11,700 s (3.25 h) | ~9,500-10,000 s (2.6-2.8 h), -15 to -20% |

## What was not checked / not done, and the next levers

- Solo (unshared) learner throughput: every learner measurement today shared the GPU with the 64-running verdict
  eval actor, so absolute tok/s are about half of solo; the A/B pairs ran back to back under the same load.
- An FP32-master gradient comparison of the old vs new nucleus forward on real chains (the head-level comparison and
  the bitwise reference/policy agreement were checked; the GPU went to the verdict eval).
- The `--slots 64` peak itself: estimated, not measured; the launch's step-1/2 peak check decides (fallback 48).
- Not changed, measured: `--activation-gb` (above), MPS (above), streaming/pipelined learning (no gain: GPU work is
  conserved under time-slicing).
- **Next actor lever: SGLang's overlap scheduler.** With it off (`--sglang-overlap-schedule` false: on the GB10 with CUDA
  graphs it once sampled 2-6% of RL turns from a wrong distribution, 2026-11-03) each decode step has ~7 ms of scheduler
  CPU in series with ~22-30 ms of GPU work: ~20-25% of collection time. Fixing that fork defect (a correctness hunt,
  then parity on every engine) is the largest remaining collection gain.
- **More slots:** 96 running would decode ~2k tok/s; it needs ~8 GB more than 64 (collector node processes ~0.26 GB
  per slot are the main per-slot cost; reducing them is the lever).
- Episodes in flight at the learner's start drain for ~45 min at a few sequences (run1), each step taking a full
  weight read: a learner-start drain wait or request holding in the actor front could recover part of that share.
- Objective-side cases (not changed here): none argued for; the KL reference pass (~20% of the learner) is required by
  the k2 term as defined.

