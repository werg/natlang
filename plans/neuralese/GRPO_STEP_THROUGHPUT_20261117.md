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

### 6. Learner kernel profile (torch.profiler, kernel classes)

One real run1 group (g521: 24 chains, 494k chain tokens, 424k computed, 100k loss tokens, 10 chains with imported
state), reference + policy pass at `--activation-gb 4`, new nucleus code, GPU shared with the verdict eval actor (so
kernel durations include time-slice preemption; shares, not rates, are the reading):

| class | GPU s | share |
|---|---|---|
| GEMMs (nvjet sm121 tiles; cutlass sm80 fp32-out dW of the head ~9 s) | 146.3 | 59% |
| flash attention (FA2 varlen fwd 22.2 + bwd 18.4) | 40.6 | 17% |
| inductor Triton (norms, SwiGLU, RoPE, gates, the width-7 conv, CE/nucleus) | 50.5 | 21% |
| eager aten elementwise (grad adds into the masters, etc.) | 7.1 | 3% |
| top-k (nucleus) | 0.8 | 0.3% |
| wall - GPU busy (Python, launches, syncs) | 6.3 of 252 s | 2.5% |

Reading: the learner is kernel-bound with almost no host gaps, so CUDA graphs and larger micro-batches have nothing to
win; the GPU's time is ~59% GEMM, ~21% fused elementwise (2026-11-12: inductor's kernels already run at ~85% of the
GB10's bandwidth; fusing them into the GEMMs is not available on sm_121) and ~17% attention. The low tok/s per model
FLOP in run1 is mostly the shared GPU (the actor's time slices), not kernel inefficiency; a solo measurement of the
same 3 groups is queued for when the GPU is free. Candidates left: a faster attention kernel for the 6 full-attention
layers on sm_121 (FA2 is what `torch.nn.attention.varlen` offers here), and fewer recomputed layers (measured above:
small).

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


## Prefix tree packed: levels and tree-mask (2026-11-17)

The shared learner prefix tree (`lfm3_fast_train.run_prefix_tree`, both learners) ran every prefix node and every leaf
as its own forward and backward: on `sft-v3-run1` steps the leaves are short (median 360-1,500 tokens) continuations of
~5k-token histories, so a step was 60-280 small forwards and as many backwards (backward 70-80% of the time).
Two packed modes now exist (`--prefix-tree-mode`, both learners):

- `levels` (992237193): each tree level's nodes, then the leaves in tree order, run as varlen batches of
  `--pack-tokens` that continue their prefixes' cached keys, values and conv inputs (`cu_seqlens_k` longer than
  `cu_seqlens_q`). A node's state is held as FP32 leaves (`Segment`), so all continuations' cotangents add up in FP32
  and each prefix batch backpropagates once, deepest level first.
- `tree-mask` (e9037c976, a601339f6, ba661f8cf; owner directive): each batch of whole subtrees is one flattened tensor,
  one forward and one backward. `hidden(parents=...)`: RoPE positions along each token's own path, the conv's taps
  gathered by a per-token path-predecessor index (a branch's first tokens read the parent's last ones, never a
  sibling's), attention under the path mask (ancestor-or-self, causal along the path, the sliding window along the
  path) as a flex-attention `BlockMask` built from the tree's structure (no T x T evaluation; `create_block_mask`
  recompiled per sequence length, 2-17 s each). Imports and pass-through producer documents ride in the same batch.
  Batches (`tree_units`) are whole subtrees up to a cap; an oversize subtree splits among its children, each batch with
  its own copy of the path. The default cap (`plan_tree_batches`) scores prefix copies against layer recompute.

Bug found on the way: the sliding-window `mask_mod` took the window as a default argument; flex tells a mask_mod from a
score_mod by arity, so the sliding layers' mask was wrong on CUDA (0.73 relative error vs SDPA with the dense mask).
Fixed (a601339f6) with a CUDA test (flex == dense mask in FP32 at 1e-4).

Measured (real LFM3-1.7B, `sft-v3-run1` step groups, exp0 `step_backward`, `--activation-gb 6`, pack 3050, FP32
master accumulation as the trainer, GPU time-sliced with other jobs; arms of one row ran back to back in one process
after a compile warmup; scratchpad `pack/pbench.py`). Step seconds (computed tok/s):

| group (chains, chain tok, computed tok) | whole chains | per-node (8364ef22e) | levels | tree-mask |
|---|---|---|---|---|
| 1 (72, 447k, 64k) | 211 s (2,113) | 107 s (595) | 45 s (1,402) | 47 s (1,343) |
| 5 (39, 483k, 151k) | 234 s (2,064) | 114 s (1,326) | 82 s (1,848) | 104 s (1,455) |
| 4 (240, 1.46M, 255k) | ~690 s (old code) | 208 s (1,228) | 130 s (1,964) | 209 s (1,220) |

Levels: 1.6-2.4x the per-node path, 2.9-5.3x whole chains per step. Computed tok/s reaches 0.9-0.95x whole chains on the
larger groups (the leaves attend to ~5k-token histories, extra attention work per computed token). Tree-mask does not
beat levels: under a fixed activation budget a one-forward subtree batch recomputes ~23 of 24 layers on every token
(`plan_tree_batches` on these groups: 22.5-23.6 mean recomputed layers, +6-44% prefix copies at ACT 4-6, 0-5% at
ACT 16), while levels recomputes only the prefix rows and keeps its leaf batches resident. Default stays `levels`;
`tree-mask` is opt-in. Saved activations per token and layer are the same on both paths (conv 85.0 KB, attention
80.2 KB, `pack/layermem.py`); tree-mask's per-batch CUDA peak on group 5 was 9.3-13.0 GB allocated without masters.

Gradients vs whole chains (relative L2 of FP32-master gradients, embedding / layer 0 w1 / layer 2 k_proj; layers 22-23
at 1-2% for every path): group 1 per-node 5.2/6.1/5.5%, levels 4.3/5.0/4.6%, tree-mask 7.4/8.4/7.5%; group 5 per-node
8.5/12.7/14.6%, levels 6.2/8.2/8.8%, tree-mask 9.0/12.8/15.0%. Whole chains are not the reference to trust on long
chains: on the 24.8k-token chain (`pack/diag2.py`, FP32 masters) whole vs a second whole run 1.2% (flash dQ atomics),
whole vs stock HF 19.9%, levels vs stock 9.9%, tree-mask vs stock 8.4%, tree-mask vs levels 9.1%: solving the pairwise
distances for independent errors puts the fast whole path at ~19-22% and stock/levels/tree-mask at 5-7%. The earlier
"tree gradient 1.4x further from whole" was the whole path's own noise. An FP32 reference (stock HF in FP32) of the same chain is queued to confirm which path is closest to exact.

GRPO learner bench (grpo-v3-run1 group 31 replayed 2 steps from the base, `--activation-gb 4`, KL on, `packbench/`):
levels: step 2 reference pass 9.3 s, policy pass 10.4 s (18.3k computed of 53.5k chain tokens; 1 prefix batch, 8 leaf
batches), `kl_mean` exactly 0 at step 1 (reference and policy passes round identically with packed batches), policy
peak 35.8 GB allocated at step 2 (22.8 at step 1). The per-node, tree-mask and off arms are queued for a free window
(each needs ~40 GB beside grpo-v4/SFT; the ledger stopped two attempts).

Not checked yet: the GRPO arms beyond levels; the FP32 truth; a solo-GPU (unshared) measurement; levels' own CUDA peak
exceeds `--activation-gb` by several GB on group 5 (24.2 GB allocated with 10.2 GB static at ACT 6: the history keys
and values a continuation batch gathers are only roughly costed in `effective`).


## SGLang overlap scheduler defect: investigation (open, 2026-11-17)

Observation (LFM3_RL_ACTOR_ENGINE_20261028.md, 2026-11-03): on the DGX GB10 with the overlapped scheduler, CUDA-graph
decode with padding and the flashinfer backend, 2-6% of real RL turns held single tokens whose recorded probability
was high (e.g. 0.53) while the learner gave them -24 nats, and the continuation returned to the learner's law (a
one-step mixup). Each of these alone removed it on 64 turns: no CUDA graphs, `--disable-overlap-schedule`, the triton
attention backend, `--disable-cuda-graph-padding`. The cloud RTX PRO 6000 (flashinfer, overlap, padding) had 0/424.
Since then the fork moved (a06adeda40 padded-row conv slot fix was already in; hybrid SWA KV, conv-state tracking at
input end, extra_buffer mamba radix default 2026-11-12, path handoff, bounded conv prefill), so whether the defect
still reproduces at dbc51ddb72 is itself unmeasured.

Inventory of the mechanism (the sampled token and its recorded law):
- Engines: the SGLang actor (training actor, DGX held engine, Pop held engine; all through
  `serve_native_checkpoint.py --engine sglang`, overlap off by default, attested as `overlap_schedule` in the engine
  format and in `held_eval.ENGINE_FIELDS`); other SGLang launchers (`serve/sglang_bench.py`, which runs upstream's
  default, overlap on; the parity checks, `check_sglang_hybrid_swa.py` with overlap off). The managed llama-server,
  the in-page wllama build and the Python native engine have no overlapped scheduler (their decode loops sample and
  then build the next step synchronously), so the mechanism does not exist there; their sampled-law capture is
  separate code.
- Consumers: `lower_rollout_turns.episode_turns` (capture -> sampled ids and logps), the GRPO learner (`rho`, the
  tail floor and the nucleus law use the recorded logp), `held_eval` (outcomes only), `native_replay.actor_engine_guard`
  (the older guard that rejected such episodes).

Ruled out (checked): **flashinfer's decode plan staging buffer.** `DecodePlan` writes the split-KV plan (request
indices, KV tile indices, output indptr) into a page-locked host buffer and enqueues `cudaMemcpyAsync` with no host
fence, and SGLang's CUDA-graph wrappers plan through `fast_decode_plan` with a host indptr (no implicit sync), so a
next plan written before the previous copy ran would give a batch another batch's tile map: exactly a one-step
wrong-context distribution, and split-KV is common on the GB10's 48 SMs. Repro (`scratchpad ovl/plan_race.py`, fork
venv flashinfer 0.7.2, sm_121): plan A queued behind a 50M-cycle `_sleep` on the stream, plan B written at once,
then A run: 0 wrong outputs in 20 trials, both CUDA-core and tensor-core decode, while the positive control (A's
device tables with B's host plan) differs by 0.16 from the reference. The plan copy is small and the driver evidently
captures it at enqueue, so this race does not occur on this stack.
Checked, not a cause by construction: padded rows' request slot (ReqToTokenPool reserves row 0 for CUDA-graph padding,
so the fork's per-request-slot producer flags and adapter indices written by padded rows land in the reserved row);
padded rows' conv state (a06adeda40: slot 0, the reserved padding slot).
Not checked yet: the mamba extra_buffer tracking under overlap (`track_mamba_states_if_needed` after the fork's own
`natlang_conv_decode`, which replaced the CuTe kernel the upstream overlap path was validated with); the future-token
map with padded batches; the natlang sampler capture (it can only mislabel the recorded logp, not change the drawn
token, so it does not explain the observation); request-slot and mamba-slot reuse across the one-batch overlap lag
for requests that finish (upstream delays frees under overlap; the fork's conv-state tracking at input end,
33230160ba, adds a write the delay may not cover).
Next (needs the GPU, after the verdict): rerun the 2026-11-03 detection at dbc51ddb72 (`serve/sglang_bench.py
serve-measure` with overlap on, c=32 and 64, then `rescore` with the learner arithmetic; count turns with
|dlogp| > 5) against `--disable-overlap-schedule`. If it reproduces, bisect with one switch per run: CuTe conv decode
instead of `natlang_conv_decode`, `--mamba-radix-cache-strategy no_buffer` (upstream refuses overlap there),
flashinfer CUDA-core vs tensor-core decode, adapters/producer features absent. Overlap stays off until the cause is
found and fixed.

### Detection rerun at fork dbc51ddb72 (2026-11-17 14:10-14:45)

`serve/sglang_bench.py all` (the 2026-11-03 harness, workload-2: 78 real pilot turns) with the fork clone at
dbc51ddb72, base 5ff07638 (run1's export), the production engine settings (FP8 KV, `extra_buffer` mamba radix cache,
hybrid SWA ratio 0.1, 64 running, CUDA graphs with padding, flashinfer, vendor projection), sampling c=32 (64
requests) and c=64 (128 requests), rescored with the learner's arithmetic (native engine). Outputs:
`ops/overlap-hunt/{on,off}/`.

| arm | turns | tokens | mean abs dlogp | p99 | max | turns > 5 nats | mean turn k3 | out tok/s c32 / c64 |
|---|---|---|---|---|---|---|---|---|
| overlap on | 192 | 59,314 | 0.0285 | 0.261 | 1.76 | 0 | 0.0026 | 549 / 1,137 |
| overlap off | 192 | 53,622 | 0.0306 | 0.257 | 2.97 | 0 | 0.0026 | 506 / 1,020 |

The 2026-11-03 positive run (fork a06adeda40, BF16 KV, full KV for every layer, no extra_buffer, full67, 32 running)
had 2 of 80 turns at ~16 nats; at that rate 0 of 192 has a probability of about 1%. So the defect does not reproduce
in the production configuration at dbc51ddb72. Not yet established: whether it is fixed or only masked by the changed
configuration (KV dtype and layout, mamba radix strategy) or by the base policy instead of full67. Overlap's measured
gain here is +8-11% output tok/s on these short (prefill-heavy) turns, below the ~20-25% estimated from the scheduler's
CPU share; long-reply RL collection should sit between the two.
Next: an overlap-on arm in the 11-03 configuration at dbc51ddb72 (BF16 KV, `--disable-hybrid-swa-memory`, 32
running); if clean, a ~500-turn overlap-on sample in the production configuration; then overlap on by default at the
next pin with `--sglang-overlap-schedule` as the switch, with the learner's per-token rho statistics as the guard.


## Learner phase per-token rate vs SFT (2026-11-20, grpo-v4-run2 steps 1-40)

Question: step 40's policy pass ran 1.8k computed tok/s against sft-v5's ~4.8k on the same `lfm3_fast_train`.

- **Step 40 is not the learner's rate.** Clean steps run the policy pass at 3.8-4.4k tok/s and the reference pass at
  14-16.7k (`forward_tokens / phase_s`). The slow steps (policy 1.8-2.6k: 12, 23, 34, 36, 39, 40; reference 7-9k: 30,
  33) each overlap a GPU diagnostic unit in the ledger's events (step 40's policy pass 11:10-11:50 with `hz3-phx`
  10:34-12:15; step 39 with `hz3-push*/kl37*/phase`; 36 with `hz2-haz35`; 33-34 with `hz2-haz31b`; 30 with `hz2-end29`):
  the GB10 time-slices processes (~45/45, above), so the learner ran at about half rate. Those steps also swapped
  0.3-1.4M pages in (the diagnostics' memory). GPU work is conserved, so the diagnostics' cost shows up in whichever
  phase they overlap; nothing in the learner to fix.
- **The clean 1.2x to SFT is the workload.** Per computed token GRPO trains 65% loss tokens (2.77M of 4.25M at step 40)
  against SFT's 11%: each runs the tied 128k-vocabulary head 4 times (forward, backward's logit recompute, dh, dW; 0.52
  GFLOP each against ~2.9 for the body's forward) plus the nucleus top-k, ~11% of the pass. Leaves are long (median chain
  11.3k tokens, max 45k) with only `leaf_gb` 3.65 at ACT 12 / retained 8, so 75% of the layer-tokens are recomputed
  (~17% of the pass); the reference pass is a forward of the same tokens (~3.85x faster than the policy pass, as a
  forward + recompute should be). Contexts are longer than SFT's (13k vs 5k mean), so attention is a larger share.
- **Shipped: the prefixes' retained bound is chosen per step (`--prefix-retained-gb` default -1,
  `lfm3_fast_train.choose_retained_gb`).** What the prefixes retain comes out of `--activation-gb`; the fixed 8 GB left
  the leaves 3.65 GB on step 40. Each candidate bound (shares of the budget) is scored by `levels_plan`'s cost
  (3 layer-token forwards per token plus recomputed layer-tokens, prefix copies of extra parts included) and the least
  wins. On run2's archived step chains at ACT 12: step 40 8 -> 3 GB, 5 -> 11 parts, body compute -3.0%; step 37 -7.2%;
  steps 38-39 -2.3 to -2.8%; at the same memory bound (retained + leaves <= ACT). Exact by construction (parts already
  were); tests: the plan is what the pass ran, the chosen bound has the least cost, outputs and gradients equal the
  unsplit tree. `--prefix-retained-gb 8` restores the old bound. Both learners (exp0_sft, grpo) take it.
  `prefix_tree.recomputed_leaf_layer_tokens` and `retained_limit_gb` are now in train.jsonl.
- **Memory for speed, not taken here (a launch flag):** ACT 12 -> 20 cuts the remaining body compute a further ~5%
  (step 40: r 0.62 -> 0.43 with the chosen bound) for ~+8 GB of learner-phase peak; run2's unit peaked at 58.5 GB under
  a 84-92 GB budget.
- **Not changed, with reasons:** the reference pass is a necessary forward of the start policy at ~49 TFLOPS effective;
  fusing it into the policy pass saves no FLOPs (different weights), caching cannot apply (every step trains fresh
  samples), overlapping it with collection gains nothing on the time-sliced GB10. Keeping the head's logits between
  forward and backward would save 1 of 4 head matmuls (~3%) for 256 KB per loss token, about what the same memory buys
  as saved activations.
- **Not checked:** a GPU timing of the new bound (the DGX had ~10 GB free beside sft-v5; the change is a schedule
  whose cost is settled by first principles, and the next run's `phase_s` shows it); the 88 KB/token-layer activation
  constant against the real peaks (step 40's policy peak 49 GB allocated vs ~29 budgeted: the gap is long single
  leaves, the head chunks and history gathers, not modelled); per-chain `.item()` syncs in `train_leaves` (the 11-17
  profile put all host gaps at 2.5%).
