# Program-header efficiency, 2026-10-12 (REVIEW_20261012 recommendation 5)

Agent: dgx-eff. Question: Mellum v4's direct_sft phase forwards about 766M tokens to train about 43M decision-target
tokens (5.6%), at about 16 s per 32k-token update over 31,281 updates (about 5.6 days). What gives the largest cut in
wall-clock per supervised token, under the owner's whole-trajectory rule (train on prompts and inputs too, tool output at
a lower weight)?

Scripts and outputs: `runs/header-efficiency-20261012/` (`header_stats.py`, `chain_stats.py`, `plan_stats.py`), bench
arms `runs/training-speed-20261012/eff-*`.

## What the data and the current windowing do

Union r3, Mellum twin. Train split only. Current windowing: `tokens` 32768, `target_tokens` 1024. A document is cut
into 1024-token spans, each forwarded with up to 31,744 tokens of preceding context. Only target-bearing spans train.

| | direct_sft (program header) | direct_trajectory (whole trajectory) |
|---|---|---|
| Rows | 357,632 | 165,822 |
| Row tokens | 765.6M | 258.4M |
| Decision-suffix tokens | 43.3M | 258.4M (no declared suffix: every position is a target) |
| Forward tokens (current windows) | 863.5M | 981.9M |
| Supervised positions (current) | 230.6M (26.7%) | 258.6M (26.3%) |
| Forward with whole-row windows | 774.9M | about 265M |
| Forward with whole-row windows + merged chains | 439.3M | (no chains) |
| Prefix trie within programs (tree batching bound) | 279.5M | n/a |

What the program-header masking does now. Inside a trained span, every position is supervised:
`balanced_position_weights` puts half the mass on all positions and half on the decision suffix, with tool output at
`feedback_weight` 0.25. But only the last 1024 positions before or around a suffix are in the span. The rest of the row
is context, forwarded and never trained. v4's log: 27.0k real tokens, 6.7k supervised positions and 1.37k decision
targets per update.

Program-header rendering writes one row per decision. Decision k+1's row repeats decision k's row as its history. Of
357,632 train rows, 149,911 (335.4M tokens) are an exact token prefix of a later decision row of the same program
(`chain_stats.py`, hashing; the merge itself uses byte order). The trie size counts what tree batching could also share
across diverging sub-calls: 279.5M.

## Options

| Option | Forward tokens, direct_sft | Supervised positions | Cost per update | Data effect |
|---|---|---|---|---|
| Current | 863M | 231M | 16.0 s | — |
| (c) whole-row windows (`train_target_tokens` 31744) | 775M (−10%) | about 765M (3.3×) | +about 2 s readout (v4 profile: readout + projection 0.7 s at 6.7k positions; fused readout 2.7 s at 32k) | the owner's whole-trajectory rule: prompt, header and history trained at the context weight. Held windows unchanged |
| (a1) (c) + merge linear chains (`merge_prefix_rows`) | 439M (−49%) | about 430M: each position once | same as (c) | each decision keeps its own balanced segment. A contained row's positions see exactly their own context (causal). Only the framing close token after a contained row is replaced by the next real token |
| (a2) tree batching of the full trie (FlexAttention ancestor mask, per-token rope positions) | 280M (−68%) | 280M | needs a tree mask for the flex and flash layers, per-token positions in the fused rope kernel, and a tree packer | exact, but most of the gain is already in (a1). Follow-up |
| (b) shorter header windows / truncated context | rows ≥ 8k are 20% of forward | — | — | drops long-context training of exactly the programs the model must run; the owner's rule says keep whole trajectories. Not adopted |
| (d) packing | right padding is 7.4% (29.1k forward for 27.0k real tokens). Fill is 27k of 32k | — | a document-masked varlen pack (Flex) could recover about 7–15% | none. Second-order; next after (a1) |

Bench (2 layers, 8 × 3400 tokens, FP8 experts, beside v4 and other jobs, so noisy): span 850 → 3400 per row costs
+2.9 to +4.3 s per update at 2 layers, which is mostly the full-vocabulary readout of 27k positions. Readout scales with
positions, not layers. For the 28-layer model it is about +2 s on a 16 s update. Peak memory 10.95 → 11.13 GB.

whole-trajectory phase: the same `train_target_tokens` alone cuts its forward from 982M to about 265M with the same
supervised set. Every position is a target there already, and a row up to 31.7k tokens sees the same context as before.

## Adopted: (c) + (a1), in the shared trainer behind declared options

- `direct_sft --train-target-tokens N` (recipe `train_target_tokens`): the supervised span of train windows. Held
  windows keep `--target-tokens`, so held CE stays comparable across the change.
- `direct_sft --merge-prefix-rows` (recipe `merge_prefix_rows`): `text_warmup.merge_prefix_rows`. Within one cohort
  and sampling group, and bucketed by the first 64 tokens, a train row with a decision suffix whose tokens are a prefix
  of another train row moves into that row. The longest row keeps its identity and gets `supervised_spans` (every
  constituent's suffix) and `merged_rows` (constituent hashes and ids).
  - Not merged: rows with declared `context_only_spans` and test rows.
  - A row that is a prefix of two branches goes to the one that follows it in byte order.
- `text_supervision.segmented_position_weights`: a window with several suffixes is cut at each suffix end. Each
  segment gets half its mass on all its positions (role-weighted) and half on its suffix, with mass equal to its
  length. One suffix that ends the window is exactly `balanced_position_weights` (test).
- Recipe: the shared base `neuralese-direct-v3` declares `train_target_tokens: 31744` on direct_sft and
  direct_trajectory, and `merge_prefix_rows: true` on direct_sft. Both lines get it (lockstep).

Parity tests (`tests/neuralese/test_header_efficiency.py`):
- segmented weights equal the balanced weights for one suffix;
- chain merge and branch handling;
- span clipping;
- train vs held geometry;
- on a tiny Mellum (sliding window 8, YaRN past 16), the token losses of a contained row inside its container's forward
  equal its own forward's;
- the windowing continuation and the recipe handoff.

## Moving the running v4 (no restart)

A handoff to a recipe that changes these two parameters on an unfinished direct stage is accepted
(`recipe.direct_windowing_handoffs`, recorded as `windowing_changes`). The trainer sees the same inputs with a new
train-window geometry and makes a windowing continuation through the data-continuation path:

1. The plan the run trained on is rebuilt under the saved geometry, through the r2 → r3 record. It must hash to that
   record's `plan_documents_sha256`. This is a new exact check.
2. The documents of the consumed updates count as seen.
3. The rest of the epoch is the new geometry's windows of unseen documents. A merged window is unseen while any of its
   constituents is.
4. The QAT progress continues from its value at the handoff update (`quantization_rebase`). An objective continuation
   at the same handoff reads that rebase.

direct_trajectory has not started, so it takes the new geometry as an ordinary parameter change of an unstarted stage.

## Expected ETA

See "Exact plan" below for the plan-only numbers. Estimate from the data stats:
- direct_sft remaining: from about 30.4k updates at 16.5 s (about 5.8 days) to about 16k updates at about 18 s
  (about 3.3 days).
- direct_trajectory: from about 36.6k updates (about 7.0 days) to about 9.9k updates (about 2.1 days).
- v4 total remaining: from about 12.8 days to about 5.5 days.
