# Training-time corrective-prefix SFT with asynchronous repair

Implemented as an opt-in mode; production thresholds and throughput gains are not
established. This is now the preferred experiment over mandatory corpus-wide
upfront search. Existing frozen search artifacts remain immutable.

## Training

Supply `--repair-mean-nll M --repair-token-nll T` to `scripts/train_lora.py`.
Both thresholds are positive finite NLL values in natural-log units, not perplexity.
The existing training forward produces unreduced target cross-entropy. Difficulty
checks use detached losses; easy rows retain vanilla per-example mean SFT loss.
The first supervised token above T triggers a cutoff through its complete assistant
action. If no token qualifies but whole-row mean exceeds M, select the first action
whose mean exceeds M. Later targets are masked before backward; their observations
remain context. Ignored prompt, observation and reasoning tokens never trigger a gate.
Held-out evaluation remains ordinary unmasked loss.

Use merged conversation rows (`scripts/merge_sft_chains.py`) to withhold later
assistant actions. A single-action row retains the complete gold action and queues
repair; it has no suffix to mask. Independent later rows are not silently skipped.
Parent held-out rows must remain byte-identical: merge only reviewed train rows and
recombine with the original protected rows, then obtain ordinary corpus/source gates.
Never merge across splits or change source membership to satisfy this experiment.

`--repair-full-gold-every 10` retains full gold exposure every tenth training example
while still flagging difficulty. This is a configurable anti-forgetting safeguard;
zero disables it. Thresholds/cadence are bound into resumable corpus identity. A
change from vanilla training requires an explicitly approved post-training phase
manifest specifying the same online-repair policy. Optimizer, RNG and protected
held-out membership remain subject to the existing phase checks.

## Queue and repair

`OUT/repair-outbox.sqlite` stores every visited row's mean NLL, first hard token,
action cutoff, retained/total target counts and full-gold flag, with corpus/tokenizer
and policy identity. Observations are idempotent by optimizer step and row offset.
A consumer may read only observations at or before its pinned committed checkpoint
step; replayed steps replace earlier receipts. Latest easy observations supersede
older hard flags. Training metrics report unchanged/flagged rows and retained tokens.

Prepare a reviewed `natlang.student_chunk_rewrite_plan/1` against an exact exported
adapter of that checkpoint. Add `online_repair_corpus` naming the training corpus.
Run:

```sh
python scripts/prepare_online_repair_batch.py \
  --outbox OUT/repair-outbox.sqlite --checkpoint OUT/checkpoint \
  --source-plan REVIEWED_PLAN --output NEW_BATCH --limit 32
```

The adapter weight hash must match the committed checkpoint. Only flagged programs
with already reviewed executable sources enter the derived plan. Unmapped sources
and capacity-deferred programs are explicit in its receipt. Run that plan with the
existing `rewrite-student-chunks.mjs` operator and exact plan SHA. It supplies
student-biased teacher candidates, fresh tool execution, oracle validation, changed
observation continuation regeneration, student difficulty gates and corrective-prefix
fallback. Ordinary render/token/source admission is still required. Non-passing
full continuations cannot become positive suffix targets; valid corrective prefixes
remain separately identified. Append admitted repairs through the existing reviewed
append/phase mechanism; never mutate a running corpus in place.

## Scope and next measurements

No additional student scoring forward is needed for easy training rows. This does
not avoid their ordinary forward/backward, nor the forward computation of masked
suffixes. Unreduced logits/losses, detached transfer and durable receipts introduce
cost relative to the fused single-example loss path. Measure wall time, GPU memory,
unchanged fraction, retained supervision, repair calls and downstream task accuracy.
Training-mode dropout may influence flags (our LFM candidate typically has no dropout).
Current boundaries are complete actions, not finer sentence/code-line AST spans.
The repair bridge batches only hard programs but replays/scores their reviewed
trajectories; it does not yet reuse cached training activations or resume teacher
regeneration from an arbitrary training-token offset. The queue is durable, but an
always-on periodic dispatcher and automatic reviewed admission/append are not yet
launched. Do not describe this as a measured speedup or a completed production loop.
