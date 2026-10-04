# S3 full-run handoff — 2026-10-03

## Current decision

Finish reviewing the existing resumable A–F port experiment before launching a
larger run. Its phase-D checkpoint is retained at
`runs/neuralese-s3-pilot-20261003-resume-d-v1/checkpoint.pt` on DGX. The original
interrupted checkpoint was not overwritten. E200/F300 completed in the preserved branch
`runs/neuralese-s3-pilot-20261004-resume-ef-v1` at global step1900;
the normal local crisp-student full Muon run continues independently.

Phase D establishes modest content use on extractive and multihop QA; tool digest
has little correct-versus-shuffled gain. All deterministic payload lengths were
16. Phase E inherited a nearly deterministic stop-at16 boundary and had almost no
action exploration:197/200batchmeans16 and nearly zero centered policy scores.
A future exploration trial must match behavior sampling and policy log-probability
and preserve the completed baseline. Phase F produced mixed small-heldout gains;
ordinary Natlang execution replay is still required. Review
family results continuously rather than inventing a single aggregate threshold.

The cache diagnostic isolated BF16 top-logit ties on four fixed held-out prefixes;
float32 agreed and snapshot-fork readback matched an unforked control exactly.
Retain detailed margins/dtypes/kernel identities in every future harness. This
small diagnostic is not evidence that every cache path is correct.

## Checkpoint and GPU ownership

Qwen generation owns DGX again after the port diagnosis. The v6 reviewed
controller performs readiness checks and owns its importer. Never start a second
GPU server implicitly. The owner authorizes generation pauses: record the exact
campaign state, stop/restart its retained server, and preserve all raw artifacts.
Any further continuation must preserve data/seed/schedule and optimizer state;
do not restart A–D or silently replace AdamW with Muon.

The existing port trainer uses AdamW despite the earlier prose describing Muon.
This is distinct from the local crisp model's actual Muon run. A new full port
run must implement and record a Muon/AdamW parameter split and fully resumable
optimizer state first. Reuse the project's optimizer implementation; do not send
vocabulary-sized feedback readout/embedding matrices through an unsuitable
orthogonalization step. Do not reinterpret an existing AdamW checkpoint as Muon.

## Data and base selection

1. Complete S1 dedup, protected split closure, schema validation and writer-target
   leakage review. Finalizer completed; independent current-schema/structural/
   leakage/duplicate-ID audit passed all1,870,591records on2026-10-04 with
   zeroerrors/duplicates. Source-policy and protected split admission still need
   explicit review. Keep source/licence/outcome labels in its manifest. Unchecked
   teacher trajectories require an explicit quality decision before imitation.
2. Complete v13 compiler migration and replay its held candidates with the current
   runtime. Compiler success alone cannot promote a trajectory. Preserve input,
   source/gold, grouping, lineage, and failure/success pair relations.
3. Select the crisp checkpoint by held-out execution results as well as loss.
   Export/merge its exact LoRA into the pinned LFM2.5 base, retaining a model
   identity receipt. Port phase-F adapters remain separate deltas.
4. Build a bounded streaming/tokenized loader with deterministic source-group
   sampling and resume position. The current pilot `load_family` materializes a
   family and is not a full-corpus loader. Bound producer/target lengths explicitly;
   oversized source records are held, never silently truncated into positives.
5. Freeze the reviewed starting snapshot. Later admitted data enters through a
   recorded refresh boundary with sampler/RNG/optimizer continuity, not an
   untracked replacement of a running dataset.

## Experiment and evaluation

Start with one reviewed cutoff (6) rather than three simultaneous runs. Expand
cutoffs (3/9) only when the first result motivates it. Preserve phase boundary
checkpoints, held-out correct/shuffled/zero/no-block/full-source ablations,
family margins, length histograms, truncation, representation rank/collapse,
payload temperature and ordinary text/Natlang execution results. Inspect source
coverage and reducer performance separately; a loss decrease does not establish
runtime task success.

Checkpoint handoff must preserve cutoff, length cap, trained head dtype, control
rows, LoRA layer/rank/alpha and base identity. The reference loader now restores
these instead of silently discarding phase-F deltas. Verify a trained phase-F
checkpoint through Python serving, GGUF export, C++ non-streaming HTTP and the
TypeScript transport before rollouts. SSE/gradient/optimizer endpoint parity is
still incomplete in the C++ implementation and must remain advertised honestly.

## Work remaining before claiming a full port run is ready

- Fresh stopping-exploration review, plus ordinary Natlang replay of completed F.
- Full S1 quality admission and v13 candidate execution replay.
- Streaming/tokenized sampler and its full resume state.
- New-run Muon parameter policy and optimizer checkpoint support.
- Crisp base selection/merge with exact provenance.
- Trained phase-F serving/export/HTTP transport verification.

This document is a concrete handoff plan, not a claim these prerequisites are
implemented or that candidate data has been admitted.
