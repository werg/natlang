# Direct token-readout integration

The natlang program is unchanged: a producer returns a neuralese value, and a
consumer uses that value to make its next typed decision. This change implements
exact tensor transport and readout, not a new task policy.

1. **Shared projection equation — crisp helper.** Apply TokenChoiceProjection
   to causal final states, then the existing final norm and vocabulary head.
   Slice the last position only after the causal projection. Keep the base
   readout explicit for NeuraleseProjection's soft-token mixture.
2. **Producer — existing DirectWriter.** A call starts with the last prefix
   state, which predicts its first value. Reset convolution history at that
   boundary. Retain the maximum receptive field of both projections. Use TCP
   for token stopping and base logits for value mixtures. Both scheduled
   serving and differentiable replay use this writer.
3. **Consumer — existing Engine and GradSession.** Route ordinary decisions,
   target likelihoods, option scoring, and readback through the same helper.
   Keep sequence history in rollback snapshots. Prompt context is not a target
   convolution segment; the last prefix state begins the next target segment.
   Injected values retain the backbone's exact causal position alignment.
4. **Artifact — existing direct exporter.** Restore both projections and record
   their identities. Keep the current TCP export refusal until every relevant
   consumer route has been implemented and exercised together.
5. **Qualification — existing direct runtime evaluator.** Compare incremental
   readout with sequence readout, serving with replay, and gradients through a
   nonidentity TCP as well as identity initialization. Exact runtime parity is
   separate from held consumer task usefulness.

Steps 1–2 can be published before 3–5; they do not make an artifact with TCP
servable. No private model-specific trainer, readout, or qualification path is
introduced. Ordinary HF generation bypasses TCP, so a TCP lineage's behavior
monitor must use the adapted shared runtime rather than silently reuse that path.

## Integration status (2026-10-11)

Shared Engine and GradSession now carry causal token-readout history through
ordinary targets, returned payloads, guidance snapshots, and decision-option
forks. The exporter restores TCP using its declared configuration and strict
state dictionary. Both projections remain distinct and the value mixture is base.

The synthetic nonidentity TCP GPU mechanism passed on the Pop V8 underlying
artifact (lengths 8 and 16): scheduler payload, producer replay gradient, public
input gradient, decision score, and cached readback deltas were all zero. The
higher-precision readout comparison passed (maximum incremental logits delta
2.57e-5; history-gradient delta 1.91e-6). The default-precision first attempt
failed with a 0.00351 incremental-logit delta and remains preserved; production
precision was not changed.

A further held synthetic TCP stream diagnostic used deterministic synthetic
state `ad78f387d1ef14a7bd436726181ca046ba7791f9038bda72a7725ffbd4d17f79`.
The multi-turn `GradSession._context_logp` path matched the full-sequence causal
readout on 42 weighted targets across four target segments (maximum log-probability
delta 0). A real Engine syntax-guidance rejection restored and replayed five
reply tokens; restored token history and next logits matched a fresh prompt plus
kept-prefix replay exactly (both deltas 0). This covers only the exercised
synthetic mechanism branches. `runtime_qualified` remains false, and the
synthetic state is not a trained TCP artifact or certificate.

Still required are a trained TCP artifact/export-load path, the owner-defined
end-to-end task and behavior gates, and consumer improvement. Synthetic mechanism
success does not qualify those items or grant training admission.

## Frozen readout storage (2026-10-11)

Native recurrence smoke v5 exhausted Pop VRAM before its first update at the
LFM vocabulary readout: each FP32 hidden state cast the frozen BF16 embedding
table into another 256 MiB allocation retained by autograd. Reusing the identical
frozen cast is a crisp numerical/storage mechanism, with no model-facing policy.
The shared helper owns one nonpersistent module buffer, invalidates it on source
identity/version/device/dtype changes, and bypasses caching for trainable tables.
Module device transfers also transfer/release this buffer; no global cache holds
GPU tensors. Hidden-state gradients and vocabulary precision remain unchanged.
The failed smoke remains held; this change alone grants no qualification.
