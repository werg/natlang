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
