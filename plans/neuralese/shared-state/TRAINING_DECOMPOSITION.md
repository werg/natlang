# Native-state training and cross-model adaptation decomposition

This is implementation work, not a substitute for working training stages.
The owner requires complete training and inference, no prototype shortcuts,
and an independent integration review before replacing the diagnostic runs.

## Natural-language stages

1. Prepare each source as typed invocation graphs and ordinary text invocations.
   Retain the complete source trajectory, actual masks, role-marker boundaries,
   split/source/quality provenance and authenticated imports. A source that
   lacks the text or state provenance needed for conversion is an explicit
   repair input, never silently fabricated or dropped from coverage tracking.
2. Train each model through the complete native recipe. Use one shared trainer
   and graph objective for LFM and Mellum; ordinary text is a single-node graph.
   Train backbone, low-rank embedding transform and the four role-token rows.
   Do not instantiate the old communication projection as a trainable module.
3. Read task and held-loss feedback and revise data, optimizer rates, objectives
   and stage exposure as useful. Preserve valid optimizer/RNG state on
   continuation. Unsupported changes require a declared conversion or restart,
   not silent state substitution. Sampling-law capture for RL remains exact.
4. As soon as two native models are ready, select them for shared-space
   adaptation. Do not wait for every registered model. Additional ready models
   join later through explicit adjustable continuations.
5. Train the shared semantic VAE on aligned producer spans and target native
   state. Predict target-state deltas relative to its text-prefill baseline;
   do not pretend that source and target KV coordinates have equal meaning.
   Preserve token/character alignment provenance for different tokenizers.
6. Jointly adapt the semantic space, producer/control write-out and target
   behavior. Pass gradients through decoded target state and the consumer;
   supervise actual producer tokens and closing markers, native-state
   reconstruction and downstream successful behavior. Alternate/update
   participating models as declared; checkpoint all jointly trained state.
7. Distill useful behavior and incorporate new verified trajectories. Check
   real cross-model producer/consumer execution through the same inference
   transport, including serialization, positional handling and termination.

## Crisp mechanisms and exact verifiers

- Graph compiler/executor: the implementation agent's shared model module.
  It implements the prescribed graph, attention positions and finite hybrid
  state operations; it does not decide which data or objective is useful.
- Data reader: reuse streaming JSONL, source manifests and protection indices.
  Verify graph references, token masks, source provenance and split closure.
  Imported/forced boundary positions are not sampled policy actions; do not
  attach a next-token target to an unrelated producer's aliased hidden state.
- Trainer plumbing: reuse backbone configuration, MuonSR/AdamW, named optimizer
  restoration, checkpoint policy, memory ledger and exact task evaluators.
  Both models use the same training loop and stage implementation.
- Recipe stages: executable native training, shared semantic VAE, joint
  adaptation and behavioral distillation handlers, with explicit inputs,
  checkpoints, adjustable parameters and completion/exposure reports.
- Readiness verifier: count distinct model identities whose declared native
  prerequisite stages completed and whose checkpoint and native runtime
  certificate match. Two suffice. This verifies the owner's threshold; it
  does not invent a task gate or count artifact availability as exposure.
- VAE mechanics: model-specific state encoders/decoders connect to a common
  per-token semantic posterior; explicitly align supervision across tokenizers.
  Reparameterized samples, KL/reconstruction/semantic/behavior losses and
  configurable weights must be actual differentiable computations.
- Joint optimizer/checkpoint: preserve participating model parameters, VAE,
  marker/control rows, optimizer states, sampling RNG, data cursor and exposure.
  Exact trainable-group declarations support safe adaptive continuation.
- Inference adaptation: decode the learned semantic code into the target native
  state delta and apply it using the graph agent's external-state splice API.
  Until trained, only explicit best-effort text reencoding is available across
  incompatible models; report that distinction in execution receipts.

## Required integration evidence

Forward and gradient equivalence on native graphs, actual LFM/Mellum updates,
complete recipe execution and resume, prompt/data-mask agreement, two-model
readiness triggering, meaningful VAE and joint updates, and durable real
producer-to-consumer inference. Include different tokenizers, nested exports,
fanout, foreign-state rejection, learned cross-model decoding, marker
termination, new-data admission and adjustable continuation. No mocked stage
completion, zero-loss placeholder, or text-only route can qualify the learned
cross-model state path. A separate agent reviews all integrated code and the
evidence; the supervisor fixes gaps before cloud cutover.
