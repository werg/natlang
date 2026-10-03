# Neuralese: Port Mechanics and Training

## Executive overview

Neuralese is a proposed extension to a causal language model that reads and writes **soft tokens**: continuous vectors in its input-embedding space. These vectors communicate compressed context and intermediate results. Their content is learned from the performance of a model that consumes them, rather than from a requirement to reconstruct source text.

The system has **one execution procedure for writing a block**: a shallow part of the decoder autoregressively generates a sketch and decides its length; the remaining layers complete the block in one causal pass. Final payload vectors do not feed back into the sketch recurrence. Completed blocks enter subsequent computation through the read port.

The initial backbone is **LFM2.5-350M**, with a 1,024-dimensional residual stream and 16 layers combining attention and short convolutions.[^backbone] This is a design proposal; recurrent depth and the attainable quality–latency trade-off remain experimental.

## 1. Port contract

### Read port

A block contains an ordered sequence of embedding-width vectors, its actual length, and a short header describing its purpose. The read port places the vectors at reserved positions inside an ordinary chat message. Each vector occupies one sequence position and participates in the model's normal causal computation.

A consumer receives the payload and its own context, not the producer's private decoder cache. The interface supports text-to-neuralese, neuralese-to-text and neuralese-to-neuralese computation. Byte-sensitive content remains text or a discrete reference; soft tokens do not guarantee exact string recovery.

### Write port

Writing separates three responsibilities:

| Component | Responsibility |
| --- | --- |
| Sketch generator | Produces the next continuous input from the preceding shallow residual state and causal context. |
| Block-control head | Decides whether to add another sketch position or close the block. |
| Content projection | Converts completed, full-depth states into the vectors delivered through the read port. |

The sketch is an internal computational state, not the public payload. A learned, normalised feedback projection maps the shallow residual to the next input sketch. Matching dimensions alone does not make a residual state a suitable input embedding.

Initialise the content projection near the input-embedding table's scale and statistical structure. Token-embedding mixtures can support this initialisation without becoming a separate execution option. Content and boundary decisions have separate heads; the distribution over control actions is never the content representation.

## 2. One block-generation procedure

Let `D` be the full decoder depth and `k` the selected recurrent depth. The model prefills the preceding context normally. Once it opens a neuralese block, the following procedure always applies.

**Generate the sketch.** Run layers `1…k` autoregressively from the block-opening context. The current shallow state determines whether to stop and, on continuation, produces the next input sketch. Process that input through the shallow layers and retain its cutoff-layer residual. Each decision sees only preceding context and sketches.

**Complete the block.** After stopping, run layers `k+1…D` over the collected residuals in one blockwise pass with the usual causal mask. Project the completed states into payload vectors. Completion neither changes the length nor triggers another refinement pass.

**Commit and consume.** Publish the completed block. A subsequent invocation reads it through its own read port. To continue the same conversation from the exported payload, restore the cache at the start of the payload positions and prefill the final vectors and closing marker through the full model before resuming ordinary generation. This is consumption of the completed block, not another write or refinement.

The recurrence is over **sketches**, not final payloads. Consequently, reasoning performed only in the upper layers cannot revise subsequent sketches or the stopping decision within that block. This is a different learned recurrence from full-depth output feedback; its required depth must be established experimentally.

The selected `k` is an architecture setting, not a runtime choice among execution modes. Retained residuals avoid repeating the shallow computation. Upper-layer work is deferred and batched, not eliminated.

### Stopping and length

The full-depth language-model head opens a block by emitting its opening control token. Inside the block, a small head on the shallow state predicts **continue** or **stop**, conditioned on the available context, position and requested budget. The number of continuation decisions determines the length; an independent exact-length predictor is not required.

Boundary supervision teaches closure; subsequent training ties length to usefulness and cost. A hard maximum, reserving room for the closing marker, guarantees bounded execution. Reaching it is recorded as truncation, not learned stopping. Stop is masked before the first vector unless empty blocks are explicitly allowed.

Final training and inference use self-generated sketches. Stopping accuracy on teacher-supplied inputs alone is insufficient.

## 3. Chat templates and decoder integration

### Message structure

Keep the backbone's native chat template, including role delimiters, tool-message structure and end-of-message tokens. Neuralese is inserted **inside message content**, not introduced as a new conversation role. Apply the normal assistant-generation prefix when starting a response; continuing an open assistant message must not add a second assistant prefix.[^templates]

A schematic assistant message is:

```text
<|im_start|>assistant
Optional text <|neuralese|>[purpose; budget] ⟦z1⟧ … ⟦zL⟧ <|/neuralese|> continued text<|im_end|>
```

Register and train `<|neuralese|>` and `<|/neuralese|>` as dedicated neuralese control tokens, each with its own token ID. Treat literal occurrences in ordinary source text as text rather than protocol boundaries. `⟦zi⟧` denotes an embedded vector, not tokenized text. Block closure is distinct from the native end-of-message token.

An incoming block can declare its known length. A generated block declares a budget, with actual length returned at closure. Changing an already-consumed header requires recomputing the affected state.

### Sequence and cache handling

Render and tokenize the surrounding chat structure, reserve the payload positions, then splice the vectors into the embedding sequence supplied to the model. LFM2 supports direct `inputs_embeds` input.[^interface] Attention masks, position indices and sequence lengths must account for every actual soft-token position, delimiter and header token; batch padding is masked out.

For ordinary text, the preceding input's state predicts the next token, whose embedding is then consumed. Inside a block, the shallow generator instead predicts the next sketch input. Upper layers subsequently complete the payload. Later answer tokens must remain invisible to both stages during training.

A cache computed from sketches is not a cache computed from the final payload. The canonical continuation uses the read-port prefill described above. Cache restoration must include both attention key/value state and short-convolution state, with positions reset to the committed sequence rather than advanced twice. Any readback cost is part of the system's execution cost.

The decoder loop must implement block entry, shallow recurrence, completion and readback; tokenizer changes alone are insufficient. Its output carries text/control-token IDs and vector payloads. Text sampling penalties apply only to text; block stopping uses its own head and budget.

## 4. Training

### Migrate bgkit training data

Make migration of **all eligible, reusable bgkit training data** an explicit workstream. Inventory prepared corpora, dataset adapters, generation scripts, teacher trajectories, task fixtures and held-out suites, including material not used by the current training run. Cover compression and reconstruction, summaries, purpose-conditioned question answering, browse-and-answer traces, repository and Git-history context, large tool outputs, and trajectory compaction wherever available.[^migration]

**Preserve tasks and supervision.** Convert each usable example into a common record containing the source or causal trajectory prefix, purpose, consumer context, complete target, recorded observations, outcome labels and representation budget. Retain exact identifiers and source snapshots. Reuse prompt contrasts, distractors, failure-and-repair cases and compression-budget variants; label failed attempts rather than treating them as successful demonstrations. Keep reconstruction examples for bootstrap and ordinary-text replay, with consumer usefulness as the main objective.

**Rebuild model-dependent material.** Render the current chat template and new delimiters from structured records; retokenize text, recompute masks and budget accounting, and generate payloads through the single sketch-based writer. Do not treat cached embeddings, decoder states or token IDs as portable. Reuse teacher distributions only when tokenizer and target alignment are verified; otherwise use recorded target text or regenerate teacher scores. Existing fixed-length slots can initialise boundary supervision, but final training must use generated sketches and learned stopping.

**Validate coverage and isolation.** Preserve source permissions, attribution, hashes and train/validation/test membership; group related repositories, documents and trajectories before creating new windows, and deduplicate across imported corpora. Never place later observations, final patches or target answers in earlier writer inputs. Give every inventoried dataset a disposition—migrated, queued for a specific adapter, or excluded with a reason—and report accepted examples, rejection counts and task coverage. Migration is complete only after rendering, causal-mask and replay checks pass; measure each imported task family's contribution on held-out consumer evaluations.

### Bootstrap the ports and boundaries

Use ordinary text with explicitly designated neuralese spans. Initially, supply embeddings of known text at the sketch positions to establish readable payloads, block entry and closure, and the transition back to text. These are training inputs, not an inference option.

Copying a supplied same-position target establishes only interface behaviour, not generation ability. Next-sketch and stopping supervision must remain causally aligned: a decision cannot inspect the target input it is being trained to predict.

Distil the shallow predictor from a frozen copy of the unadapted full-depth model on ordinary-text continuations, using its next-token distributions or embedding-space targets. A temporary vocabulary readout can support supervision. Explicit span labels teach block boundaries; ordinary end-of-message behaviour alone does not define internal block closure.

### Transition to generated sketches

Progressively replace supplied embeddings with the sketch generator's own outputs and extend the autoregressive rollout length. Train both completion and stopping on these generated trajectories. Finish with entirely self-generated sketches for neuralese positions, while ordinary answer text can remain teacher-forced for token-level losses. Teacher forcing and free-running generation expose a recurrent model to different input distributions, so validation must use the latter.[^rollouts]

Target-embedding warm-up and text distillation are complementary initialisation tools, not additional execution procedures. Reduce their auxiliary losses as downstream training takes over; final representations need not reproduce text embeddings or teacher latent coordinates.

### Learn through a consumer

The main training example separates a producer from a consumer. The producer sees the source and purpose and writes a block. The consumer sees that block and its task, with the original source withheld where necessary to prevent bypassing the communication channel.

The teacher is the same model with the full source, or a larger same-tokenizer model. Combine teacher-to-student KL divergence on target tokens with cross-entropy on correct outputs. Backpropagate through the consumed payload, projection, completion and sketch recurrence, including chains of neuralese-to-neuralese computations.

Pair different purposes for the same source to teach purpose-sensitive encoding, and vary compression budgets. For a general-purpose representation, train against several questions rather than a single continuation. Couple a length or compute cost to task quality so that stopping does not simply learn the longest allowed block—or an empty one.

Hard start/stop decisions are discrete. After boundary supervision, train them explicitly with task-and-budget feedback, such as scored length choices or policy gradients. Backpropagation through continuous payloads does not itself differentiate a sampled stopping point.

### Adapt the shared model gradually

Train the new feedback projection, content projection, control head and writer adapters first, with the shared backbone and reader initially frozen. Introduce reader adapters when gains from correctly matched payloads plateau; then release shared layers gradually with small, layer-specific learning rates. Reader and writer share the backbone, so shared-weight updates affect both.

Retain ordinary-text replay against the original model and held-out capability checks throughout adaptation. Monitor representation norms, effective rank, common-direction collapse and sensitivity to payload substitution from the beginning. Freezing is a stabilisation schedule, not a permanent separation of the ports.

## 5. Evaluation and architecture selection

Choose the recurrent cutoff empirically using **downstream quality, stopping reliability and measured end-to-end latency**. For the hybrid backbone, cutpoints must account for attention placement and convolution state. Including an attention layer lets the sketch generator consult the full causal prefix rather than relying only on local convolution windows.[^backbone]

Compare correct, shuffled and zeroed payloads at matched lengths under the current reader, with full-text input as a separate reference. Report absolute losses as well as the teacher–student gap. Correct payloads must improve held-out consumer performance relative to shuffled payloads; aggregate task improvement alone can conceal a reader that ignores the channel.

Measure premature stopping, excess length, truncation and post-block continuation quality. Test fully generated sketches on longer contexts and unfamiliar tasks, including text–neuralese transitions and successive blocks.

Benchmark prefix processing, shallow generation, block completion, projection and readback separately and together, alongside arithmetic cost and peak working-state size. Verify that cached execution agrees with recomputation on the same committed payload sequence. No speedup or successful cutoff depth is assumed in advance.

[^backbone]: LiquidAI, [LFM2.5-350M model configuration](https://huggingface.co/LiquidAI/LFM2.5-350M/blob/main/config.json).
[^templates]: LiquidAI, [Chat Template](https://docs.liquid.ai/lfm/key-concepts/chat-template); Hugging Face, [Chat templates](https://huggingface.co/docs/transformers/en/chat_templating).
[^interface]: Hugging Face, [LFM2 model interface](https://huggingface.co/docs/transformers/en/model_doc/lfm2).
[^rollouts]: Bengio et al., [Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks](https://arxiv.org/abs/1506.03099).

[^migration]: bgkit, [Training workflows](https://github.com/werg/bgkit/blob/main/docs/02_training_plan.md) and [context-compression task families](https://github.com/werg/bgkit/blob/main/plans/capability_packaging_2026_08_20.md). Inventory establishes which prepared datasets and generators are available; this section specifies the migration work.
