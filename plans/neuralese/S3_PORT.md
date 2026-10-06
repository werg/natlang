# S3: Port model and port training

Draft, 2026-10-03. Detailed plan for stage S3 of the [Neuralese programme](README.md). It implements the [port document](sources/port-mechanics-and-training.md) on the Natlang crisp student, under the decisions in the README and the contracts in [S0](S0_SPEC.md) revision 2. It ends at gate **G1 (port viable)**. Items marked **Proposed** await the owner; they are collected in §13.

S3 builds the PyTorch reference implementation of both ports, trains them through a fixed curriculum, chooses the recurrent cutoff, and tags the trained model's space as the first dialect version, `nd:natlang@1`. Runtime, server and llama.cpp work is S4; program-level training and law objectives are S5. S3 makes the evaluation harness and law hooks ready for them.

## 1. Starting point and fixed facts

| Fact | Value | Consequence |
| --- | --- | --- |
| Backbone | LFM2.5-350M, revision `9e6c6ccf…`, 16 layers, width `d = 1024`, FFN 6656, 16 query / 8 KV heads, RoPE θ = 10⁶ | |
| Layer types | Full attention at layers 2, 5, 8, 10, 12, 14 (0-indexed); short convolution (kernel `conv_L_cache = 3`) elsewhere | Cutoffs just after an attention layer let the sketch consult the full causal prefix |
| Embeddings | Vocabulary 65,536, **tied** input/output embedding `E ∈ ℝ^{V×d}` | New control-token rows act on both input and logits; content initialisations can reuse `E` statistics |
| Reserved tokens | `<|reserved_6|>` … unused in the native template | `<|neuralese|>` = `<|reserved_7|>`, `<|/neuralese|>` = `<|reserved_8|>` (S0 §10) |
| Starting weights | Best Natlang crisp student: the v13 line (`runs/lfm25-350m-broad-20261002/full-v13-muon-source-clean-periodic-epoch1-v3`, rank-16 LoRA, Muon), at the checkpoint selected by its execution evaluation | |
| Compute | Whole DGX Spark (GB10, 128 GB unified memory, ARM64, `ssh dgx`); local RTX 4060 for development and small tests | No gradient checkpointing or other throughput-costly memory savers |
| Teacher | No separate model. The KL target is the same model given the full source | Implemented by toggling S3's trainable deltas off (§4.4) |

**Proposed: LoRA handling.** Merge the selected v13 LoRA into the BF16 base to form the **crisp base**, and freeze it. All S3 changes to shared layers are new, separately stored deltas (fresh LoRA on the reader and writer paths, plus the new modules). The self-distillation teacher is then exactly "crisp base with S3 deltas disabled", with no second copy of the model. At G1 the deltas are kept separate; merging is decided in S5.

The active v13 run is frozen and must not be touched. S3 runs under its own run identities (`runs/neuralese-s3-<date>/…`) and W&B project `natlang-neuralese`.

## 2. Architecture

All new modules are small relative to the backbone. Shapes are for 350M.

| Module | Shape | Role | Initialisation |
| --- | --- | --- | --- |
| Control-token rows | 2 rows of `E` | `<|neuralese|>` opens a block (predicted by the full-depth LM head); `<|/neuralese|>` closes it (inserted by the write procedure, read by consumers) | Mean of `E` plus small noise, RMS matched to `E`. Only these rows of `E` are trainable. |
| Feedback projection `F` | `RMSNorm(d)` → vocabulary-mixture branch + residual MLP `d → 2d → d` | Maps the shallow residual `h_k[i]` to the next input sketch `s[i+1]`: an input latent generated autoregressively by the shared shallow layers, learned through consumers, not a next-token reconstruction (2026-10-06) | Mixture branch: `E^⊤ softmax(R_k(h_k)/τ)` through a temporary readout `R_k` (`d → V`, initialised from the tied head with the final norm) distilled in phase B; MLP branch zero-initialised behind a learned gate. Output passes the interface norm. |
| Stop head `S` | `RMSNorm(d)` ⊕ learned in-block position embedding (`L_max × 64`) → MLP `d+64 → 256 → 1` | `P(stop | h_k[i], i)` after each sketch position | Bias −3 (rarely stop); masked at `i = 0` (no empty blocks unless a task explicitly allows them) |
| Content projection `P` | `RMSNorm(d)` → `Linear(d, d)`, residual on the sketch; plus a log-scale head `Linear(d, d)` | Mean `μ[i] = s[i] + P(h_D[i])`, scale `σ[i] = exp(S(h_D[i]))`; payload `z[i] = μ[i] + τ·σ[i]⊙ε` at Neuralese temperature `τ` | `P` zero-initialised, so the first payload equals the sketch. The log-scale head is initialised to a small constant scale. During bootstrap the sketch is a known-text embedding, so early payloads are readable by construction. |
| Interface norm | RMSNorm with learned gain, applied to every vector entering the read port | Prevents scale drift and the rank-1 collapse bgkit hit | Gain set to the RMS of `E` rows |
| New LoRA deltas | Rank 64 on attention and FFN projections, per layer | Reader and writer adaptation of shared layers (phase F) | Zero (B matrix), as standard |

The temporary readout `R_k` and the mixture branch are parts of `F`, not a separate execution mode. Whether the mixture branch stays, is annealed, or is removed after phase C is an ablation inside the cutoff sweep (§7).

## 3. Write and read procedures

### 3.1 Write

One procedure for every new block, as in the port document §2. With cutoff `k`:

1. **Open.** The full model emits `<|neuralese|>` at depth `D`. Snapshot the cache at the block start: attention KV for all layers is a position index; each short-convolution layer's rolling input window (`Lfm2HybridConvCache.conv_cache[layer]`, sized by `conv_L_cache = 3`) is copied.
2. **Sketch.** Supply the inputs with layers `0…k−1` only (sequentially as below, or blockwise for all positions at once with a fixed number of refinement passes, `execution.blockwise_sketch`); the sketch is an efficiency device between a perceiver and the autoregressive model: autoregressive through shared shallow weights, without next-token input fidelity; the out port reads the top layer (step 3). At position `i`, compute `h_k[i]`, the stop probability `S(h_k[i], i)`, and, on continue, the next input `s[i+1] = F(h_k[i])`. The shallow layers' KV and conv states advance; the upper layers' caches stay at the block start. Stop is sampled (or greedy at inference); the runtime hard maximum `L_max` forces closure and sets `truncated`.
3. **Complete.** Run layers `k…D−1` once over the collected residuals `h_k[0…L−1]`, causally, from the upper caches at the block start. Project the payload from the top states and store the block with its dialect tag. Autoregressive layout (`latent-sketch-v2`, the default from 2026-10-06; see below): `p[j]` comes from the top state at `j−1`, with `j = 0` read from the position before the block. Earlier profiles: `p[i] = s[i] + P(h_D[i])`.
4. **Read back.** Restore every layer's cache (KV positions and conv states) to the block-start snapshot. Prefill the payload (through the interface norm) and `<|/neuralese|>` through the full model. Ordinary decoding resumes. Positions are reset to the committed sequence, never advanced twice.

Cost per block of length `L`: `L` sequential steps through `k` layers, one blockwise pass of `L` positions through `D−k` layers, and one blockwise readback of `L+1` positions through `D` layers. Only the first is sequential. No speedup over text is assumed; S3 measures it (§6.6).

Because the upper pass is causal, the payload of any prefix of length `l ≤ L` equals the first `l` vectors of the full payload. Phase E (§5) uses this to score several stop points from one completion.

**Stop source (2026-10-04).** The pilot's stop head read `h_k[i]` and the count `i`. Every phase A/C span had 16
tokens, and in phases D–F every one of 1,700 steps wrote exactly 16 vectors: the head learned the count, not the
content, and phase E had nothing to explore. Two changes, both configurable and recorded in checkpoints
(`port_config.stop_source`, `stop_position`; older checkpoints keep the sketch source with the count):

- **Full-depth stop with lookahead** (`stop_source = "final"`). The decision after `i` vectors reads the completed
  state `h_D[i]`, without the count. The writer sketches `c` positions ahead (`lookahead`, default 4), completes them
  through the upper layers, and stops at the first completed position whose stop logit says so; positions written
  ahead are dropped and the block is completed over its kept positions. By the causality above this is exactly the
  block that stopped there, at a cost of at most `c−1` extra sketch steps and one chunked upper pass. Training
  writes to the cap, completes once and decides along `h_D`; gradient replay scores recorded lengths the same way.
- **Supervised lengths.** Where the data gives a length, the writer is teacher-forced to it and the stop head is
  trained with BCE at that boundary (continue before, stop at it). Phases A/C draw span lengths from a spread
  (`--span-lengths`), batched by length. Phase D takes `ceil(source tokens / tokens_per_vector)` vectors per record
  (`--tokens-per-vector`, clamped to `[min_length, max_length]`). Only phase E lets the head choose, by policy, with
  exploration.

**Autoregressive block layout (`latent-sketch-v2`, owner 2026-10-06).** A block is organised like text. The
position before the block (the last context position for raw profiles, which have no open marker in the sequence)
takes the block-start input. Its top-layer output is the first Neuralese vector. Every block position's top output
is the next vector, and the last position's top output predicts the close token through the model's own LM head.
Indexing, with `h_k`/`h_D` the shallow and top states and position `−1` the one before the block:

- payload `p[j] = P(h_D[j−1])`: the frozen causal reference (the greedy next-token embedding at init) plus a
  zero-initialised residual;
- sketch input `s[j] = F(h_k[j−1])`, which predicts `p[j]` in the same slot. That is the same position in the stack
  whose top output it stands in for, and it feeds slot `j` the way text feeds a token to the next position;
- stop "after `j` vectors" is the close token's log-odds against all other tokens at `h_D[j−1]` (`CloseTokenStop`, no
  parameters of its own). Every position has a payload target, and the last one also predicts the end token.

At init, with greedy tokens supplied as inputs, the block is greedy text in the slots text would use
(`tests/neuralese/test_raw_port.py`). `latent-sketch-v1` projected each position's own top state, one slot early
against text, with a separate stop head. It stays loadable; new handoffs use v2 (`train/sketch_handoff.py`). The
one-step sketch gradient's self-target is same-slot under v2 (`sketch[j]` vs `p[j]`, every position). C++ fork: v2
is not served there yet (payload shift and close-token stop pending).

The C++ (llama.cpp fork) writer still implements the sketch-state stop only; a `final` checkpoint must not be served
there until it gains the lookahead procedure.

### 3.2 Read

The renderer produces token IDs with the two control tokens and `L` placeholder positions between them. The model input is built in embedding space: `E[token]` for tokens, `InterfaceNorm(u[i])` at placeholder positions. Position IDs are contiguous across text and payload. Attention masks are the usual causal masks; batch padding is masked. The consumer never receives the producer's cache or sketches.

### 3.3 Cutoff candidates

| `k` | Layers in the sketch generator | Attention layers included | Sequential cost per position |
| --- | --- | --- | --- |
| 3 | 0–2 | 2 | 3/16 of a full step |
| 6 | 0–5 | 2, 5 | 6/16 |
| 9 | 0–8 | 2, 5, 8 | 9/16 |

Every candidate ends just after an attention layer, so the sketch state has consulted the whole causal prefix. Schnitzeljagd found the layer-8 state at a call token nearly degenerate (effective rank ≈ 2.5) for retrieval queries; S3 measures the effective rank and next-token predictability of `h_k` for each candidate before and after training rather than assuming any depth works.

## 4. Implementation

### 4.1 Package layout

A Python package inside this repository, `training/neuralese/` (import name `natlang_neuralese`, registered in `pyproject.toml`):

| Path | Contents |
| --- | --- |
| `model/lfm2_port.py` | Wrapper over the installed HF `Lfm2ForCausalLM`: layer-range forward (`0…k−1`, `k…D−1`), `inputs_embeds` splicing, cache snapshot and restore for KV and conv state |
| `model/heads.py` | `F`, `S`, `P`, interface norm, temporary readout `R_k`, control-token rows |
| `model/dialect.py` | Dialect tag, normalisation, block serialisation compatible with the S0 store entry |
| `write.py`, `read.py` | The procedures of §3, batched, with an optional differentiable mode for training |
| `data/` | Loaders for S1 port records, ordinary-text spans, rendering into the native chat template and into Natlang eval contexts (§5.3), span labels |
| `train/` | Phase configurations, losses, schedules, the resumable trainer |
| `eval/` | The harness of §6 |
| `laws.py` | Law-consistency loss hooks (§6.8), disabled in S3 |
| `tests/neuralese/` (repo root) | Unit tests: cache agreement, splicing, causal alignment, prefix-payload equality |

The trainer reuses the repository's conventions: `scripts/training_optimizers.py` (Muon and AdamW), the resumable supervisor, SIGTERM-safe checkpoints at optimizer-step boundaries, immutable run plans, and W&B scalar reporting.

### 4.2 Training-time execution

- **Teacher-forced parallel paths.** Bootstrap and distillation phases (A, B) run the whole sequence in one parallel forward: sketch positions take supplied embeddings, and the shallow predictions at every position are computed at once.
- **Parallel scheduled sampling.** For the transition (phase C), a two-pass scheme avoids sequential unrolling: pass 1 runs teacher-forced and produces `F(h_k)` at every position; a schedule replaces a fraction of supplied inputs with those generated sketches; pass 2 trains on the mixed inputs. Several passes approximate longer rollouts.
- **Full unroll.** From the end of phase C onward, blocks are generated sequentially through the `k` shallow layers, with backpropagation through time over the whole sketch recurrence. A step is one position through `k` layers with a KV cache; it is batched across many producers and captured with CUDA graphs / `torch.compile` to amortise launch overhead.
- **Producer → consumer in one graph.** Producer and consumer sequences are packed into one step. The consumer loss backpropagates through the payload, `P`, the upper blockwise pass, the sketch recurrence and `F`. Neuralese-to-Neuralese chains (phase D) extend the graph by further producer stages.
- **Memory.** No activation checkpointing. A 350M model at ~64k tokens per step needs on the order of 40–50 GB of activations in BF16; unrolled sketches for blocks up to `L_max = 128` add little. This fits the GB10's 128 GB with the backbone, optimiser state and a frozen-teacher pass. Step size is tuned for throughput, not memory.
- **Teacher pass.** The self-distillation teacher is the crisp base with S3 deltas disabled, run under `no_grad` on the full-source rendering of the same example in the same step.

### 4.3 Kernels on ARM64

The LFM2 short-convolution path benefits from `causal-conv1d`; an ARM64/GB10 build exists in the home directory tree and must be validated against the PyTorch fallback before training. Flash-attention availability on GB10 is checked the same way. Correctness tests run on both the 4060 and the DGX.

### 4.4 What trains when

| Phase | Trainable |
| --- | --- |
| A–C | Control-token rows, `F` (with `R_k`), `S`, `P`, interface norm |
| D | Same modules, now trained through producer/consumer objectives |
| E | Plus the stop policy objective on `S` |
| F | Plus reader LoRA on layers `k…D−1`, then writer LoRA on layers `0…k−1`, then shared layers with small, layer-specific learning rates |

From phase D on, the soft system-prompt bank (decision 40) trains with the phase's modules: one free block per prompt piece, initialised from the piece's token embeddings, with its own learning rate. It is saved with each checkpoint as a `.nz` bank in the checkpoint's dialect; the crisp text stays the initialisation and the fallback for crisp drivers.

## 5. Curriculum

Phases are sequential within one run lineage. Each has an exit check measured by the harness (§6) on held-out data. Ordinary-text replay runs throughout from phase C onward.

### 5.1 Phases

**A. Bootstrap the ports and boundaries.** Ordinary text with designated spans. A span's text-token embeddings are supplied at sketch positions between the control tokens; the model reads them as a block and continues. Losses: continuation after the block; LM-head prediction of `<|neuralese|>` at span starts (span labels); stop-head boundary at span ends on supplied inputs. Copying a supplied same-position embedding teaches interface behaviour only, not generation. *Exit:* post-block continuation loss within a small margin of the plain-text version; reliable block entry and closure on supplied inputs.

**B. Shallow distillation (optional initialiser; owner 2026-10-06).** Not a required phase: inside a block there is no autoregressive input fidelity to reach, and the sketch is trained through consumers (D). If used to initialise `F`: on ordinary text continuations, train `R_k` and `F` so that `h_k[i]` predicts the next input: KL from the frozen crisp base's next-token distribution to `R_k(h_k[i])`, plus cosine and RMS-matched MSE between `F(h_k[i])` and `E[t_{i+1}]`. Causal alignment: the decision at `i` sees only `h_k[i]`, never `t_{i+1}`. *Exit:* next-token agreement of `R_k` with the full model reaches a level set from the cutoff sweep's probes.

**C. Transition to generated sketches.** Parallel scheduled sampling with the generated fraction rising from 0 to 1 and rollout length rising to `L_max`, then full unroll (§4.2). Completion and stopping both train on generated trajectories. Validation is free-running only. *Exit:* free-running blocks on held-out spans keep the post-block continuation loss within the bound set from phase A, and stopping no longer depends on supplied inputs.

**D. Learn through a consumer.** From S1 port records. The producer sees the source and the write-site context and writes a block; the consumer sees the block and its task, with the source withheld. Loss: cross-entropy on the consumer's correct output plus token-level KL to the teacher (the crisp base given the full source instead of the block). Variations:
- **Purpose pairing through the write site.** The same source written under different producing functions (different instructions, declared types, causal context), each read by the matching consumer. Purpose is never a separate argument; it lives in what precedes the opening marker (S0 §4.4).
- **Several consumers per block** for general-purpose views.
- **Chains**: Neuralese → Neuralese → text, up to three hops.
- Length enters only through the stop head and a length cost (phase E); there is no requested budget.

*Exit:* correct payloads beat shuffled ones on held-out consumers in every main task family (§6.1).

**E. Stop policy.** Boundary supervision taught closure; now tie length to usefulness and cost. For each example, score several stop points using the prefix-payload property (§3.1): one completion, a consumer pass per candidate length. Train `S` towards the candidate that minimises `consumer loss + λ · L`, as scored-choice supervision, with a REINFORCE-with-baseline variant as an ablation. λ is swept; the chosen value goes into the dialect metadata. *Exit:* stopping metrics within G1 bounds (§9) without loss of consumer quality.

**F. Gradual adaptation of the shared model.** When gains with correctly matched payloads plateau, add reader LoRA (layers `k…D−1`); then writer LoRA (layers `0…k−1`); then release shared parameters with small, layer-specific learning rates. Reader and writer share the backbone, so every shared update affects both. Ordinary-text and Natlang crisp-corpus replay with KL to the crisp base keep ordinary capability. *Exit:* no further consumer gains at stable ordinary capability.

### 5.1a Temperature-gated payload noise

Writing is stochastic at Neuralese temperature `τ` (spec/NEURALESE_PORT.md, Temperature). Training raises `τ` on a schedule once payloads are readable (from phase C onward), with an optional β-weighted `klPrior` term toward a standard normal in the normalised payload space. The aim is VAE-style robustness: consumers that tolerate perturbation, and a smoother representation space for direct training of encodings later. The write procedure exposes `τ` (default 0 at inference) and records `μ`, `σ`, `τ` and the seed so that the payload log-density is available to S5–S7. The harness reports channel use and consumer robustness as functions of `τ`, and watches for posterior collapse (the correct/shuffled/zeroed comparison). Noise on the sketch recurrence, in addition to the payload, is a studied variant.

### 5.2 Replay and monitors

From phase C on, every step mixes in ordinary text (from the raw text sources S1 inventories) and the current Natlang crisp SFT corpus rendered for this model, with KL to the crisp base. Representation monitors run from phase A (§6.4).

### 5.3 Natlang form

From phase D on, a share of examples is rendered in Natlang form: the block is written as a literal in eval code inside LFM's native Pythonic tool-call format (`<|tool_call_start|>[eval(code="const notes: Neuralese<Notes> = <|neuralese|>…<|/neuralese|>; …")]<|tool_call_end|>`), and read as a Neuralese-typed declaration in a call's opening scope. Eager type annotations come from S0's rewrite pass (S0 §11.4). This ensures G1 measures the port where S4 and S5 will use it, not only in plain chat messages. Natlang-form examples carry the call's system prompt as its soft pieces (decision 40), so the port is trained under the prompt it will run under.

## 6. Evaluation harness

Runs from the first training step, on held-out splits from S1 that are closed over source groups.

### 6.1 Payload ablations

For every consumer evaluation, at matched length: **correct** payload, **shuffled** payload (another example's block of the same length and family), **zeroed** payload, **no block**, and the **full-text** reference. Report absolute losses per family, the correct-versus-shuffled margin, and the fraction of the no-block-to-full-text gap recovered. Aggregate improvement alone is never accepted as evidence; the shuffled control is mandatory.

### 6.2 Stopping

Length distributions per family; premature stopping (blocks that hurt the consumer relative to a longer prefix of the same completion); excess length (a shorter prefix scores as well); truncation rate at `L_max`; correlation of length with source information content within a family; behaviour on longer sources and unfamiliar tasks than training.

### 6.3 Continuation and transitions

Text quality after a block, successive blocks in one message, text → block → text → block, and blocks inside tool-call arguments.

### 6.4 Representation monitors

Payload and sketch RMS, effective rank of payloads per batch and per family, cosine similarity to the batch mean direction (collapse), similarity of payloads for different sources, and sensitivity of consumer loss to payload substitution. Alerts trigger before training continues past a phase boundary.

### 6.5 Ordinary capability

Held-out Natlang SFT loss, the existing CPU execution evaluation on protected cases, and general text perplexity, all against the crisp base.

### 6.6 Latency and cost

Separately and together: prefix processing, shallow generation, blockwise completion, projection, readback. Also arithmetic cost and peak working-state size. Measured on the DGX and the 4060, against generating the same number of text tokens.

### 6.7 Cached versus recomputed

The incremental write-and-readback path must agree with a from-scratch forward over the committed sequence (prefix, markers, payload): same greedy continuation, logits within BF16 tolerance. Includes conv-state restoration and position handling.

### 6.8 Law hooks

`laws.py` implements the S0 §4.2 consistency measurements over `read`, `map`, `combine`, `split` and `zip` as soon as those operators exist (S5). In S3 the hooks are tested with stand-in operators so the harness is ready; no law objective is trained.

## 7. Cutoff sweep

Run phases A–D at `k = 3, 6, 9` with identical data, steps and compute, plus a mixture-branch ablation at the best `k`. Before training, probe the crisp base: effective rank of `h_k` and linear next-token predictability at each candidate. Choose by:

1. consumer quality (correct-versus-shuffled margin and gap recovered),
2. stopping reliability on free-running sketches,
3. measured end-to-end latency per block.

The winner continues through E and F. The losers' runs are kept as evidence.

## 8. Dialect `nd:natlang@1`

Dialect stability is not a concern yet (README, "Dialects"). `nd:natlang@1` is simply the version tag of the space the first S3-trained model reads and writes: width 1024 (the model width; no adapters for 350M), BF16 storage, RMS-normalised vectors, the control-token strings, and informative metadata (`L_max`, the stop-policy length weight, cutoff `k`, training run and checkpoint). When later training changes the space, the version is bumped and stored blocks are regenerated or converted.

The S3 evaluation set doubles as a small reader/writer check for the S4 servers (parity against the PyTorch reference). Law-agreement measurements arrive with S5's operators and enable S0's compiler rewrites on the current model.

## 9. Checkpoint G1

G1 is a review point, not a fixed threshold. All dimensions are measured on held-out splits closed over source groups, with free-running sketches and learned stopping, and tracked across training. The review judges whether the port works well enough to tag `nd:natlang@1` and hand trained weights to S4 and S5, or what to improve first.

| Dimension | Questions the review answers |
| --- | --- |
| Channel use | Do correct payloads beat shuffled and zeroed ones at matched length, by how much, and in which task families? A reader that ignores the channel is the main failure to rule out. |
| Usefulness | How much of the gap between no context and full text do payloads recover, at what compression? |
| Stopping | How often is the hard maximum hit? Are there degenerate blocks? Does block length track source information? |
| Continuation | Is text after a block as good as text after the plain-text reference? |
| Ordinary capability | Has Natlang held-out loss, the execution evaluation or general perplexity moved relative to the crisp base? |
| Natlang form | Do literals in eval code and opening scopes round-trip, and does channel use hold on Natlang-rendered tasks? |
| Cache agreement | Do cached and recomputed paths agree on continuations and logits? |
| Latency | Cost per phase, reported. |
| Failure reading | A sample of bad blocks and bad continuations, inspected and categorised. |

## 10. Data dependencies

| Phase | Needs | Source |
| --- | --- | --- |
| A–C | Ordinary text with span labels; Natlang crisp corpus for replay | Raw text and code inventoried by S1 (independent of port records); the current Natlang render |
| D–F | Port records for the main task families: compaction, tool-output digests, purpose-conditioned QA, repository QA and localisation, browsing, memory, SWE traces; write-site contexts; consumer tasks; teacher full-source renderings | S1 port records (first families must pass S1's rendering and leakage checks) |
| §5.3 | Natlang-rendered port examples with eager types | S1 + S0 rewrite passes |
| §8 | Held-out sources for server checks | S1 protected splits |

S3 phases A–C start as soon as S1 inventories raw text, without waiting for port records.

## 11. Compute plan

- **Development** on the 4060: unit tests, cache agreement, tiny runs.
- **Training** on the DGX: phases A–C for three cutoffs (in sequence or interleaved), then the winner's D–F. The DGX also hosts teacher generation for S1 and S2; S3 gets dedicated windows, scheduled with the programme's DGX schedule in the README.
- **Evaluation** runs periodically inside training (held-out losses and ablations) and as separate jobs for the full harness and latency, so the training step stays fast.

## 12. Risks and fallbacks

| Risk | Detection | Response |
| --- | --- | --- |
| Shallow sketches carry too little (small `k` fails) | Sweep §7, consumer quality | Raise `k` only as far as consumers need; the sketch should stay small and quick and has no quality gate of its own (owner 2026-10-06) |
| Reader ignores the channel | Correct ≈ shuffled (§6.1) | Stricter source withholding; tasks with high target entropy and diverse redundancy (Schnitzeljagd's lesson that content-free "format prompts" pass aggregate metrics); payload dropout |
| Representation collapse | Rank and mean-direction monitors (§6.4) | Interface norm (bgkit's lesson), decorrelation penalty, lower learning rates on `P` |
| Stop head degenerates to always-max or always-min | Stopping metrics (§6.2) | Rebalance λ, stronger scored-choice supervision, boundary-label replay |
| Teacher too weak (350M with full source fails a family) | Teacher accuracy per family | Down-weight KL, keep cross-entropy, filter to families the teacher solves |
| Ordinary capability loss | §6.5 | More replay, slower unfreezing, stop phase F early |
| G1 not reachable at 350M | G1 table | Diagnose with the harness; repeat the curriculum on a larger LFM backbone (2.6B) before revisiting the design. A full-depth-feedback model may be trained only as a diagnostic comparison, never as an execution mode |

## 13. Decisions on former open questions

Resolved by the owner on 2026-10-03:

1. **LoRA handling (§1):** the selected v13 LoRA is merged into a frozen crisp base; all S3 changes are separate deltas, and the self-distillation teacher is the model with the deltas off.
2. **Width (§8):** 1024 for `nd:natlang@1`, the model width. Other widths get adapters only if they must share this dialect; otherwise a new version.
3. **G1 (§9):** a review against a rubric rather than fixed thresholds.
4. **Natlang form in S3 (§5.3):** included in S3.
5. **Starting checkpoint:** the crisp checkpoint selected by execution evaluation.

## 14. Work items

In dependency order; items on the same line can run in parallel.

1. Merge the chosen v13 LoRA into the crisp base; record hashes. ‖ Validate `causal-conv1d` and attention kernels on GB10 against the PyTorch path.
2. `model/lfm2_port.py`: layer-range forward, `inputs_embeds` splicing, KV and conv-state snapshot and restore; tests for cache agreement and position handling.
3. `model/heads.py`, `model/dialect.py`; control-token registration on the reserved IDs; tokenizer tests for marker escaping (S0 §3.3).
4. `write.py` and `read.py` with the prefix-payload property test.
5. `eval/` harness §6.1–6.7 with stand-in payloads; `laws.py` hooks with stand-in operators.
6. Data loaders: ordinary text spans with labels; Natlang replay render. ‖ S1 delivers raw text inventory.
7. Trainer, phase configs, parallel scheduled sampling, full unroll with CUDA graphs; run plans and W&B reporting.
8. Probes of `h_k` on the crisp base; cutoff sweep phases A–C at `k = 3, 6, 9`.
9. Port-record loaders and producer/consumer packing; phase D for the sweep. ‖ S1 delivers port records for the main families.
10. Pick `k` (§7); phase E stop policy at the winner.
11. Natlang-form rendering (§5.3) with S0's rewrite passes.
12. Phase F gradual adaptation with replay.
13. G1 review; tag `nd:natlang@1` and hand weights to S4 and S5.
