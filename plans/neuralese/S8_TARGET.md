# S8: Target model and release

Draft, 2026-10-03. Detailed plan for stage S8 of the [Neuralese programme](README.md). S8 chooses the backbone and size of the Natlang/Neuralese model from the evidence of S3–S7, runs the full pipeline at that scale, and publishes the model together with its directly trained Neuralese artifacts, serving builds and parity results.

S8 follows the README decisions and the S0 specification. Like the other stages it has no fixed exit thresholds: it evaluates continuously against a rubric (§8), and the review decides when a release is ready and what to improve first.

## 1. Starting point

What S8 inherits:

| From | What |
| --- | --- |
| S0 | The language, `.nz` file format, dialect version tags, graph record, rewrite rules. |
| S1 | Model-neutral port records and `natlang.program/2` executable tasks with lineage, licences and split groups; all execution environments as host services. |
| S2 | The crisp skill-authoring ability, its episode corpus and evaluation rubric. |
| S3 | The port architecture and curriculum on LFM2.5-350M; `nd:natlang@1` (width 1024) as the first dialect version tag; the evaluation harness and cutoff-sweep method. |
| S4 | The runtime, the Python reference server, and the llama.cpp, wllama and vLLM forks (designed for upstreaming), with parity tests against the PyTorch reference; browser storage in OPFS. |
| S5 | Soft program conversion, the replay trainer, the trained combinators and `compose`, and law-agreement measurements that enable compiler rewrites on the current model. |
| S6 | Soft skills initialised from crisp ones, the gradient-tuning baseline, learned updaters. |
| S7 | Task-level RL on fresh rollouts across the validated task mix. |

S8 starts in earnest once S3–S7 have produced evidence on 350M. Preparing candidate backbones (§3) can begin earlier.

## 2. Candidate backbones

Configurations available locally (from `~/.cache/huggingface/hub`, `/mnt/external/bgkit-data/models` and `/mnt/external/hf-cache`):

| Model | Family / type | Layers | Width | Vocab | Full-attention layers | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| LFM2.5-350M | `lfm2`, dense | 16 | 1024 | 65,536 | 2, 5, 8, 10, 12, 14 | The S3 development model. Short-convolution layers elsewhere, convolution cache 3. |
| LFM2.5-2.6B | `lfm2`, dense | 30 | 2048 | 128,000 | 2, 5, 9, 13, 17, 21, 24, 27 | Present locally only as a GGUF (`LFM2.5-2.6B-Q6_K.gguf`); HF weights must be fetched for training. Same convolution cache length. |
| LFM2.5-8B-A1B | `lfm2_moe`, MoE | 24 | 2048 | 128,000 | 2, 6, 10, 14, 18, 21 | HF weights at `/mnt/external/bgkit-data/models/hf/LFM2.5-8B-A1B`. Expert routing runs inside the shallow loop. |
| Ling-3.0-tiny | `bailing_hybrid`, MoE | 24 | 1536 | 157,184 | One full-attention (MLA) layer per group of four, the rest linear attention (KDA); exact indices to be read from the modelling code | 128 experts, 8 active. Natlang's crisp training already has compatibility patches (`scripts/ling_training_compat.py`). |

Other cached models (Qwen3.5-0.8B, Qwen3.6-35B-A3B, Gemma-4-26B-A4B, K2-Horizon) are teachers or references here, not target candidates, unless the evidence changes that.

**Consequences.**

- **Tokenizers differ.** 350M has a 65,536-token vocabulary; 2.6B and 8B-A1B share a 128,000-token vocabulary with each other but not with 350M; Ling has its own. Moving to any larger candidate means registering the control tokens again, re-rendering all training data from model-neutral records, and re-measuring tokens-per-vector compression. Model-neutral S1 records and template-neutral S2/S5 records make this a render step, not a migration.
- **Cutoffs depend on attention placement.** For LFM2-family candidates the S3 rule carries over: candidate cutoffs end just after a full-attention layer (for 2.6B: after layers 2, 5, 9, 13; for 8B-A1B: after layers 2, 6, 10, 14). For Ling, linear-attention (KDA) layers carry recurrent state in place of convolution windows and MLA layers carry compressed KV; cache snapshot and restore, cutoff placement and the shallow loop's expert routing need their own design and S4 server work. That is engineering, not an obstacle in principle.
- **Width differs from 350M.** All larger candidates are wider than 1024. A target either gets its own dialect version at its own width, or, where it must share `nd:natlang@1` with a smaller model, non-square adapters (§4).

## 3. Choosing the target

The final size is not fixed in advance. S8 decides it from evidence, in three steps.

### 3.1 Questions the evidence must answer

| Question | Evidence |
| --- | --- |
| Where is 350M limited? | S3–S7 failure readings: is the limit channel capacity (payloads that do not carry enough), reading, stopping, program-level reasoning, skill authoring, or RL-time exploration? |
| Does scale help that limit? | Short pilots of the S3 curriculum on each candidate (§3.2), compared on the same harness. |
| What does it cost to run? | Latency per phase (prefix, shallow loop, completion, readback) and memory on the serving targets, including the browser through wllama. |
| What runs where? | Whether the candidate is usable in llama.cpp, in the browser, and in vLLM rollouts; MoE and hybrid layers affect all three. |
| What do the crisp results say? | The crisp Natlang student results for the same candidates, so that Neuralese and crisp capability are judged together. |

### 3.2 Candidate pilots

For each candidate that the evidence makes plausible:

1. Port the S3 modules (feedback projection, stop head, content projection, interface norm, control-token embeddings) to the candidate's layer structure.
2. Run the S3 cutoff sweep over the candidate's attention-aligned cutoffs, through phases A–C of the S3 curriculum.
3. Run the S3 evaluation harness on the candidate (channel use, stopping, continuation).
4. Measure a small slice of S5 program-level tasks and S2 authoring episodes with the pilot model.
5. Measure latency and memory on the Python reference server and the llama.cpp fork.

### 3.3 Decision

The review compares pilots against 350M on the S3 rubric, the program-level and authoring slices, and serving cost, and records the choice with its reasons in `DECISIONS.md`. More than one target is allowed: for example a small model for the browser and a larger one for servers, both speaking the same dialect. The decision also states the target's dialect version (§4).

## 4. Dialect for the target

Dialect stability is not a concern yet (README, "Dialects"). The target's dialect is a version tag:

- **Default: a new version.** A target of a different width gets its own dialect version at its own width. Stored 350M-trained artifacts are carried over with `convert` or regenerated from their crisp ancestors, then tuned further (S6).
- **Shared dialect where useful.** A small browser model and a larger server model may share one dialect so their artifacts are interchangeable. Then the model of the other width gets input and output adapters (non-square maps) trained against the shared space, and both are checked with the S3 harness.
- Models declare the dialect versions they speak; mismatched versions are rejected (S0 §12).

## 5. Full pipeline at target scale

Once the target is chosen, every stage runs again on it:

| Stage | At target scale | What carries over from 350M |
| --- | --- | --- |
| Crisp base | The crisp Natlang student for the target, from the crisp training line, re-rendered for its tokenizer; selected by execution evaluation. | All model-neutral data and the crisp corpus. |
| S1 | Re-render port records and executable tasks for the target's tokenizer and template; regenerate teacher data only where a family's coverage was the limit. | All records, lineage, splits, licences, environments. |
| S2 | Train the target on the authoring corpus; evaluate on the same held-out episodes. | Corpus and episodes; 350M-authored skills as crisp files. |
| S3 | Port curriculum on the target with its chosen cutoff. | Method and harness. |
| S4 | GGUF conversion and the llama.cpp, wllama and vLLM write graphs for the target's architecture; parity against the PyTorch reference. | Runtime, protocol, store; fork APIs, which are architecture-general by design. |
| S5 | Replay training of programs and combinators on the target. | Converted programs and graph records; trained combinators as initial values, converted or regenerated for the target's dialect version. |
| S6 | Soft skills and learned updaters on the target. | Soft skills and operators as `.nz` values, converted or regenerated for the target's dialect version and tuned further. |
| S7 | RL on the target with vLLM rollouts on the DGX. | Environments, reward checks, rollout infrastructure. |

Distillation from the 350M line into the target is not needed for the port: the target learns the port itself, and the 350M-trained Neuralese artifacts are converted or regenerated from their crisp ancestors.

## 6. Artifact library

The directly trained Neuralese artifacts are released as `.nz` files, each tagged with its dialect.

### 6.1 Contents

| Group | Contents |
| --- | --- |
| Standard library | The soft bodies of the combinators (`map`, `zip`, `ap`, `combine`, `split`, `splitList`, `read`, `convert`, `gloss`), `compose`, and the iteration predicate and progress-judge functions where they have soft forms. |
| Skills | Soft and mixed skills from S6, in the S2 skill format (`SKILL.md` or `.nz` metadata with `name`, `description` and the `natlang:` block), each with its tests. |
| Data | Trained data blocks that are reusable across programs (rubrics, task knowledge). |
| Conformance | The S0 conformance cases and the server parity sets. |

### 6.2 Versioning and metadata

- Every file names its dialect and version (`nd:natlang@1`) in its header; every export carries its natlang type.
- Releases are versioned as a whole (library version) and per file by content ID; a skill's `provenance` names its episodes, author model and parent revision.
- Each release publishes, per model: parity results, law-agreement measurements and which compiler rewrites are enabled on it.
- Publication: models and `.nz` libraries on Hugging Face; specifications and conformance suites on GitHub.

## 7. Serving deliverables

- **GGUF builds** of each released model with the Neuralese heads, adapters where needed and dialect metadata, in the quantisations that pass parity.
- **llama.cpp fork** release with the `llama_neuralese_*` API and server endpoints, kept as small upstreamable patches against a pinned upstream revision.
- **wllama build** for the browser, with blocks and `.nz` files in OPFS.
- **vLLM plugin** for the target architecture, for high-throughput serving and rollouts.
- **Python reference server** release, as the specification-defining implementation.
- **Runtime release** of natlang with the Neuralese chapter implemented, the standard library `.nz`, and documentation.
- **Upstreaming.** Proposals to llama.cpp and vLLM for the general pieces (mixed token/embedding batches, block-write procedure, model-declared heads and adapters), prepared from the forks.

## 8. Release evaluation rubric

Evaluation runs throughout S8; each review answers:

| Dimension | Questions |
| --- | --- |
| Program correctness | On held-out Natlang tasks and families, how do Neuralese programs compare with crisp programs on the same model, and with the 350M line, at the same inference cost? |
| Channel use | Do correct payloads, skills and soft bodies beat shuffled and zeroed ones at matched length on the target? |
| Exactness and authority | Are exact values preserved, type checks and authority boundaries respected, termination rules honoured? |
| Skills and learning | Do authored and soft skills transfer to held-out families? Do learned updates beat gradient tuning and crisp authoring at matched compute? |
| RL | Does the post-RL model beat the pre-RL model on held-out tasks at matched inference cost? |
| Ordinary capability | Has the target kept its ordinary-language and crisp Natlang capability? |
| Artifacts | Do the converted or regenerated 350M-trained artifacts work on the target? |
| Serving | Do all builds (Python, llama.cpp, wllama, vLLM, quantisations) pass parity? What are latency and memory per phase and per target? |
| Failure reading | A sample of failures, read and categorised, with categories compared against 350M. |

## 9. Risks

| Risk | Response |
| --- | --- |
| Larger models do not improve the limiting dimension | Release 350M as the target, or a small/large pair; record the evidence. |
| Sharing one dialect across widths limits the larger model | Give the larger model its own dialect version (§4). |
| A candidate's architecture (MoE routing in the shallow loop, Ling's KDA/MLA caches) complicates the write procedure or server graphs | Handle it in the S4 forks for that architecture; prefer LFM-family targets if the cost outweighs the benefit. |
| Quantised builds drift from the reference | Ship only quantisations that pass parity; report deviations. |
| Upstream projects decline the changes | Keep maintained forks; the runtime pins releases of them. |

## 10. Decisions on former open questions

Resolved by the owner on 2026-10-03:

1. **Targets:** a small browser model plus a larger server model sharing one dialect is an acceptable outcome.
2. **Publication:** models and `.nz` libraries on Hugging Face; specifications and conformance suites on GitHub.

## 11. Work items

1. Fetch HF weights for LFM2.5-2.6B; confirm Ling-3.0-tiny's attention layer indices and cache structure from its modelling code; record all candidates' configurations.
2. Generalise the S3 modules and training loop to arbitrary LFM2-family depth, width and attention placement, and to MoE layers in the shallow loop.
3. Re-render pilot data for the 128,000-token vocabulary and register the control tokens there.
4. Run candidate pilots (§3.2) and the decision review (§3.3); record the decision.
5. If needed, design Ling's write procedure and caches, and extend the S4 forks.
6. Assign the target's dialect version; train adapters only if it must share a dialect with a smaller model; train `convert` where stored artifacts are carried over.
7. Run the full pipeline at target scale (§5).
8. Assemble the artifact library with metadata and parity results (§6).
9. Produce GGUF builds, fork releases, the wllama build and the vLLM plugin; run parity.
10. Write model cards and datasheets: architecture, dialects, training data families, evaluation results, known failure categories, intended use.
11. Prepare upstreaming proposals from the forks.
12. Release review against the rubric (§8).
