# Shared neuralese training recipe

New neuralese training lineages start with the declared recipe at
`training/neuralese/recipes/foundation-v1.json`. Machine-specific experiment scripts
must not bypass its foundation gates. Code moves through `origin/main`; data and
checkpoints move through the corpus registry and immutable manifests.

## Required order

1. **Token-aligned identity:** raw token embeddings, unchanged positions and a
   zero-residual content projection must reproduce the ordinary model's logits.
2. **Causal embedding distillation:** freeze the chosen backbone. Initialize the
   feedback reference from its actual full-depth output head and normalization;
   output at position i predicts the input embedding at position i+1. Distill a
   full-depth reference first; its zero correction gives exact next-token
   embeddings. The out port is at the top layer: the foundation's reference is
   `cutoff: "full"` on every backbone. A shallow cutoff belongs to the sketch, the
   inputs generated autoregressively through the shared shallow layers only, so a block
   is written without generating its inputs through the whole transformer (between a
   perceiver and the ordinary autoregressive model); the sketch needs no autoregressive
   input fidelity and is trained through consumers, not by shallow next-token
   distillation (owner 2026-10-06). Raw embeddings are not RMS
   normalized. Training completion is not qualification.
3. **Runtime qualification:** validate the actual production encode/read/write
   path, token boundaries, gradient replay and typed task execution. The foundation
   certificate has `runtime_qualified: false`; the separate raw-port checkpoint
   records its passed runtime scope. Neither qualifies the old marker/RMS channel.
4. **Function execution and recurrence:** consume the exact qualified weights for
   soft instructions/arguments, child results and skill use. Keep text replay and
   requalify the channel when backbone deltas change its states. Compression is an
   explicit task/operator (for example a digest or a summarizing natlang lambda),
   not a mandatory global curriculum or fixed ratio for ordinary calls.

The shared runner implements stages 1–3, including certified raw-port checkpoint
construction and actual serving encode/read/write and gradient replay controls.
The raw trajectory trainer rejects unqualified handoffs. A declared recurrence handler is available (see below); stopping and broader soft-function qualification remain subsequent gates. Explicit
compression operators require their own task qualification when introduced. This is an explicit foundation, not a claim that the entire pipeline
has already been harmonized. Existing A–F/trajectory CLIs remain legacy research
entry points and their old checkpoints are not automatically qualified.

## Run and extend

```sh
PYTHONPATH=training/neuralese python -m natlang_neuralese.train.recipe \
  --recipe training/neuralese/recipes/foundation-v1.json \
  --heads HEADS.pt --records RECORDS.jsonl --pieces PIECES.jsonl \
  --out runs/NEW-RECIPE-LINEAGE --device cuda
```

`--inspect` validates the declaration without starting a model.
`--until token_identity` stops at an explicit stage boundary. Run the same command
to resume: the runner freezes the Python package, pins recipe/data/code identities,
checks previously qualified artifacts and restores child optimizer/RNG state.
SIGTERM is forwarded to the active trainer, which saves at a step boundary.

Recipes declare named stages, implementation kinds, dependencies and typed
parameters. Add a shared handler and its validation/tests to `train/recipe.py`,
then add a stage declaration. The runner executes Python modules with argument
lists, not shell command strings. Unknown implementations/parameters, unsafe stage
paths, missing identity dependencies and failed gates are rejected.

The result is `foundation-certificate.json`, which binds the exact frozen backbone
heads and feedback checkpoint hashes, recipe and passed stage reports. Downstream
code uses `require_foundation` to check that handoff. Changes to input weights or
reports invalidate it. Reconfiguration creates a new lineage rather than silently
resuming under changed inputs. The raw-port handoff consumes this API and creates fresh token-preserving heads.
The runtime report qualifies fixed-length transport/replay only; autonomous stop
selection and semantic compression require subsequent training and evaluation.

## Current experiments and decisions

On Pop, full-depth initialization is verified bit-exact. This uncovered a real
numeric discrepancy: LFM's native final norm rounds to BF16 **before** multiplying
its gain, while the old generic RMSNorm did so afterwards. The bootstrap copies
the actual normalization module and reproduces separate control-row projections.

The initial cutoff14/40-source experiment reaches62% held greedy agreement and
fails qualification. Adding128 admitted trajectory-context windows improves the
aggregate held score to82% at2048steps, still below the declared90%/KL.25 gate.
Context windows are projection-training examples, not additional independent task
facts or a serving context-size restriction. Exact train-source duplicates are
excluded from held scoring with a receipt. Further work must report source and
context strata separately so easy prompt tokens cannot mask a weak value channel.

The temporary inverse experiment `h_final[i] -> E(token_i)` is not this causal
distillation stage and has failed its reconstruction diagnostic.


## Full reference and optional shallow optimization

The default recipe uses `cutoff: "full"` and stops immediately if the exact
initialized projection meets all gates. On Pop this qualified 132,883 held
positions (source and context strata) with zero KL and exact token agreement,
without optimizer updates. It still saves full optimizer/RNG state for continuation.
A cutoff15 candidate was distilled independently and is paused; its strong aggregate score
cannot mask its weaker source stratum and cannot qualify a different full-depth
runtime. Legacy RMS soft parameters must not initialize raw-token ports; re-encode
their source texts. Raw native GGUF export is rejected until the native runtime
implements this transport.


Certificates resolve adjacent stage reports after a registered recipe directory
is relocated. They still validate the exact report/heads/feedback hashes; no
path rewrite or weaker admission is necessary. Pass the relocated checkpoint
explicitly to `foundation_port`. `runtime_qualification/heads.pt` is the shared
serving/trajectory input after a complete recipe run.

## Declared recurrence extension

`recipes/raw-recurrence-v1.json` adds `raw_recurrence_training` after the qualified
runtime stage. Its heads argument comes from that predecessor's exact artifact,
not the original legacy heads input. The runner freezes shared code and validates
that the final full optimizer/RNG checkpoint belongs to that runtime handoff.
Completion requires the declared steps and zero recorded errors; it explicitly
**does not grant semantic channel admission**. Learned weights need task/stopping
qualification, not inheritance of their initializer's certificate.

This declaration starts with one vector per source token and a512-vector writer
ceiling, without global compression pressure. Payload length follows the function result;
compression is trained only when an explicit task/operator calls for it.
It is a reference for new runs. The Pop experiment began at2tokens/vector; its
full-state warm-up continuation explicitly declares a change to1token/vector and
producer supervision, with frozen packages and pinned parent checkpoint hashes. The fourth handler's declaration/gate is
covered by tests; a complete long four-stage run has not finished yet. The
three-stage foundation/runtime declaration has completed end to end on Pop.


The producer-supervised recurrence warm-up trains the ordinary gold producer reply
under its actual opaque ancestor context. Its own output is not substituted into
that gold reply. This supplies a direct source-value objective before compression.
Structured raw writes share the native typed-value prefix across serving, training
and execution evaluation. Runtime qualification additionally checks that prefix
against ordinary greedy generation and tests opaque host-argument restoration.
These transport checks still do not qualify autonomous stopping or task success.


## Historical full-depth residual reference continuation

`recipes/learned-residual-curriculum-v1.json` declares the transition from a
qualified learned raw-parent checkpoint to a trainable content residual at one
vector per token, retaining the fixed corpus and full optimizer/RNG. This is a
curriculum transition declaration consumed by the trajectory CLI; the four-stage
recipe runner does not yet orchestrate continuation transitions automatically.
The CLI requires explicit `--curriculum-change content_transport` and
`--curriculum-change content_residual_initialization`, with
`--content-residual-initialization fresh-zero`. An existing raw-identity checkpoint
may contain nonzero *bypassed* residual weights. It must not activate those stale
weights. Fresh-zero resets only the content projection weight/bias and their
optimizer slots after full-state restoration. In-place resume preserves the
learned projection and never repeats initialization. `eval.residual_transition`
qualifies the actual starting channel against its exact parent; subsequent learned
weights do not inherit this certificate. This phase has no compression pressure.


## Owner clarification: compression is an operator, not the training goal

2026-10-06: ordinary neuralese natlang lambda calls are not required to compress.
The current priorities are soft instructions and arguments, typed child-result
transport/recurrence, skill discovery/use and task execution. Large-input digest
or summarizing lambda tasks can supply explicit compression examples separately,
with their own output contract and evaluation. Do not impose a global shortened
payload ratio or defer ordinary function training until a compression curriculum
has run. The full-depth residual reference kept one vector per token and no
compression pressure. It was safely stopped at step2851 after the sketch
architecture correction below; it is not the target shallow-sketch lineage.


## Autoregressive shallow latent sketch — owner correction 2026-10-06

Use `latent-sketch-v2` for fresh lineages. Retain the certified full-depth causal
reference and its parent LoRA; initialize a small vocabulary-free shallow sketch
and a zero input-space output residual. There is no global compression objective
or independent shallow next-token fidelity gate.

Layout is like text: payload p[j] projects the top state at position j-1;
p[0] comes from the last context position, without an opening marker. Sketch s[j]
comes from the shallow state at j-1 and predicts that same-slot payload. The last
completed top state predicts the close token via the backbone LM head. Greedy
ordinary token inputs supplied to the initialized channel reproduce the token
embeddings in their normal slots. This does not assert that a fresh random sketch
already generates good consumer content.

Use `write_generated(..., sketch_gradient="one_step")`: detached greedy rollout,
then a parallel one-step rerun, with positive same-slot sketch self-target on
all positions including the first. The dedicated F projection receives the full
auxiliary gradient; source-state/backbone gradients are attenuated with
`sketch_target_backbone_scale` (default0.05;0 freezes this path,1 restores full
auxiliary input gradients). Consumer gradients are unchanged and the payload target
is detached. Do not train full recurrent BPTT by default.
`one_step` cuts recursive feedback-projection rollout gradients; it retains ordinary
causal attention/convolution gradients through the recomputed sketch inputs.
It is not a strict one-stage horizon for those cache paths.
The trajectory CLI requires explicit `--sketch-gradient one_step`, positive
`--sketch-target-weight`, and `--train-control-rows` for close-token supervision.
Reference weights remain frozen. Save/restore control rows with optimizer/RNG.

Fresh initialized v2 runtime diagnostics passed on the actual Pop 350M GPU:
`runs/neuralese-latent-sketch-v2-diagnostic-20261006-v1`. Exact writer/cache/public
producer and input-gradient/typed-wire checks are separate from consumer task
quality and autonomous stop qualification. Changed weights require requalification.
Historical v1 projections used each position's own state and a separate stop head;
those checkpoints and certificates cannot qualify or resume v2.

Before freezing a new run, audit deferred ready corpus conversions, source/split
closure, and performance work. Record both inclusion and exclusion decisions in
the declared recipe/run receipt. Broader unreviewed corpora are not automatically
admitted. C++ serving does not yet implement v2; use the shared Python runtime.

### One-stage sketch credit (`local_stage`)

Keep full-stack self-target distillation, and give each sketch direct consumer
credit through the one full-stack position it feeds. Under v2 indexing, sketch
`s[j]` enters position `j`, whose completed state emits payload `p[j+1]`.
Later generated positions replay detached generated **inputs**, without using
the differentiable sketch branch's cache. Both attention KV and convolution
history therefore cannot reach earlier sketch outputs. Original prompt/scope,
child-result and shared backbone parameter adjoints remain connected through the
fixed-input history. Shared F parameter gradients sum over local uses; cutting
individual output paths does not freeze F or the backbone.

Qualify forward values/cache replay and per-sketch gradient support separately:
local completion must reach its sketch, later completion must not reach it via
generated history. `write_generated(..., sketch_gradient="local_stage")` and the
trajectory CLI's `--sketch-gradient local_stage` implement isolated stage replay. `--local-stage-batch-size 1` is the sequential
reference; a positive explicit size groups independent branches with a shared
immutable prefix (singleton prefixes use expanded views). Earlier positions
inside each branch use fixed inputs; causal attention prevents later positions
from influencing its selected completion. This preserves the same truncated
operator while reducing per-position kernel launches. Groups4/8 pass FP32
adjoint comparison with the reference; group16 passed the actual350M GPU probe. Positive same-slot self-target and source gradient scale0.05 remain.
The exact greedy rollout supplies forward states; isolated local replay supplies
the truncated adjoints. This avoids BF16 layout rounding changing discrete
payload choices. `Written.local_replay_max_abs_error` exposes replay discrepancy;
these are explicitly surrogate adjoints, not exact full-BPTT Jacobian parity.
Active stochastic dropout is rejected. Batched unequal lengths use valid masks.

Check `tests/neuralese/test_local_stage.py` and the exact-weight GPU diagnostic:

```sh
python -m natlang_neuralese.eval.local_stage --checkpoint CHECKPOINT \
  --out FRESH_OUTPUT --lengths 8 32 --context-tokens 256
```

Report execution support, forward parity, numerical discrepancy, time and memory
separately from consumer quality/autonomous stopping. Groups add extra full-stack work and transient memory; choose their size under
the machine memory budget. Pop350M128vectors/4Kcontext: group16forward+backward
2.322s vs1.550s oldmode, peak3.048GiB; group8uses2.446GiB. These are execution
diagnostics (including reference rollout), not whole-training throughput.
No silent changes to active pinned jobs: a new stage uses `--continue-from` plus
`--curriculum-change sketch_gradient`, retaining optimizer/RNG/data and recording
the parent SHA. Existing resume still rejects changed controls. See the declared
`local-stage-sketch-credit-v1.json` option; it does not qualify a trained channel.
