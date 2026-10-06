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
   embeddings. Distill shallower-state corrections only as separately qualified
   efficiency variants. Raw embeddings are not RMS
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


## Learned residual continuation

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
has run. The active learned-residual phase keeps one vector per token and no
compression pressure; it trains the content representation/child-return channel.
