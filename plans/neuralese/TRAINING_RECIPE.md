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
   certificate explicitly has `runtime_qualified: false` until this stage exists
   and passes; it cannot qualify the old marker/RMS channel.
4. **Compression and recurrence:** consume the exact qualified weights. Keep text
   replay and requalify the channel when backbone deltas change its states.

The shared runner implements stages 1–3, including certified raw-port checkpoint
construction and actual serving encode/read/write and gradient replay controls.
The raw trajectory trainer rejects unqualified handoffs. A declared stage handler
for recurrence/compression remains work in progress. This is an explicit foundation, not a claim that the entire pipeline
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
A cutoff15 candidate is being distilled independently; its strong aggregate score
cannot mask its weaker source stratum and cannot qualify a different full-depth
runtime. Legacy RMS soft parameters must not initialize raw-token ports; re-encode
their source texts. Raw native GGUF export is rejected until the native runtime
implements this transport.


Certificates resolve adjacent stage reports after a registered recipe directory
is relocated. They still validate the exact report/heads/feedback hashes; no
path rewrite or weaker admission is necessary. Pass the relocated checkpoint
explicitly to `foundation_port`. `runtime_qualification/heads.pt` is the shared
serving/trajectory input after a complete recipe run.
