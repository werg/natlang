# Cotangent accumulator backends (2026-10-19)

## Natlang-native design

A training recipe declares where exact low-precision parameter cotangents are
accumulated. The existing shared cotangent operation receives each gradient
contribution, adds it in FP32, and sends one once-cast result to the original
parameter. This policy is a declared recipe choice, inherited by its resumed
runtime and by read-only qualification/VJP calls. It never changes according
to an environment variable or an allocation failure.

The default `cpu_file` route keeps the existing sparse, file-backed CPU
accumulator. The optional `device_fp32` route places each lazily used FP32
accumulator on that parameter's CUDA device. Both routes use the same named
parameter/view bindings, contribution ordering, embedding frequency semantics,
anchor and higher-order paths, and scope cleanup. A scope reports allocated
accumulator bytes to its capacity evidence; recipe/checkpoint identity records
the backend. A backend change is a declared continuation transition retaining
optimizer state, RNG and exposure, and needs a new capacity qualification for
the exact device/model/update shape.

## Exact components

1. The recipe/trainer selects `cotangent_accumulator_backend` from the explicit
   stage option and records it in checkpoint training options. A missing value
   means the historical `cpu_file` behavior. Resume rejects an unrecorded
   change.
2. Every `ParameterCotangentScope` constructor in supervised/RL/preference
   updates, actor/functional VJPs, complete-gradient checks, attention checks
   and capacity checks receives the same declared selection. Device placement
   is validated against the bound parameter; there is no silent fallback.
3. The shared accumulator owns exact allocation, views, embedding frequency
   scratch, allocation telemetry and close-time release. Numeric accumulation
   stays FP32 and `finish` retains the existing once-cast semantics.
4. Memory qualification records backend identity and peak allocated bytes along
   with the existing model, optimizer, source, objective and device bindings.
   A CPU-file certificate cannot qualify a device accumulator, or vice versa.

The code is crisp because tensor allocation, device identity, autograd edges,
checkpoint persistence and capacity measurements require exact machine
semantics. Policy choice remains a natural-language recipe declaration. No
model-facing text changes.

## Validation

CPU-file regression tests remain unchanged. Device tests compare CPU-file and
CUDA-FP32 accumulation on real BF16 module parameters, repeated calls,
optimizer updates, embeddings (including frequency scaling), views/expert
banks, higher-order references and cleanup. They bind RNG and optimizer
state before/after. Capacity tests assert exact backend identity and allocation
accounting. Live runtime activation is outside this change.
