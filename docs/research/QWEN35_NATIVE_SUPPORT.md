# Native Qwen3.5 support

Natlang's native decoder path supports the text stack of Qwen3.5, including its
alternating full-attention and Gated DeltaNet layers. The outer conditional
generation wrapper is loaded for checkpoint compatibility; its vision tower is
not executed or selected for text-backbone training.

## Gated DeltaNet imports

Each imported transition carries the producer's pre-convolution projected
`mixed_qkv`, `z`, `b`, and `a` tensors. The consumer applies its own causal
convolution and recurrent transition over the imported logical token sequence,
then continues through its owned tokens. It never imports a terminal recurrent
snapshot or a producer convolution window. Consequently, the producer's private
prefix is not transferred as explicit state, while its influence on the
producer hidden states and projected payload remains.

Imported query and `z` channels are currently retained in the wire payload for
schema consistency, though consumer readout is evaluated only for owned tokens
and those imported channels have no direct consumer-loss gradient. A compact
payload would require a versioned schema change. A producer's query/`z`
parameters can still affect higher-layer exports through the producer hidden
state.

## Training precision

Fresh Qwen3.5 native loads select
`fp32-norms-qwen35-delta-controls-v1`: layer norm gains and the low-dimensional
DeltaNet `A_log` and `dt_bias` controls use FP32 storage. This lets the default
native optimizer at the production learning rate update the recurrent dynamics
instead of rounding updates away in BF16. An inherited `fp32-norms-v1` setting
resolves to this Qwen3.5 profile; LFM and Mellum keep their existing profile
behavior. The effective profile is stored in model identity and optimizer
parameter bindings and checked during restore and raw-policy validation.

## Validation and limits

CPU tests cover HF full-attention parity, DeltaNet batched and incremental
serving, ragged padding, graph imports and branches, stop-gradient behavior,
serialization, actor gradients, generic SFT/RL objectives, adapter restore,
optimizer updates, and FP32 control save/resume. The text-only implementation
does not qualify Qwen3.5 vision execution. No real Qwen3.5 checkpoint has been
downloaded or exercised on a GPU, and no new training job has been launched.
Qwen-specific cohort scheduling remains part of a future integration recipe.

On 2026-10-23, the full CPU Python check completed with 5,101 passed, 526
skipped, 17 deselected, and the three baseline runtime-handoff failures below.
Its TypeScript builds, browser types, and layering check passed. The full
TypeScript runner then reached `neuralese-learning.test.mjs`, which loads the
cached 350M LFM checkpoint and is documented in `check-main.sh` as not
finishing on CPU; that isolated test process was stopped under the ledger. The
focused native-state-store suite passed all five tests, including the gated
DeltaNet portable-wire case. The final quick check also passed its 22 Python
ratchets and TypeScript check.

The full check also encounters an existing runtime-handoff failure, independent
of this port. At `9538bf0b95d08acb924f3ee0177ba23b7d602209`, the same narrowed
CPU test fails with `ValueError: runtime handoff current resident plan changed
its train/held source binding` in
`native_runtime_handoff._current_adopted_resident_directory`. The failing
parametrizations are `test_actual_later_resident_adoption_does_not_replace_runtime_authority`
for `[None]`, `[lfm]`, and `[mellum]`; see ledger service
`qwen35-handoff-head-r43-20261023.service`. The runtime-handoff issue remains
owned by its maintainer.
