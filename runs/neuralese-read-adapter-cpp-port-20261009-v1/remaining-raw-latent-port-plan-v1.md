# Raw-token and latent-sketch native port plan

Status: plan only. No native code, builds, model downloads, or GPU jobs were started for this work.

## Current boundary

The new `full-residual-v1` map is an optional read-port step in the existing legacy projector. It is applied by `nz_read` after the legacy interface RMS norm. `nz_interface` remains the encoding operation and does not apply this map. The parity receipt covers only that legacy sidecar, one CPU build, and a nonzero LFM2.5 adapter.

The checkpoint exporter explicitly accepts only `legacy-rms-v1`. The `export_heads_gguf` tensor layout and C++ `nz_heads_load`/`nz_write` graph still assume legacy fields: interface RMS weights, vocabulary feedback readout/table plus MLP, the positional stop MLP, and the residual content projection. The current read path also assumes the legacy interface transform. Merely removing the Python refusal would emit the wrong schema or fail on profile-specific modules.

## Profiles and required work

| Profile | Python head behavior | Native work before claiming support |
| --- | --- | --- |
| `raw-token-v1` | No block markers or interface RMS. Feedback is `CausalFeedbackProjection`; content can use learned residual or declared raw identity. The current foundation recipe uses full-depth token-aligned output and raw next-token embedding targets. | Add a versioned projector profile and export the actual trained causal projection, required frozen-table/base identity, content transport, stop mode, control rows, and epsilon metadata. Add profile-specific loader validation and a raw read path that does not RMS-normalize. Implement the full-depth writer path without assuming a nonempty upper-layer completion range when `cutoff == depth`; preserve the raw token/output projection behavior. Keep runtime marker/control-token handling distinct from legacy marked blocks. |
| `latent-sketch-v1` | No block markers. Small normalized state plus learned correction; payload projection references the top state. Stop is the learned `StopHead`; v1 uses same-position top states. | Export/load its learned sketch correction, content reference/correction and scale head, stop tensors and stop source as an explicit profile. `nz_write` must preserve the v1 position alignment and shallow-to-full completion semantics. Define how the optional reader adapter composes with this profile, independently of the legacy interface norm. |
| `latent-sketch-v2` | No block markers. Autoregressive sketch; each payload is predicted from the previous top state, and the close token is selected by the backbone output head rather than a learned stop MLP. | In addition to v1's profile-specific maps, implement the shifted state/payload alignment, close-token log-odds stop, and correct empty/truncation behavior. Do not reuse the positional `StopHead` or v1 same-position content path. |

## Shared artifact contract

Use one versioned sidecar schema for all profiles, with explicit `profile`, `schema_version`, embedding width, model-family/base identity, control-token IDs and tying policy, cutoff, payload cap, all norm epsilons, stop source/mode, content transport, and reader-adapter variant/epsilon. Each profile declares the exact required tensor-name/shape/dtype set. Reject missing, extra profile-specific, wrong-rank, wrong-dimension, non-finite, or inconsistent metadata before exposing a port. Keep model GGUF and projector GGUF identities joined in the export receipt. The exported reader map is trained head state, not backbone state.

Keep the shared request-facing API (`encode`, `read`, `write`, and typed block store) stable where behavior is identical. Route only the internal read/write math by the declared profile. `encode` remains ordinary text tokenization/embedding; it must never receive the Neuralese reader map. Read adapters consume only block vectors. Reference Python semantics and C++/WASM must share the same order of interface transform and adapter for each supported profile.

## Backbone coverage and constraints

- LFM2.5 is the only architecture for which this fork currently reports `llama_model_supports_layer_range() == true`; the partial graph is implemented in `src/models/lfm2.cpp`. Legacy cutoffs depend on that split.
- llama.cpp has a Mellum model implementation, but this fork's layer-range capability check does not include Mellum. Full-depth raw-token support may avoid a split only if the complete output-state/control-row path is exact. Any shallow cutoff or latent sketch profile on Mellum needs verified range execution and cache/convolution behavior first.
- LFM2 parity does not qualify Mellum, and native support does not establish the new foundation's runtime/gradient qualification. Preserve per-backbone model export, token-row, full-depth state, and exact-reader parity checks.

## Sequence and acceptance

1. Inspect current Python `raw-token-v1`/latent fixture checkpoints and name every required state tensor versus frozen tensors re-derived from the exact model GGUF. Do not duplicate or silently substitute the backbone tables.
2. Agree one shared schema/version and profile tensor table before editing. Keep the legacy sidecar byte/schema behavior stable when the optional adapter is absent.
3. Implement `raw-token-v1` first because it is the current foundation requirement. Add a full-depth-only LFM2 CPU parity fixture with a deliberately nonzero projector; verify open/close and raw payload positions against Python, not only tensor loading. Add strict malformed-sidecar rejection checks.
4. Implement latent v1 and v2 as separate profile branches only after raw parity. Validate position shifts, v2 close-token decision, stop source, max-length truncation, and content payloads with deterministic CPU parity.
5. Add WASM support and parity only once the shared tensor schema and operators compile under Emscripten. No WASM compiler was available for the legacy adapter check.
6. Repeat exact end-to-end export/read/write and semantic task qualification for each claimed backbone/profile. Keep all currently unqualified foundation flags false until those gates pass.
