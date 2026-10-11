# Native trajectory writer-site extraction

## Scope and review

Owner review for this narrow shared-helper change is recorded in the 2026-10-11 root task: use one pure-data write-site extractor for trajectory preparation and donor selection, preserve inline-sidecar validation and all donor checks, and do not broaden selector eligibility. This change does not alter model prompts, task policy, training admission, or frozen runtime inputs.

## Parts

| Part | Form | Reason |
| --- | --- | --- |
| Parse target JSON writer markers and compiler-attested `neuralese_code` sites | Crisp pure-data helper | Exact paths, JSON schemas, and source spans are provenance facts. The inline validator must accept or reject each sidecar as a whole. |
| Resolve legacy path/type/prefix/argument views for trajectory training and evaluation | Crisp adapters over the shared site | Existing training/render call sites depend on these views; they should not reparse the target independently. |
| Find matching donor candidates and enforce schema/body/group constraints | Crisp verifier in `trajectory_probe` | Matching requires exact structural equality and split/source-group protection. This patch centralizes write facts only and does not infer semantic compatibility. |
| Interpret inline natural-language body meaning or decide task quality | Not part of this helper | Preserve the authored source string; this layer performs no semantic judgment. |

## Shared record contract

A parsed site carries its tool name, top-level argument, exact value path, writer name, raw descriptor, source text, declared value type, source kind, and full/native call arguments. JSON-marker paths include the top-level argument. Sidecar sites bind the exact original `function.arguments` JSON to the validated code hash and compiler sidecar source span. The parsed schema is resolved at that same argument/value path. Existing APIs remain compatibility views over this shared record.

Donor matching still requires a real function-body soft ID, exact schema and value path, distinct writer source text, source-group containment, and reciprocal disjoint factual groups. Missing proof remains ineligible with a specific reason.

## Direct writer with a known token sequence — 2026-10-11

The direct-v1 producer reads gold token embeddings during text-sized training;
its emitted values are read by consumers. Known producer tokens can therefore
be processed as a causal sequence instead of repeating a full backbone call
for each token. This is execution plumbing, not a model policy change.

| Part | Form | Reason |
| --- | --- | --- |
| Select a known complete token sequence versus incremental greedy writing | Exact crisp dispatch in shared `write_direct` | The supplied token count must cover the forced write length; greedy decisions remain sequential. |
| Advance the producer backbone over detached ordinary token embeddings | Shared causal backbone sequence call | Same cache, positional shift and discrete token inputs as incremental writing. |
| Emit values from prefix state plus all but the last advanced state | Shared Neuralese projection | Each value describes the next token; no new projection or private trainer. |
| Record stop scores and final states | Shared token-choice readout | Scores align to post-token states, preserving the incremental writer's diagnostics. |
| Qualify outputs, cache and gradients | GPU mechanism diagnostic against the incremental writer | Sequence and decode kernels may differ numerically; measure before adopting the execution change. |

The forced-sequence diagnostic also compares downstream synthetic-consumer CE/logits and combined producer-plus-consumer parameter/prefix gradients; its FP32 reference disables TF32, and it qualifies mechanism parity rather than task quality.

## Declared direct-stage resume and completion — 2026-10-11

Checkpoint restoration belongs to the existing shared trainer. The recipe runner
verifies the same stage, inputs, device and training parameters, binds the full
checkpoint, then invokes that trainer in its existing output directory. A code
handoff uses the existing `--handoff` convention and preserves the earlier frozen
runtime and launch/result receipts. These are exact durability/provenance checks;
no model decides whether a saved optimizer or source identity matches.

Committed checkpoint steps and error counts determine recurrence-stage completion
through one shared reader. A clean child exit after a requested checkpoint is not
completion. Completion remains distinct from semantic channel/task qualification.

## Native direct-recurrence serving handoff — proposed decomposition

### Goal and boundary

A completed native end-to-end recurrence checkpoint must be exportable through the existing direct artifact format and restored by `load_direct_artifact`. The recurrence checkpoint remains the only full-resume authority; the serving export is a separate inference payload and never claims runtime, task, stop, or data qualification by itself.

### Parts

| Part | Form | Reason |
| --- | --- | --- |
| Decide whether checkpoint identity, saved values, parent direct artifact, base, and clean-head anchor agree | Crisp verifier in the existing `train/direct_export` path | These are exact schema, hash, path, and tensor-catalog checks. Missing historical provenance must be supplied as an explicit pinned proof, never inferred from a mutable path. |
| Map one declared recurrence-checkpoint schema to the existing direct-export payload | Small crisp schema adapter in `train/direct_export` | The transform is a lossless field mapping: `backbone_trainables` to `student_parameters`, plus the two projection state dicts, controls, and port configuration. It adds no learning rule or model decision. |
| Restore the exported payload for inference | Existing `load_direct_artifact` | It already restores the pinned base, full declared LFM trainables, projections, and control state with strict checks. Do not route serving through legacy `heads.pt`. |
| Preserve optimizer, moments, counters, and RNG for further training | Existing recurrence checkpoint/resume path | Serving exports intentionally omit optimizer and RNG state; the original full checkpoint remains untouched and resumable. |
| Judge whether the model is a qualified task/runtime/stop policy | Outside this handoff | Serialization success is not qualification or training admission. Existing qualification receipts remain separate and explicitly scoped. |

### Checkpoint provenance change

New recurrence checkpoints that start from `--direct-artifact` should embed the exact loaded parent manifest document and its canonical manifest hash in a dedicated provenance field. On resume, require the current declared parent artifact to match that embedded hash and verify the manifest/payload/source bindings before constructing training state. Keep the checkpoint's existing identity, full optimizer state, and Python/write/stop/Torch/CUDA RNG fields unchanged.

Historical recurrence checkpoints without this embedded binding are not retroactively repaired. A migration/export may use an explicit proof packet that pins the exact checkpoint bytes, stage plan, launch intent, declared `direct_artifact_manifest` input path/hash, and the parent's manifest bytes/hash. The verifier must check that packet as a declared relationship and refuse only when the required relationship cannot be proven; it must not infer parentage from a path name.

### Serving payload mapping

The adapter accepts only the current recurrence checkpoint schema when its options declare direct-artifact training and the required parent proof is complete. It must validate the source identity and trainable policy before mapping:

1. Confirm exact recurrence checkpoint schema, step, checkpoint SHA, frozen runtime/source-code identity, and parent manifest binding.
2. Confirm the parent direct artifact's base binding and anchor. Verify any explicit stage-plan/launch proof against its recorded input manifest SHA and the actual files.
3. Require the full declared LFM backbone-trainable name set; preserve the `backbone.` name prefix expected by the direct artifact loader. Reject missing, extra, colliding, or wrong-shape values.
4. Require strict `neuralese_projection` state. Require `token_choice_projection` when the E2E stage enabled/initialized it; otherwise record absence explicitly rather than fabricating trained values.
5. Map exact saved controls and `port_config`; build the existing tensor catalog and payload; retain the parent source/base/anchor binding while binding the new training checkpoint and export payload hashes.
6. Run the standard strict `load_direct_artifact` restore check on the resulting artifact before publishing its receipt. Keep artifact runtime/task qualification false unless separate matching evidence exists.

### Known compatibility gap to resolve during implementation review

`trajectories` currently checkpoints `natlang.neuralese_recurrence_checkpoint/1`, uses `identity.files`, saves trainables in `backbone_trainables`, and records direct projections/controls. `direct_export` currently expects `natlang.neuralese-direct/1`, `identity.inputs`, and `student_parameters`. It also needs the direct artifact manifest that the recurrence loader used; the recurrence checkpoint currently does not embed it. Thus direct payload tensors are present, but a trustworthy exporter mapping and inherited source binding are not. No current artifact is assumed exportable until this bridge is implemented and a checkpoint-specific proof is reviewed.
