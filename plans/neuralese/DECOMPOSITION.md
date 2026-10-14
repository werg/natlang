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

## Native causal-convolution dependency selection — 2026-10-12

The LFM2 forward keeps one causal-convolution call site. On the Pop executor it may use the official Dao-AILab `causal_conv1d_fn` source build only when the exact pinned Torch/CUDA build, package version, and SM 8.9 device match its immutable build manifest. The invalid Hub revision is not resolved or substituted. Any absent or mismatched package selects the existing `torch.nn.functional.conv1d` reference path. This is exact ABI and runtime plumbing, not a model policy change; same-input CUDA fidelity and speed remain a separate adoption gate.

| Part | Form | Reason |
| --- | --- | --- |
| Check that the installed extension matches its package, Torch/CUDA, and device pins | Crisp exact loader in shared `model/lfm2_port.py` | These version and architecture fields determine whether the compiled extension can load and execute. |
| Apply causal convolution when the exact native package is available | Existing `PortBackbone` convolution site with `causal_conv1d_fn` | Keep one model path and preserve its existing inputs and weights. |
| Use the reference convolution otherwise | Existing `F.conv1d` branch | It already implements the same operation and remains available for every runtime. |
| Record which implementation and artifact served a run | Existing runtime/evaluation receipts | Bind performance and parity claims to the actual loaded implementation and source artifact. |

## Direct SFT window gradients and memory choice

Direct SFT still uses one global weighted objective and one optimizer update for a packed schedule batch. The default `window_execution=auto` backpropagates independent windows sequentially, accumulating parameter gradients and applying MuonSR/LionSR/AdamW once after all windows. It asks the shared `backbone_memory_layout`/`geometry_bytes` estimator for the largest-window activation geometry, combines that with the shared optimizer gradient/state floor and live reusable CUDA memory, and selects uncheckpointed execution when the predicted increment fits; otherwise it uses layer checkpointing. The estimate includes promoted FP32 RMSNorm inputs retained for backward on uncheckpointed layers and the largest layer-local promoted norm input while checkpointing. `AdaptiveGraphMemory` supplies its calibrated geometry ratio and 5% prediction margin. Every update records the plan before its forward, including selected checkpoint mode, activation layout, estimate, and reusable CUDA bytes; successful updates also record the observed peak and update calibration. `packed` remains an explicit diagnostic override. The recipe's `checkpoint_layers` may pin a mode operationally; omitting it enables the adaptive choice.

| Part | Form | Reason |
| --- | --- | --- |
| Select packed or per-window gradient accumulation and checkpointing | Shared crisp resource policy in `train/direct_sft.py` and `train/memory_estimator.py` | This is a resource choice based on declared activation geometry, optimizer storage, and live CUDA headroom; it does not change the loss or the one-update batch semantics. |
| Compute each window's existing CE and value losses | Existing `forward` objective with global denominators | The shared loss definitions and weights remain unchanged. |
| Accumulate gradients across windows | Ordinary autograd `backward()` per window, shared gradient buffers | Each window graph is freed before the next forward; the optimizer remains unarmed until the complete packed update is ready. |
| Clip gradients and update MuonSR/LionSR/AdamW | Existing optimizer path, once after accumulation | Optimizer state, update count, global clipping, and packed batch semantics remain one update. |

## Experimental FP8 training backend — 2026-10-12

The experimental backend is an opt-in implementation detail of the existing
`neuralese_direct` trainer. Normal recipes remain BF16. Transformer Engine
handles eligible dense LFM linear operations under an explicitly declared FP8
recipe; BF16 parameters remain the master weights and the existing MuonSR,
LionSR, and AdamW path owns all updates. This experiment is not a production
precision policy and does not change the serving artifact.

| Part | Form | Reason |
| --- | --- | --- |
| Select eligible LFM dense layers and validate their shapes/runtime | Crisp exact adapter in shared training code | TE availability, module paths, dimensions, and SM/Torch/CUDA compatibility are exact facts. |
| Run supported dense GEMMs in FP8 | Existing shared trainer with TE autocast and an explicit recipe | This changes numerical execution while preserving the shared SFT objective and optimizer. |
| Keep BF16 master parameters, optimizer state, and non-eligible operators | Existing shared model and optimizer | Attention softmax, norms, convolutions, embeddings/readout, and Neuralese projections do not silently change precision. |
| Save FP8 scaling state and activation/cache accounting | Direct checkpoint and training receipts | Exact resume requires scale history; memory reports must distinguish FP8 caches from BF16 parameters and saved tensors. |
| Compare mixed short/long updates, evaluation, memory, and speed | Dedicated paired diagnostic over the shared direct recipe | This measures an experimental backend; it neither changes production defaults nor qualifies a model. |

`natlang_neuralese.eval.fp8_training_comparison` prepares two pairs from
declared `neuralese_direct` templates and is prepare-only unless `--run` is
passed. It requires the short and long templates to share exact source inputs,
lineage evidence, and non-context training options. Its summary separates the
first three cold updates from warmed median update time, checks that each pair
used the same source windows, and retains GPU clock/power/memory samples,
initial/final held evaluations, gradient summaries, and saved-tensor inventories.
This inventory and the configured `save_original_input=False` are evidence to
inspect; no FP8 activation-memory reduction or quality/speed result is assumed
before a scheduled CUDA run.

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

## Declared projection-rate continuation — 2026-10-11

The optimization decision is to compare gentler NL/TCP updates when training
loss collapses before held performance improves. The machine owner declares
that decision in a recipe; the mechanism neither selects the rate nor promotes
the resulting checkpoint.

| Part | Form | Reason |
| --- | --- | --- |
| Choose a smaller projection rate and assess held soft/crisp quality | Owner optimization judgment, recorded in the recipe and handover | This is experimental policy, not an automatic numerical rule. |
| Declare `heads_lr` as a continuation change with fixed inputs | Existing recipe/trainer interface | Bind the changed objective controls without replacing the trainer. |
| Restore every optimizer child including LionSR/MuonSR | Shared crisp named-state mapper | Slot identity, parameter shapes and moment preservation require exact checks. |
| Apply the declared rate after restore, splitting mixed groups if needed | Shared crisp optimizer plumbing | A saved group rate must not silently overwrite the declared new rate; preserve moments and all other group settings. |

Earlier named receipts omitted latent names while saving their full optimizer
state. Recover only that omission from exact one-parameter latent group names;
record the recovery explicitly. Unknown slots remain an error. No optimizer
reset or checkpoint mutation is implied.

## Complete task evidence in E2E acceptance — 2026-10-13

The natural-language acceptance policy remains the declared same-task comparison:
1. Read the text, value and parent task reports.
2. Compare only completed, eligible runs of the identical packet and case indices.
3. For each fused case, require an explicit Boolean outcome in both arms; missing
   outcomes are unknown, never failed tasks with an assumed zero rate.
4. Compare measured success rates against the existing declared margins.
The crisp verifier checks these exact report invariants. The existing task-report
adapter carries the packet hash from TaskEval. No serving prompt or training
objective changes. Legacy reports without packet identity remain pending.

## Logical tokenizer twins and nonempty union publication — 2026-10-13

VerifyTokenizerTwins takes rendered rows for one logical document and returns
whether they represent the same admitted source:
1. Require the same document ID, split, source groups and source-record sequence.
2. Require identical source-record hashes and context/action attestations where
   present, using the existing tokenizer-twin converter's document identity.
3. Permit different rendered text and token boundaries: each backbone has its
   own chat template. Keep both text hashes for exact cross-member deduplication.
4. Preserve existing held-source and duplicate-ID rules over eligible twins.
The existing union planner implements these exact verifier checks; no rendering
or model prompt changes. Hashing and row streaming remain crisp mechanisms.
PublishTrainingUnion requires nonempty training output for every declared
rendering/tokenizer before a receipt can grant training admission. A zero-output
build is an implementation failure to repair, not an empty source-data decision.

## Account for trajectories without authenticated edges — 2026-10-13

AccountForMinedRows preserves the existing whole-trajectory mining policy:
1. Authenticate producer-to-consumer references using the existing exact verifier.
2. Keep every record of each trajectory that has at least one authenticated edge.
3. For every other record, retain an existing failure reason or record that its
   trajectory has no authenticated producer-to-consumer edge.
4. Count those trajectories and rows in the shard receipt; retain original sources.
The Python miner is crisp accounting over the verifier's output. This adds no
eligibility rule, prompt, renderer change or new training admission. A completed
zero-edge conversion must not look like lost or unprocessed input.
Record unreadable source rows before authentication. A nonempty input producing
no parsed records is a failed conversion, not a successful zero-edge trajectory;
keep independently converted rows from a partly unreadable input.

## Cancel a queued resource request — 2026-10-13

The caller supplies the unit to cancel; this adds no scheduling decision.
1. Stop that unit's detached waiting client, if loaded, so it cannot renew the request.
2. Remove the durable queue entry and record the existing cancellation event.
3. Leave the execution unit and running claims alone.
The memory ledger performs these exact systemd and durability operations. Stopping
only the queue entry allows its still-running client to put it back and prevents
an immediate relaunch with a corrected reservation.

## Continue Mellum on the shared R5 corpus — 2026-10-14

ContinueMixedTraining uses the existing mixed objective and data continuation:
1. At the measured handoff, retain the active Mellum weights, optimizer state,
   update counter, RNG and completed-stage history.
2. Bind the admitted R5 Mellum tokenizer twins for the current program-header
   stage and the unstarted whole-trajectory stage. Name the current R3
   program-header data as continued_text_data so seen-document accounting survives.
3. Keep the inherited mixed objective, exact parent projection qualification and
   measured fraction selection. The inherited 0.25 is a preparation placeholder;
   it grants no launch or adoption before the equal-wall comparison.
4. Validate the concrete handoff against the active recipe-plan and checkpoint
   before replacing the frozen runtime.
The recipe is a declarative natural-language training plan. Existing crisp
recipe, input-hash and optimizer restoration verifiers enforce it; no new trainer.

## Preserve JSONL record framing — 2026-10-14

ReadTrajectoryRecords separates records only at the JSONL byte newline (LF),
retaining Unicode line/paragraph separators inside JSON strings. Reuse the
shared jsonl-stream reader for both converter passes, exposing raw lines so
invalid JSON still receives the existing unreadable accounting. The framing
and UTF-8 decoding are exact crisp mechanisms, with no policy or model change.
Node 24 readline treats U+2028/U+2029 as line endings; those are valid JSON
string contents and must not fragment a trajectory row.

The same exact LF reader serves native SFT export and the trajectory rewrite
stage. Both retain their current malformed-JSON behavior and rendering/rewrite
policy; this only prevents valid Unicode-bearing records from being fragmented.

For large rows, decode each source chunk once, scan that new chunk for LF,
accumulate fragments until a complete record, then join once. Do not rescan or
flatten the whole growing record on every input chunk. This preserves exact
JSONL bytes after UTF-8 decoding and CRLF normalization while making framing
work proportional to source length. Verify against the captured 9.5 MB source
row's hash and the existing Unicode/chunk-boundary regressions.

## Preserve Unicode-bearing Python probe and rollout rows — 2026-10-14

ReadJsonlObjects uses the existing protected-task JSONL object reader for
monitor probes, precision probes, paired behavior evaluation and RL rollouts.
Move that exact reader to common/jsonio and re-export it from task_packet so
there is one implementation. Iterate physical LF-framed file lines rather
than str.splitlines, which also splits valid Unicode separators within JSON
strings. Keep object validation and physical-line error locations. These are
exact input verifiers; task scoring, drift policy, rendering and prompts do
not change. Leave actual source-code/text line splitting alone.

## Compare fully scored task runs, including observed failures — 2026-10-14

CompareTaskSuccess keeps the declared same-packet V/T and parent comparisons:
1. Require the same packet hash, ordered unique case indices and exact case count.
2. Require every selected case to be counted in a known terminal outcome; reject
   missing/negative/noninteger counts, infrastructure errors and unresolved cases.
3. Treat observed repetition, budget exhaustion and model failure as scored
   unsuccessful cases, just as the shared in-run task evaluator already does.
   The collector's complete flag means result rows were exported for every case;
   it does not mean all selected cases have been scored. Keep the flag explicit
   but do not require it true when the outcome accounting is exact.
4. Compare explicit Boolean fused-case outcomes; missing V/T outcomes stay pending.
Move the existing accounted verifier into task_packet and reuse it, rather than
introduce a second accounting policy. The acceptance code remains an exact
verifier of this natural-language qualification rule. No prompt or model change.

## Mirror an intact checkpoint while training publishes its successor — 2026-10-14

MirrorCheckpoint follows the existing asynchronous storage policy:
1. Open the source checkpoint and identify that exact file, not an earlier
   observation of its pathname.
2. Copy and hash its bytes. Reject an in-place change to the opened file.
3. Verify and durably publish the copy even if training atomically published a
   successor under the source pathname. Keep the opened file's identity in the
   receipt, so the older mirror is never certified as the successor.
4. Preserve the existing identity check before an archive link replaces a local
   source; never replace a newly published checkpoint with an older mirror.
The crisp helper verifies opened-file identity, bytes, hash and atomic durability.
It changes no model-facing text or training state, and does not relax pruning.
