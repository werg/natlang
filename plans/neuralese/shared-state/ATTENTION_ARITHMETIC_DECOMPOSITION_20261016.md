# Accelerate the measured attention bottleneck without losing the native contract

Policy: spend optimization work on the measured dominant cost, retain strict
forward/adjoint and state-continuation checks, and deploy a useful qualified
improvement promptly. Both students share the mechanism. This work does not hold
up the already qualified complete-runtime/raw-RL handoff.

The existing H100 normal-graph trace has SHA
`c40ff9559ff17f02f84b7ff9b7bdfbe4b845ae068c49db3f5475af2d328a9e9b`.
Across its old/new attention-index regions, attention backward sums to49.962s
and forward to15.875s, roughly85% of the77.7s region duration. These sums identify
the dominant kernels; inclusive CPU `AccumulateGrad`/scalar waits do not prove a
CPU bottleneck. The trace summary is
`/home/werg/data/dgxops/lfm-normal-profile-summary-root-20261016.json`.

An isolated local GB10 qualification compared IEEE and TF32x3 attention on two
independent uneven invocations (257/259 tokens), grouped heads8:2 and widths64/128,
including a high-amplitude width128 case. Both modes retained BITWISE graph versus
cached outputs and all Q/K/V adjoints. TF32x3 maximum errors against an FP64 dense
oracle were8.67e-7/3.70e-6 forward and3.68e-9/1.76e-8 adjoints in those two cases;
IEEE had errors of the same order. This is evidence for further qualification,
not a full-model/long-graph correctness or H100 throughput certificate.
`num_stages=1` is required for this candidate on GB10: its width128 default
configuration requested114688B shared memory against a101376B hardware limit.
This was a kernel configuration failure, not a model or total-memory failure.
The complete terminal report is
`/home/werg/data/dgxops/attention-tf32x3-qualification-root-20261016.json`.
Concurrent tiny timings are not production speedup evidence.

1. Add one explicit per-model attention arithmetic setting, initially IEEE by
   default. Its setter changes no torch process-global flags, parameter values,
   optimizer state, gradients, source rows or token transforms.
2. Thread the setting through flattened graph, explicit cached reference and
   incremental serving attention. Separate compiled contracts by arithmetic
   setting to avoid specialization mixing. TF32x3 applies only to FP32 attention;
   existing BF16/carrier and higher-order reference paths keep their contracts.
3. Bind nondefault arithmetic into native model/channel/checkpoint and execution
   identities. Old absent fields mean IEEE and preserve legacy identities. Load
   archived actors with their actual saved setting. Reject mismatches rather than
   relabeling native state or historical behavior.
4. Expose the setting through the existing shared trainer and recipe parameters.
   An explicit safe-boundary transition restores exact model/optimizer/RNG and
   per-row exposure, records old/new arithmetic and qualification, and uses the
   existing numerical transition checks. It must not reset lineage/history.
5. Validate real model forward/consumer losses and all relevant adjoints against
   exact cached execution and the existing tolerances, on both backbones,
   checkpoint recomputation and source/import layouts. Exercise checkpoint and
   inference restore plus mixed-model isolation. Keep source/pending/capacity
   identities explicit; do not inherit a numerical capacity certificate silently.
6. Measure an actual normal trained-parent H100 graph, then long pending geometry
   if the normal path qualifies. Adopt through an ordinary saved continuation
   when it improves throughput. Propagate applicable gains to Mellum; its BF16
   attention is a declared architecture/runtime distinction, not a separate
   trainer. Diagnose failures rather than broadening numerical tolerances.

The shared implementation now threads `native_attention_precision` through
model construction, native identity, channel/full checkpoint restore, flattened
graph, explicit cached reference, incremental serving, trainer continuations and
SFT/preference/RL recipe tuning. Absent fields preserve the previous IEEE mode
and identity. BF16 calls retain the same kernel options AND compiled contract;
only FP32 TF32x3 selects the explicit single-stage pipeline. There is no process
global mode or production override. Transitions preserve optimizer/RNG/exposure
and record the unchanged-tolerance cached-reference qualification in history.

Local integration passed38 cases, including real interrupted training followed
by a TF32x3 update with exact optimizer/RNG restoration and prior row credits,
legacy/default and explicit checkpoint restore, actual server loading,
both-backbone graph/cached and incremental execution, and parameter adjoints.
Two additional CUDA tests verify the actual selected TF32x3 kernel options,
FP64 forward/QKV adjoints and bitwise graph/cached QKV gradients at widths64/128.
BF16 mode-on versus mode-off full-model outputs and all parameter adjoints are
bitwise identical. Strict cache comparison covers FP32 interface adjoints;
BF16 LFM full-backbone cached adjoints show an existing diagnostic mismatch and
are not being qualified for production by this work. Production LFM retains
FP32 backbone masters and BF16x6 projections. No numerical tolerance was raised.

Independent implementation review, actual trained-parent H100 normal/long
qualification and production adoption remain OPEN. Crisp code is justified for
exact arithmetic dispatch, identity, checkpoint durability and numerical
verifiers; the preceding policy governs prioritization and adoption.

Independent closure review found and repairs two nondefault-only omissions:
raw-policy metadata proxies must bind the saved attention mode on both model and
port, and immutable composite update manifests must retain that field. The
existing parent/config verifier rejects a mismatched declaration; it does not
reinterpret an updated actor. Async evaluation also checks the saved attention
mode explicitly. These exact identity/durability paths share the original model
loader, optimizer and artifact mechanisms. Default IEEE serialized forms remain
unchanged. Source review confirms shared semantic checkpoints retain complete
model configs and ordinary RL/preference execution uses the shared trainer.

The full-parent numerical observer is
`scripts/neuralese_data/qualify_native_attention_arithmetic.py`. It is an exact
verifier, not an adoption policy. It uses the existing capacity executor and
its unarmed optimizer/RNG/backend/parameter guards, compares the actual loss,
hidden outputs and every observed named parameter adjoint under explicitly
declared tolerances, and records all unobserved trainables. Reference adjoints
spill one at a time to temporary storage; timing runs omit serialization and
comparison. Both numerical passes warm their kernels before one whole-graph
timing per mode. That timing is preliminary, not a stable throughput estimate.
The CLI binds the actual imported package, saved parent, source row and original
capacity identity. This does not grant a new capacity certificate or task
qualification. Run it on actual normal and long source geometry before cloud
arithmetic adoption; graph/cached qualification remains separately required.
Every observed and timed pass with an enabled FP8 backend requires actual
positive projection-call and row counters, using the same verifier as joint
CUDA qualification. A finite backward with an inactive backend is rejected.
The report includes retained hidden-output bytes; long-geometry host admission
must include both output tensors as well as the loaded parent and active graph.
