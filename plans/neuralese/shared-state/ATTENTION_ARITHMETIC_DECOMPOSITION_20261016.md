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

All listed integration work remains OPEN. The local option override was isolated
to its qualification subprocess and has not changed training, serving or any
checkpoint. Crisp code is justified for exact arithmetic dispatch, identity,
checkpoint durability and numerical verifiers; the preceding policy governs
qualification, prioritization and adoption.
