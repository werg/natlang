# Explicit native norm execution

The training policy is to preserve a checkpoint's declared arithmetic while
adopting shared metadata and parser improvements. An absent execution declaration
means the original source norm dispatch. Selecting stable native norms is an
explicit, persisted arithmetic transition, separate from FP32 gain storage.

1. Restore parameters, optimizer, RNG, exposure and the saved execution profile.
2. Use source-v1 for an absent legacy field; do not introduce a field on restore.
3. When requested, select stable-v1 and record the transition without changing
   tensor values or parameter identities.
4. Apply the selected dispatch in graph, cached inference and checkpoint
   recomputation. Preserve existing fused BF16/FP16 norm arithmetic.
5. Bind the profile in actor identities, composite exports and recipe parameters.

Crisp helpers validate the two exact profile names, set dispatch attributes and
verify persisted identities. Checkpoint durability and recipe verification remain
their existing mechanisms. This changes no model-facing text.

Qualification compares both ports' actual outputs and adjoints with pinned
source and stable references, plus checkpoint/composite/raw identity round trips.
No active cloud runtime is changed by this implementation. Source-v1 retains
known source dispatch behavior; stable-v1 retains the existing native checkpoint
norm repair. Storage precision and attention modes remain independent.

All actual active 4ea/608 production parents lack this field and use source
dispatch. Earlier unprofiled HEAD FP32 diagnostic checkpoints may have used the
stable repair: they need an explicit stable-v1 metadata upgrade before use, and
must not be described as having previously executed source-v1. They are not
active production parents. Fresh native loaders record stable-v1 explicitly;
checkpoint restoration preserves the absence of a legacy declaration.

A transition that reaches CUDA Maple's compiled fallback uses the existing
cached-reference forward/adjoint qualification and declared tolerances. LFM,
CPU and ordinary fused BF16/FP16 norms do not change arithmetic and do not run
an additional qualification or MAX allocation merely for this metadata field.

Qualification on 2026-10-17:

- Pinned source reference: 4ea5f59a371fee5d75302f3526ee5e2544531bee.
- Pinned stable reference: 0196a9c534b35835e2c7357d11162216abbb0cf2.
- Both ports and profiles match their reference graph and incremental outputs
  bitwise on CPU and CUDA, including 84 LFM and 80 Mellum named graph/incremental
  adjoints per profile. Source actor identity is unchanged; stable identity
  explicitly includes the new declaration.
- 119 CPU tests passed (35 CUDA skips), plus 18 existing stable CUDA regressions
  and four dispatch-transition cases. Restore, raw metadata, composites, shared
  participant checkpoints and real no-update/update continuation are covered.
- Current HEAD curriculum task-source tests pass without a CPU preload fork.
- Exact local reference receipts are
  `/home/werg/data/dgxops/native-norm-execution-reference-{cpu,cuda}-extended-20261017.json`.

Whole HEAD adoption still uses ordinary bound source-index and runtime capacity
continuation checks; this profile is not a capacity certificate. It removes the
unconditional arithmetic reinterpretation requiring a frozen Python package.
Attention remains explicitly IEEE unless selected otherwise, and existing
optimizer, FP8/expert and linear arithmetic kernels are unchanged. Marker
feedback retains its independent executing-model measurement requirement.
