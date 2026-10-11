# Best225 heads learning-rate continuation v3 (held launch evidence)

The byte-pinned shared recipe declares only `heads_lr: 3e-4 -> 3e-5` and `curriculum_change: [heads_lr]`. It continues the exact full-state step-225 checkpoint with the shared recipe/trainer, retaining data, seed, objectives, other rates, 687-step target and 200-step checkpoint/evaluation cadence. `parity: diagnostic` makes the shared line-parity gate explicitly diagnostic.

This packet records the root-approved v3 recipe and stable actual stage plan/recipe launch metadata. At this snapshot the model process had started; the mutable container log and launcher receipt are intentionally omitted and will be recorded in a separate terminal addendum.

Earlier attempts are preserved: v1 failed before model launch because its recipe omitted the required parity declaration; v2 passed that declaration but failed before container start because the wrapper runtime destination already existed. V3 uses a fresh output/runtime directory. These failures are setup evidence, not model outcomes.

No weights or runtime source tree are included. This is a diagnostic rate experiment only, not a qualified line default or quality/runtime/corpus admission.
