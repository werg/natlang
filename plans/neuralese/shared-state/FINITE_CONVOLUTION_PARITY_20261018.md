# Shared finite-convolution arithmetic

The flat shared-state graph uses `_finite_convolution` for every LFM finite
history. The invocation reference and incremental consumer must use that same
reduction. Selecting IEEE matrix arithmetic must not select a separate cuDNN
convolution. History gathering, positional aliases, stopped gradients, padding,
bias, activation casts and cache retention keep their existing mechanisms.

The correction reuses the existing finite-convolution helper in both cached
paths. It changes neither the flat training computation nor the non-IEEE
cached computation. Ordinary backbone execution remains separate. The existing
nested-export test exercises the graph, invocation reference and incremental
splice, including imported aliases and gradients, and rejects a separate
`conv1d` path for the default IEEE profile.

## Actual trained-parent diagnosis

The read-only local diagnostics used LFM step9478, full checkpoint SHA-256
`f07a57e3d98224360b8a509f8f410be3de2fd37830e176c989164d2e83ebcb89`,
without optimizer updates or cloud-runtime changes. No tolerance was relaxed.

- Existing fixed-tile IEEE FFN products plus existing fused row norms, retaining
  the parent's native BF16x6 projections: exact forward, incremental and consumer
  CE agreement; 145/151 complete derivative comparisons pass.
- Changing native projections to IEEE while retaining the separate cached
  convolution: forward difference6.56e-6 and107 failing derivative comparisons.
- Reusing the flat finite convolution in both IEEE cached paths: forward,
  incremental and consumer CE differences become zero; ten derivative comparisons
  still fail. Full Python/Torch/CUDA RNG preservation is checked in this run.

Evidence is in `/home/werg/data/dgxops/lfm9478-invariant-ffn-fusednorm-root-20261018.json`,
`lfm9478-ieee-invariant-ffn-fusednorm-root-headroom-20261018.json`, and
`lfm9478-ieee-finiteconv-invariant-ffn-fusednorm-root-20261018.json` in that directory.
The first report's RNG flag checks only the CPU Torch stream and must not be
used as evidence for all streams. An intervening IEEE diagnostic failed GPU
allocation before loading the model; the headroom-guarded retry above is the
completed result.

These diagnostics localize distinct arithmetic discrepancies. They do not grant
full communication-gradient qualification, production cutover or VAE readiness.
The diagnostic FFN/norm substitutions remain local; they are not installed by
this correction. Remaining derivative discrepancies require further localization.
