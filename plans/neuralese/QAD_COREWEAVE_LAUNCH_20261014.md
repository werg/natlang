# QAD LFM CoreWeave production handoff (2026-10-14)

The clean R8 QAD warmup qualified after 410 updates. Continue its full state in
`neuralese-direct-lfm-qad-r8-ph-coreweave-20261014-v1`, beginning with the
protected R8 program-header phase. The existing optimizer, RNG, identity and
monitor reference are part of the handoff; do not restart from the base model.

## Exact state

All five files below are regular files at
`/mnt/external/natlang-development-data/pop-handoffs/qad-r8-clean-warmup-20261014-v1/`
on DGX and were SHA-256 verified after the Pop-to-DGX transfer:

| File | SHA-256 |
| --- | --- |
| `checkpoint.pt` (full optimizer/RNG state) | `13ebca2a4c2f2f087ce2cfc4a635114b4aa2b91a017714958bed71cd33551761` |
| `identity-check.json` | `e1059804617edeb679d1ffa62ea8778057c59df05c73985aa09ab1fd2bdc56b1` |
| `monitor-reference.pt` | `2c746ac3db20eb273200950748dc0649a4a5e879f05a250d5b499e195cc6ce18` |
| `clean-base-untrained-heads.pt` | `19bfa7a304d511fe4813e2625fb2855095bd1a104af4b7d6b2ed77368226483c` |
| `direct-certificate.json` (`qualified: true`) | `80cd018d19440343cfc8ca10df3e87bb124884435bd28319405ce450db85e81f` |

The recipe pins the independently audited LFM R8 PH compressed input SHA
`60c69c269ac7cfc6ee26cbc5d5c78c2d72e98d14ed8370027fe9e800b5c3edf8`
from published immutable corpus `natlang-target-union-r8-ph-lfm-prefix-20261014-v1`.
Its proof SHA is `fba843b5be43bd449d2099b44fc6c8443755767036ed08d4d5cc8515e96766c6`.
The cloud helper can transfer the file directly from DGX. The inherited R3 whole-trajectory stage is `admitted: false`
only to preserve the shared nested-family/QAT stage map; bundle and launch
`direct_sft` only. The R8 whole-trajectory phase needs separate admission.

## Runnable evaluation and runtime inputs

Bundle the recipe's named `direct_sft` inputs, the PH corpus manifest's two
files (including `prefix-proof.json`), the qualified
`direct-certificate.json` at the DGX handoff path, and these additional files at
their logical paths. They are regular files in the DGX canonical checkout:

| Logical path | SHA-256 |
| --- | --- |
| `runs/luna-neuralese-direct-clean-base-monitor-20261011-v2/monitor/probes.jsonl` | `0bce5275646cd95ae36374f41769602995c8587c871275653b26c1cf260c3d52` |
| `runs/natlang-held-task-packet-v3-20261012/cases.jsonl` | `535269866aeecc107e3cc8b165790a5b200cdb104884d4994958eee17eb6be95` |

Use the pinned `LiquidAI/LFM2.5-350M` revision
`9e6c6ccf47cd318696e137d381a7ded8fe4df09f` for model configuration and
tokenizer. Bundle code from the pushed recipe commit, build `ts-host` in the
sandbox (`NATLANG_CW_BUILD_TSHOST=1`), and record the actual NGC 26.09 image
in the launch receipt. Pop's monitor path is a symlink into `/srv/storage`;
bundle DGX's regular file rather than the Pop symlink.

CoreWeave supplied an H100 with 80 GB, not the RTX PRO 6000 described in the
earlier hardware plan. Keep the shared `window_execution: auto` and activation
memory planner initially. Its uncalibrated no-checkpoint estimate uses 2×
headroom; record the first representative windows' checkpointing choice,
predicted/actual CUDA peak, throughput and evaluation wall time. If it
checkpoints despite safe measured headroom, adjust the shared planner/profile
and continue the same full state. No LFM-specific trainer fork is needed.

## Corpus admission still in flight

The filtered R7 source feeding R8 includes the approved
`qad-student-root-recovery-s52052` and
`luna-service-result-recovery-s52054` renders. The R8 build also adds the
external static first wave. A comparison against the R7 member receipt found
three more training-admitted rendered sources that R8 has not included:
`natlang-native-turns-clef-pools-20261014-v1-render` and both LFM/Mellum
`luna-held-shapes-closed-backlog-v5` renders. DGX has the files. Bring them
through global held/source protection in a successor union or explicit data
continuation. Fresh Luna s52059+ and the newly replayed QAD repair handoffs
still need quality admission; do not silently bind them as positives. The
current production PH phase can start from the protected snapshot, then use
full-state data continuation when these additions qualify.
