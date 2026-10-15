# CoreWeave supervision handoff, 2026-10-15 12:17 UTC

Both existing runs are diagnostics while the replacement native architecture is integrated. No third GPU, new lineage or allocation change is authorized by this handoff. Continue periodic checks; current source inputs and optimizer/RNG must survive any later handoff.

## Actual allocations and processes

| Role | Sandbox | Hardware | Existing process IDs |
|---|---|---|---|
| LFM | `7ee701ae-8e56-49d8-b97a-594f7eb33fe9` | H10080GB,64GiB host,~30TB root | bootstrap200684, recipe203145, trainer203166 |
| Mellum | `5ffa592c-5f88-4b4f-b3cc-1507ab94a5dc` | RTX PRO6000 Blackwell96GB,64GiB host,~7TB root | bootstrap63126, recipe65847, trainer65896 |

LFM has not yet resumed update28001: the correct window/2 preparation is CPU-active after exact28000 restoration. The last metadata check showed GPU0%/1991MiB, no errors. An explicit process probe at12:06 found StateR, one CPU busy,2.79GB anonymous RSS,4.34GB backing bytes written and a fresh mapped shard6seconds earlier. Do not confuse the old28000 metrics row with a resumed update. The source isf5e6e1726b; live recipe is `neuralese-direct-lfm-qad-r8-ph-coreweave-operational-20261015-v6.json`.

Mellum reached3677 at12:15:36, finite loss8.82e-5,7.46seconds/25599forward tokens, GPU92%, no errors, peakallocated56.26GiB. Task3600 finished8/16 (previous5, inheritedbase13):3invalid,2repetition,3semantic, zero infrastructure/unresolved,715.56seconds. It remains unqualified against the base reference. Mixed direct/E2E updates are real; no task-lift certificate is claimed. Live sourcec59ba570e1, recipe `neuralese-mixed-mellum-r11-full-coreweave-20261015-v1.json`.

## Durable full-state parents

- LFM28000:1,149,841,883bytes, SHA `737f37ee855c68830a003e2ac6dc9591dd0ac2319e72f085bdb61f2e60dc8fbe`.
- Mellum3600:47,078,033,383bytes, SHA `e97eb68e611a1e4acd03e937306c7f293b198134f094d0b0abb72f3b132bf00c`, upload plus independent full GET verified12:09:54. This includes added P/TCP optimizer state.

Private object identities and transfer receipts are in `/home/werg/data/dgxops/coreweave-production-{lfm,mellum}-periodic-export-state-20261014.json`. Active systemd services are `cw-lfm-periodic-private-checkpoint-export.service` and `cw-mellum-periodic-private-checkpoint-export.service`. Exporters remove their temporary committed-inode pin after successful durability; recovery must verify the rolling checkpoint or create its own immutable pin, rather than assume an exporter pin survives.

## Read-only supervision commands

From `/home/werg/natlang`, use unique ledger unit names:

```bash
python3 scripts/memory_ledger.py run --unit cw-health-UNIQUE --budget-gb 1 --priority medium -- /home/werg/.local/share/natlang/cw-venv/bin/python /home/werg/data/dgxops/coreweave-production-progress-20261014.py
```

Wait for that unit to terminate and inspect its journal. The helper reads bounded1MiB log tails and writes `/home/werg/data/dgxops/coreweave-production-{lfm,mellum}-progress-20261014.json`. It reports metrics, actual GPU use, task results and error tails; old metrics can persist during startup. Other existing bounded helpers in the same directory:

- `coreweave-mellum-r11-watch-20261015.py`: Mellum process IDs, rolling checkpoint stat, GPU and8KiB log tail.
- `coreweave-lfm-terminal-watch-20261015.py`: LFM equivalent,32KiB log tail.
- `coreweave-lfm-staging-inspect-20261015.py`: explicit trainer203166 process CPU/I/O, status and newest backing-file metadata; appends JSONL snapshots.
- `coreweave-task-eval-progress-20261015.py mellum 3600`: case terminal progress only; never infer the final score from journal statuses.
- `coreweave-cloud-data-frontier-20261015.py`: H100 anonymous memory/OOM counters and native-shard counts.
- `coreweave-remote-cpu-job-20261014.py poll SPEC.json`: existing CPU-job phase, terminal receipt and bounded logs.

Run coordination with independent reader `python3 scripts/coord.py --as dgx-coreweave-sandbox inbox --ack`; never acknowledge the supervisor cursor. Do not run mutating recovery helpers merely to inspect health.

## CPU jobs and cache correction

Correct PH prewarm: `r11-ph-prewarm-lfm-50e5a079`, runner205061/worker205065. Correct WT prewarm: `r11-wt-prewarm-lfm-679497cf`, runner208847, waiting behind the same CPU preparation lock. Their local specs are `/home/werg/data/dgxops/coreweave-r11-{ph,wt}-prewarm-lfm-spec-20261015.json`.

The generic helper previously hardcoded held target_tokens1024 for LFM, whose actual recipe uses256. Those older completed caches are not compatible. The helper now derives geometry from the resolved recipe, with geometry-hashed job names. Only the wrong WT CPU worker180595 was terminated; no trainer was stopped. Future CPU runtime reuse verifies identical archive member bytes before reuse instead of overwriting a live frozen runtime. Current correct28000 staging continues independently.

The old PH handoff controller unit1040 is stopped. Do not restart it as-is: its obsolete trainer138715, old prewarm job names and v5 inheritance must be updated to actual203166/currentv6/correct256 caches before any future use. There is no active automatic PH cut waiting behind the scenes.

Dolci/Cognitive uses the existing native driver102717/task102721, affinity7,8/nice19/zeroCUDA, runtime `/work/cpu-runtime/dolci-cognitive-7c300c97e647`; last health frontier106/644 shards, later count available in the frontier receipt. Existing big2 E2E assembler176857 (parent176853), affinity6/nice19/zeroCUDA, continues under `/work/data-preparation/e2e-incremental/work-big2`; both huge archives were restored with per-member SHA verification. REST27 lazy indexes are terminal successful on both boxes. Search/OWL source/native archives are privately durable; data helper owns DGX restoration/registration.

Three prior Dolci `wrong_return` cases from shards00029/00031/00093 were replayed against unchanged saved IR/oracles in the same frozen runtime: all three freshly admitted with exact accepted results. Full sample and replay evidence: `coreweave-dolci-wrong-return-{samples,replay}-20261015.json` in dgxops. Original per-shard positions147,110,56 differ from the fresh three-case enumeration0,1,2, so process history/index/runtime-state causes remain unresolved; no causal attribution or broad rerender is claimed. Failed original rows were not retained by the existing replay CLI, only omission IDs/reasons. Keep these omissions assigned for recovery rather than rejecting their source answers. A current aggregate omission/index frontier is produced by `coreweave-dolci-wrong-return-frontier-20261015.py`.

## New native H100 proof and adoption boundary

The isolated published2ff59824 FP32 native CLI passed without an external precision wrapper: forward1.907e-6, consumerCE4.768e-6, channelgradient1.065e-6, incremental6.557e-7; unchanged tolerances1e-4/1e-5; channel optimizer update and exact restore passed;50.18seconds/1.598GB peak. Receipt `/home/werg/data/dgxops/coreweave-native-lfm-qualification-2ff59824-20261015.json`, launch receipt with exact archive/commit beside it, remote runtime `/work/native-runtime/2ff598246378`.

Earlier133f2ce0 NGC-default failure and otherwise-identical highest-precision control remain separate receipts. Defaults were matmul_precision=high and matmul/cuDNN TF32 enabled; disabling both made the original check pass. Integrated2ff now passes under its shared factory contract. This qualifies the interface/channel mechanism only, not full backbone training, BF16/full Mellum, task ability or the destination recipe.

Do not overlay canonical cloud source while the old Mellum source can import it: safe frozen Python import plumbing was published later than its active source. Use isolated Git runtimes for diagnostics. Any eventual change must stop the exact ordinary trainer PID at a checkpoint-safe handoff, preserve full optimizer/RNG/state, and then adopt the prepared runtime. PID1 is an independent holder and must not be terminated or treated as a trainer signal forwarder.
