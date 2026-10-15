# CoreWeave supervision — 2026-10-15 18:53 UTC

Exactly three owned sandboxes remain active. Both students train native positional-KV/hybrid-state graphs; an independent strong teacher generates verified trajectories. Pop and local GPU actor generation remain closed. Never manage unrelated sandbox `456d8127-1b99-48fa-9a64-edb1bc15de36`.

| Role | Sandbox | Actual latest execution |
| --- | --- | --- |
| LFM H10080GB | `7ee701ae-8e56-49d8-b97a-594f7eb33fe9` | 18:42:53 UTC step1836, loss1.13516,4388 supervised tokens/s, GPU86%/23025MiB; recipe312461/trainer312478 |
| Mellum Blackwell96GB | `5ffa592c-5f88-4b4f-b3cc-1507ab94a5dc` | 18:42:53 UTC step1559, loss0.115014,4588 supervised tokens/s, GPU100%/63567MiB; recipe146157/trainer146209 |
| Qwen teacher Blackwell96GB | `dadd0462-0175-470d-b1ee-4d52edfe110e` | Server68164/scheduler68332; eight collectors/server concurrency4; 18:47 batch12 finished939/3826queued,904 actual successful executions |

Student health unit1492 terminal0; zero OOM/errors, H100 anon14.4GB and Mellum anon4.2GB of64GiB host limits. Live runtime `e4dd6aabd3b071d3775f81ae71664a4ad52a3fa0`, FP32 pinned QAD LFM/BF16 Mellum, IEEE native arithmetic, whole-graph32768-token packing/chunk1024. Both preserved optimizer/RNG/row credits across the V4 data handoff. Both have completed37/37 currently declared genuine communication visits; no broader communication completion, task lift or final qualification is claimed.

Task cadence is1000updates with exact same-invocation/action/feedback repetition limit3; held/checkpoints250updates, hourly checkpoint fallback, held rotation524288tokens and mandatory full final held. Absolute modulo means next task evaluation is step2000 for both. Data and task cadence can change at useful safe handoffs.

## Current exact state and input

Privately durable independent full-object GET/SHA-verified parents:

- LFM1750:2027511243bytes SHA `6e97d601e32f2ad1316601e277c97eeff1046fd3945de3e2f2c6f203ae166df7`.
- Mellum1500:47761947901bytes SHA `fe66c38b383ff191f107ca6a13db2e3aadca2aebccf7d60761300ac1a043f404`.

Periodic exporter services remain active on the same sandbox IDs/output paths; their private `coreweave-native-{lfm,mellum}-periodic-export-state-20261015.json` files are authoritative as newer checkpoints become durable. Current rolling optimizer updates beyond these parents are not yet independently privately exported.

Current train sources185954 LFM/185955 Mellum rows: FULL53223 → Qwen624(priority1) → REST ordinary572 → corrected pair37 → QwenV3 2451(priority1) → REST ordinary125643/125644 → QwenV4 3404(priority1). Previous source order and actual per-row exposure credits remain exact. Availability is not trained exposure.

Source manifests `/work/native-production/data/native-state-qwen-v4-append-20261015-v1/{lfm25,mellum21}/train-sources.json`:

- LFM SHA `dbb154941dcc28654f52cac0d019503b65a931352e0baad929da61d08e0f0434`.
- Mellum SHA `0f2564469a0914bc2377bdf10e6d8fd2076a7696bddcabe7e76e5e289a1c1a0c`.

Current flat recipes `/work/native-production/control/{lfm,mellum}-qwen-v4-cached-handoff-20261015/declared-cadence-recipe.json`, current resolved identities equal their flat-file hashes:

- LFM2933bytes SHA `1bc8dd7491539343e514ddc486bf380bdf550fcbdda03f362feb4a06de06d0fe`.
- Mellum2956bytes SHA `235e7c3a1300b9a3a0452ea4621fe50f054f55851669f779ea6e5c50fecef949`.

Original lineage identities are preserved separately in recipe plans. Outputs remain `/work/native-production/{lfm25,mellum21}/native_full_sft`. Final `native-training-receipt.json` and readiness do not exist yet; current runtime does not emit the new foundation/mandatory-SFT contract.

## Interrupted evaluations and next useful handoff

The owner authorized preserving and stopping pathological synchronous evaluations to resume useful V4 training. Exact eval process trees were stopped and drained without forced kills; exact privately verified pre-eval parents were reused. LFM1750 partial:0successes,2budget-exhausted,14unresolved. Mellum1500 partial:3successes,1budget-exhausted,1invalid,1semantic mismatch,10unresolved. Both complete=false/eligible=false, never a completed0/16 or3/16 score. No new-native base reference has been measured.

Complete partial archives and explicit supervisor cancellation receipts are SHA-verified on DGX under `/home/werg/data/dgxops/native-interrupted-evaluations-20261015/`; retained solely for assigned held-only runtime/model diagnosis, never training admission. Mellum repeated an invalid Promise.all expression with identical feedback55+times. LFM nested children generated8192-token no-call responses. Request walltimes include queue/prefill/decode, not isolated decode measurements.

Next admitted data is corrected REST pair V3:26312train pairs including exact37prefix,9896held; append only26275new train rows after the ENTIRE current source prefix. Strict two-model/protection proof is admitted `c4502012a`. Recovery512 historical successful ordinary rows/model are admitted `e006813d6` and already SHA-verified on both sandboxes. Pair staging and frozen `c4502012a` runtime staging are active while GPUs train. Prewarm exact new-code indexes, verify prefix/held separation/plan, then introduce the explicit communication-foundation cohort while retaining mandatory ordinary deficits. Do not wait for the later18010+605 repair cohort or call all ordinary exposure complete.

## Teacher and CPU data work

Teacher `RadixArk/Qwen3.8-Flash-Next-NVFP4@7b719225242aacd3dbd3f9407468c2ee9a9d2594`, Penny Royal3.0.1 source `fa9c2673131621cafb9a527e61d97c25eac130a9`, uses the actual natlang plaintext tool harness. It is independent, never student/on-policy. Current1913-case packet SHA `e29b52fe990707e580df947dc152cedbebc98a23daa9f7aeec9e5043ec02433b` includes384 code/recovery cases plus1529 protected CLINC factual-choice programs. New queue generation preserves each immutable batch's actual source packet and protection witnesses. Exact duplicates are removed before admission.

Batch12 at18:47:904successes/939completed,13semantic,21invalid,1budget. Some exact label mistakes are classified invalid_action because the expected decisions.json file differs despite valid executed actions; inspect underlying ledgers before attributing parser failure. Wrong chart-depth final expressions and CLINC pet/meeting/directions labels have exact CPU-only diagnosis artifacts under `dgxops/qwen-broad-1913-failure-diagnosis-20261015/`. Rejected rows remain assigned repair/oracle investigations, not admitted positives. Independently correct rows can be retained through explicit verification. Current V4 admitted3404 unique graphs/model is LIVE, not merely exported.

Big2 byte-range conversion is active,2nice19 workers on cores5,6; Dolci uses7,8. At18:50 Big2 committed800chains/18352source edges,8chunks,17989ordinary rows/model,0render omissions; source185636995853bytes remains on29TB-free H100 root. Four Big2 workers on3,4,5,6 are approved after a durable frontier if GPU feed/headroom remain healthy.

First100-chain chunk is independently privately durable and restored to DGX:9006326byte archive SHA `f37ec195b626e75c7c935261ad26d017cbbd90a086f56848b7e22bd15eb32cab`. It contains2115train/154held ordinary rows/model,2153train/150held pairs,2303source edge IDs, exact100chainIDs, all protection/source receipts and per-file hashes. Actual strict admission audit is pending with data agent; no complete-Big2 claim. Private pack/export receipts `coreweave-big2-first-shard-pack-20261015.json` and `coreweave-big2-first100-private-export-20261015.json`. Dolci latest252/644native shards at18:42;643106closed source records,22paged-evidence conversions remain explicit.

## Monitoring and provider bridge

Every local probe runs through the memory ledger using `/home/werg/.local/share/natlang/cw-venv/bin/python`. Helpers in `/home/werg/data/dgxops`:

- `coreweave-native-health-20261015.py`: actual active control, current output progress/metrics, GPU/cgroup/errors.
- `coreweave-native-live-binding-20261015.py`: actual recipe/trainer PID and receipt recipe identity.
- `coreweave-native-v4-release-progress-20261015.py`: release/stop/cache-hit evidence.
- `coreweave-penny-collection-health-20261015.py`: exact oracle-classified teacher executions/output paths.
- `coreweave-big2-native-frontier-20261015.py`; `coreweave-remote-cpu-job-20261014.py poll coreweave-dolci-frontier-spec-20261015.json`.

Coord reader is `dgx-coreweave-sandbox`; inbox ACK18:35, next no later19:25UTC. Maintain monitoring at least every50minutes, normally10minutes during changes. Explicit owned PIDs only for signals. PID1 is an independent allocation holder; no allocation stop is currently planned.

Published provider implementation `scripts/coreweave/provider_{bridge,worker,dependencies}.py` uses exact request/result/stop bindings, owned PID/boot/start/argv/nonce recovery, signed private cloud-to-cloud transfers and complete consumed dependency closure.21 focused tests plus quick check passed; broad containing suite is not yet claimed green. Real48-byte SDK ferry passed earlier. Actual persistent two-port CPU fixture1490 staged14payloads through real private storage, then failed BEFORE worker launch in target asset preflight (remoteexit7); precise SDK diagnostic1502 is active. No orphan worker was launched, no production RL is released, and actual round/refresh/TERM/restart qualification remains unfinished. Production coordinator will reside in Mellum `/work/native-production/shared/bridge-ipc`; all account credentials remain on DGX.
