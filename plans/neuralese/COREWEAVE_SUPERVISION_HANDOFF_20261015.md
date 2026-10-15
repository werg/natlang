# CoreWeave supervision — 2026-10-15 17:11 UTC

Exactly three owned sandboxes are active. Both students train the replacement native positional-KV/hybrid-state architecture. The strong teacher is independent of either student. Pop and local GPU actor generation remain closed. Never manage unrelated sandbox `456d8127-1b99-48fa-9a64-edb1bc15de36`.

| Role | Sandbox | Current execution |
| --- | --- | --- |
| LFM H10080GB | `7ee701ae-8e56-49d8-b97a-594f7eb33fe9` | Native SFT step1395, finite loss0.54339,2122.5 supervised tokens/s on an intact31085-token graph; GPU96%/27249MiB |
| Mellum Blackwell96GB | `5ffa592c-5f88-4b4f-b3cc-1507ab94a5dc` | Native SFT step1311, finite loss0.23514,2679.7 supervised tokens/s on an intact62593-token graph; GPU75%/82349MiB |
| Qwen teacher Blackwell96GB | `dadd0462-0175-470d-b1ee-4d52edfe110e` | Penny Royal server68164/scheduler68332; eight collectors, server concurrency4; all384 protected code/recovery cases now queued |

Both students have actually completed37/37 required genuine producer/consumer communication graph visits. Ordinary training continues; this does not establish communication lift or final qualification. Current source `0c9eab0576435521aeba559086c3027cf2987f06` uses FP32 QAD LFM/BF16 Mellum,8192-token packing, intact graphs, hourly checkpoints, rotating held budget524288tokens and task cadence5000. All current OOM counters remain zero. Held step1000 subsets: LFM CE0.845135 on512839forward tokens; Mellum CE0.245855 on519694tokens. Full held and task qualification remain outstanding.

Current exact native private full-state parents are LFM553,2003768331bytes SHA `3b3f6730b0b88c18bfb07dda408066fb895bd36df15c1a308b835effbe764e71`, and Mellum562,47738204413bytes SHA `6771976dafc5c2a4d9bbe088ee5c1a30ed2fd87339b4dbd96451a31256f13e22`; both passed independent complete private readback. Exporter states in dgxops are authoritative as they advance. Retired diagnostic parents34713/4665 remain only for assigned regression diagnosis; no old diagnostic trainer/exporter remains active.

## Immediate data handoff

Live input has54456 rows/model: FULL53223 → admitted Qwen624(priority1) → REST ordinary572 → corrected pair37. Both admitted QwenV3(2451 additional unique graphs/model) and full REST ordinary tails(125643 LFM/125644 Mellum) are now SHA-verified on both boxes. New manifests preserve that entire old prefix, append QwenV3(priority1), then the REST tail. New totals182550/182551; downloaded or staged data is not yet trained exposure.

Prepared manifest hashes are LFM `aedef9e2581b587aec2c764d620417e15a2bf669cd5d2f6546f42943dd902f7c`, Mellum `9802435d045978d1399fc5beaa3931c9bff26ee0952601e3a7861d4a2c4dabe7`, under `/work/native-production/data/native-state-rest-qwen-v3-append-20261015-v1/<model>/train-sources.json`. Recipes `native-state-production-{lfm,mellum}-rest-qwen-v3-append-20261015-v1.json` retain optimizer/RNG and row credits, change packing to32768tokens and loss chunks1024, and preserve mandatory full final held evaluation. Arithmetic remains IEEE; no diagnostic precision mode is enabled.

Published code `e4dd6aabd3b071d3775f81ae71664a4ad52a3fa0` includes readiness identity repair and the native-only chunk handoff guard. CPU preparation/prewarming is in progress while existing GPU updates continue. Automatic release controllers wait for successful exact index/prefix/held-group/plan checks, then signal the explicit current trainer, drain the committed full checkpoint and resume the same output with exact state. No manual release barrier remains. Obsolete d8-only CPU preparation/watchers were stopped without touching trainers. Current preparation receipts: `coreweave-native-rest-v3-preparation-20261015.json`; remote controls `/work/native-production/control/{lfm,mellum}-rest-v3-preparation-e4dd6aabd3b0`.

## Teacher and data work

Teacher `RadixArk/Qwen3.8-Flash-Next-NVFP4@7b719225242aacd3dbd3f9407468c2ee9a9d2594`, Penny Royal3.0.1 source `fa9c2673131621cafb9a527e61d97c25eac130a9`, serves actual plaintext natlang tool execution with full trajectories and oracle verification. No student/on-policy claim is made. Current384-case packet SHA `d2fd44e24ef36d7620b34448133d611fd6aea63dbfbcdb0dcf362590a4050860` includes all eligible admitted source siblings,192 protected train groups. At17:03 batch9 had516 successful executions/518completed from768queued. Previous code batches3–8 produced764/768 successes. These are execution counts, not unique admitted examples. C8 collectors measured261.3 decode tokens/s over60seconds versus136.0 with C4, with server/model settings unchanged.

V4 positive export is active, preserving each batch's exact selected packet, selection/protection witness and immutable teacher-source receipt in `batch-source-bindings.json` (`natlang.teacher-batch-source-bindings/1`). Historical extraction batch2's six successes remain assigned to oracle/provenance repair and are not silently admitted. Existing V3 was independently deduplicated/admitted by the owner; future positives still require current protection review and native conversion.

Big2 source assembly is terminal:1333791records,70656chains,1373067authenticated edges. Exact archive747382444bytes SHA `d8946b1242119d4900540f2347fc6619c3fd3958307d98ecfeaec5988d9f7978` and both archive receipts are independently privately durable. Source185636995853bytes remains on the H10029TB-free root. Existing byte-range native conversion is being prepared there with bounded low-priority CPU workers; no185GB DGX copy, admission or training exposure is claimed. Dolci/Cognitive native replay is healthy216/644shards at17:10, two workers on cores7,8/nice19. Current protected split/source review must be checked before admitting any new outputs.

## Controls and remaining bridge work

Run all local monitoring through the memory ledger with SDK interpreter `/home/werg/.local/share/natlang/cw-venv/bin/python`. Private helpers under `/home/werg/data/dgxops`:

- `coreweave-native-health-20261015.py`: compact current output `progress.json`/`metrics.json`, exact active control, GPU/cgroup/errors. Unit1396 terminal0 verified. The earlier compact1381 failed and is not evidence.
- `coreweave-native-rest-v3-progress-20261015.py`: preparation/release phase and bounded logs.
- `coreweave-penny-collection-health-20261015.py`: actual oracle-classified executions; matching JSON retains exact output paths.
- `coreweave-native-{lfm,mellum}-periodic-export-state-20261015.json`: independently verified full-state private durability.
- `coreweave-remote-cpu-job-20261014.py poll <spec>`: owned CPU jobs; use Dolci frontier spec with output_path to count actual completed shards.

Student outputs remain `/work/native-production/{lfm25,mellum21}/native_full_sft`. Current recipe PIDs276225/118821; Mellum trainer118916; use active launch receipts and exact child ownership before signals. Namespace PID1 is an independent holder. Preserve/export useful state before any sandbox stop.

Provider coordinator will live in Mellum `/work/native-production/shared`, IPC `bridge-ipc`; account credentials remain DGX-only. Draft `scripts/coreweave/provider_{bridge,worker}.py` reuses exact owned-process and artifact mechanisms. Eight focused tests passed1392 and a real48-byte signed-URL cloud-to-cloud ferry passed1319. Full typed dependency closure and actual SDK round/refresh/TERM/restart fault qualification remain unfinished. Do not call the provider bridge operational or launch production RL yet.
