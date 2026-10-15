# CoreWeave supervision — 2026-10-15 21:10 UTC

Three owned allocations remain: LFM H10080 `7ee701ae-8e56-49d8-b97a-594f7eb33fe9`, Mellum Blackwell96 `5ffa592c-5f88-4b4f-b3cc-1507ab94a5dc`, and strong Qwen teacher Blackwell96 `dadd0462-0175-470d-b1ee-4d52edfe110e`. Do not manage unrelated `456d8127-1b99-48fa-9a64-edb1bc15de36`. Pop is closed. No further allocation is planned.

## Actual student execution

Both train native positional-KV/hybrid-state graphs using frozen runtime `/work/native-runtime/cf7c53d6019a` (`cf7c53d6019a99e4ea53d623ac9339af083ebbef`). The combined repair preserves exact weights, optimizer, RNG, per-row credits and the fixed initial26312-graph communication foundation. It adds learnable thinking boundaries with FP32 sparse masters and alternates communication with priority-positive teacher batches; all remaining quotas stay mandatory.

| Line | Exact saved handoff parent | Actual new process and progress |
| --- | --- | --- |
| LFM | step2851,2053480587B, SHA `f759a06da46ab9bfb81270bbeda1813b600e089800b0303d06b629c836390511` | recipe386145/trainer386213;21:10 step2950,loss.48711,6333 supervised tokens/s |
| Mellum | step2524,47787918397B,SHA `88c38a08463417c3d5ce2bba1abeabd392159b91b41b5df781adc48f95c16d2e` | recipe214616/trainer214628;21:10 step2561,loss.18709,4203 supervised tokens/s |

Both actual-restored-weight reference gates passed and selected `bf16x6`; FP32 QAD source/master weights and BF16 Mellum backbone remain declared. LFM gate maximum forward7.45e-7/gradient1.49e-7; Mellum forward/incremental0 and gradient4.66e-10, within unchanged1e-4/1e-5 requirements. Actual new thinking input rows move on both models; Mellum separate output rows23/24 also move in FP32 (2298–2303 of2304 coordinates, update L2~2e-5). These numerical checks are not task-quality claims.

At21:10 host anonymous memory8.55GiB LFM/5.22GiB Mellum, both zero OOM/kill counters. Recent CUDA lifetime peaks~9.8GB LFM/~63.9GB Mellum. Update times vary with complete graph geometry; a sampled zero GPU utilization at an RPC boundary does not establish idle training.

Current settings: whole-graph batch32768tokens/chunk1024, checkpoint layers, task every1000updates with exact repeated same-state/action/feedback limit3, held/checkpoint250updates and3600second checkpoint fallback. Held rotation524288tokens; full final held remains mandatory. KL is telemetry only. No current task lift or final production qualification is claimed.

## Exact active inputs and durability

Active train244407 LFM/244408 Mellum rows, held13678 each, genuine communication46899 total. Fixed initial foundation26312 remains unchanged; later communication and ordinary exposure remain mandatory through shared/RL phases.

Train manifests `/work/native-production/data/native-reasoning-all-admitted-append-20261015-v1/{lfm25,mellum21}/train-sources.json`: LF SHA `3aa6f7dab4dcb89762f0d9a6272a6fd486d224e4b87cce1b221cd6de78574d87`, Mellum `4ad05f827773b78314024df641ddc4c39e3c969303c36fe9ee92e5d9dcd6e54f`.

Recipes `/work/native-production/control/{lfm,mellum}-reasoning-all-admitted-handoff-cf7c53d6019a-v3/declared-cadence-recipe.json`: LF current resolved/file SHA `af35b4550f2f761f57aeba60847548e5fe26e4ef7fe51db55f351e748811c73e` (4230B), Mellum `15107e6ea7fa5c1e5b0a571e59ce4bf2964a6510f97db036202f3393e5e34876` (4262B). Original lineage identities are preserved separately. Outputs stay `/work/native-production/{lfm25,mellum21}/native_full_sft`.

The old good212170/212171-row prefix remains exact, with571 rejected teacher targets withdrawn in place and historical consumed visits recorded per CE frame. New32237 rows/model include admitted V5 teacher9535, typed5, REST repair18294, reviewed114+21, and Big2first100 ordinary2115+pair2153. Higher-priority older good Qwen rows precede V5; positive teacher supervision begins immediately, but V5 exposure must be measured rather than inferred.

Independent private full-object GET/SHA durability: LFM2851 is verified with the exact identity above. Mellum2250 is latest terminal verified:47787918397B SHA `75269fb3c7fe1f67f2886190d88905afcf13c5d9aec132d8f4f184ba7a1eb663`;2500 upload finished21:10:56 and independent full GET is in flight. Saved2524 is pinned on cloud but not yet independently privately verified. Export services `cw-native-{lfm,mellum}-periodic-export.service` remain active; private state JSONs are authoritative as the frontier advances.

## Teacher and CPU work

Qwen3.8-Flash-Next NVFP4 RadixArk revision `7b719225242aacd3dbd3f9407468c2ee9a9d2594`, PennyRoyal3.0.1 `fa9c2673131621cafb9a527e61d97c25eac130a9`, server68164/scheduler68332. Eight collectors feed server concurrency4; native MTP/FR-Spec, FP8 dense, NVMePLE and4GB HiCache. Teacher is substantially stronger, not student/on-policy. At21:00 batch12:1485completed/1371 actual successes,22semantic/88invalid/4budget,8active,3826queued. Batch11 completed763/768 successes. Exact immutable per-batch packets/protection witnesses accompany exports; existing materializer excludes rejected intermediate targets. V5 is admitted and active; new later positives await incremental export/admission.

Big2 cloud conversion resumed exact durable2300chains/52820edges at21:07:35. Driver389909/builder389913, job `/work/datajobs/native-big2-graphs-dgx-partition-20261015-v3`, four nice19 workers cores3–6; unchanged output `/work/data-preparation/native-big2-graphs-corrected-source-20261015-v2`. Correct source ID is `neuralese-e2e-cloud-big2-protected-20261015-v1`. Another1000 whole chains (source order60000:61000,10382records/11154edges) are explicitly excluded and assigned to DGX. Extractor391037, one nice19/I/O-idle core2, scans the original185GB only on H100 and writes a compact exact raw-chain subset for existing archive/ferry; no185GB DGX copy. Extraction/conversion is not admission.

Dolci/Cognitive remains live on reserved cores7,8:288/644native shards at21:00,643106closed source records,22explicit paged-evidence omissions. Evidence header V2 (338989train299held/model) seven-file SHA/size checks and private restore completed both sandboxes21:02; staged only. Corrected whole-trajectory V2 (158565train115held/model) staging is in progress, not active exposure.

## Supervision, controller and provider

Local probes use the memory ledger and `/home/werg/.local/share/natlang/cw-venv/bin/python`. Private helpers/receipts under `/home/werg/data/dgxops`: `coreweave-native-health-20261015.py`, `coreweave-native-live-binding-20261015.py`, `coreweave-handoff-detail-20261015.py`, `coreweave-native-reasoning-release-progress-20261015.py`, `coreweave-penny-collection-health-20261015.py`, `coreweave-big2-native-frontier-20261015.py`, and CPU job poll helper/specs. Reads are bounded; health follows actual current output/control, not old diagnostic logs.

The waiting old1938 readiness controller was drained without a live joint optimizer child. New exact recipe/source/controller bindings are being mirrored and validated on Mellum using the existing dependency closure and batched private transfer. Missing nested LF manifests were caught before controller launch. Initial semantic curriculum remains exact26312train/9896held; expanded admitted46899/10369 cohort must become subsequent semantic obligations after its raw communication exposure, without shifting the initial foundation finishline. Full participant/checkpoint/optimizer/readiness/exit ferry and automatic rendezvous still need completed operational hookup; no autonomous end-to-end completion claim.

Actual SDK CPU fixture refresh76artifacts and round12artifacts completed; batched round ferry69seconds. Actual host SIGTERM forwarded to exact remote worker, drained stopped checkpoint at638epochs/global641, and verified all12artifacts/result last. These prove transport/update mechanisms, not full-model task quality. Actual stopped-state restart and injected launch-metadata/terminal-no-result/SDK-timeout recoveries remain to be completed. Participant-sharded shared storage design is recorded in `scripts/coreweave/DECOMPOSITION.md`; it is not implemented yet. Provider and async eval now share `common/runtime_inventory.py` rather than duplicate runtime/protection inventory logic.
