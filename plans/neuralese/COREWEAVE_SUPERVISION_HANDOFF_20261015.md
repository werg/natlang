# CoreWeave supervision — 2026-10-15 21:49 UTC

Three owned allocations remain: LFM H10080 `7ee701ae-8e56-49d8-b97a-594f7eb33fe9`, Mellum Blackwell96 `5ffa592c-5f88-4b4f-b3cc-1507ab94a5dc`, and strong Qwen teacher Blackwell96 `dadd0462-0175-470d-b1ee-4d52edfe110e`. Do not manage unrelated `456d8127-1b99-48fa-9a64-edb1bc15de36`. Pop is closed. No further allocation is planned.

## Actual student execution

Both train native positional-KV/hybrid-state graphs using frozen runtime `/work/native-runtime/cf7c53d6019a` (`cf7c53d6019a99e4ea53d623ac9339af083ebbef`). The combined repair preserves exact weights, optimizer, RNG, per-row credits and the fixed initial26312-graph communication foundation. It adds learnable thinking boundaries with FP32 sparse masters and alternates communication with priority-positive teacher batches; all remaining quotas stay mandatory.

| Line | Exact saved handoff parent | Actual new process and progress |
| --- | --- | --- |
| LFM | step2851,2053480587B, SHA `f759a06da46ab9bfb81270bbeda1813b600e089800b0303d06b629c836390511` | recipe386145/trainer386213;21:42 step3000,loss.22094,4846 supervised tokens/s; current-weight task evaluation active |
| Mellum | step2524,47787918397B,SHA `88c38a08463417c3d5ce2bba1abeabd392159b91b41b5df781adc48f95c16d2e` | recipe214616/trainer214628;21:42 step2750,loss.08714,2630 supervised tokens/s; held/checkpoint boundary |

Both actual-restored-weight reference gates passed and selected `bf16x6`; FP32 QAD source/master weights and BF16 Mellum backbone remain declared. LFM gate maximum forward7.45e-7/gradient1.49e-7; Mellum forward/incremental0 and gradient4.66e-10, within unchanged1e-4/1e-5 requirements. Actual new thinking input rows move on both models; Mellum separate output rows23/24 also move in FP32 (2298–2303 of2304 coordinates, update L2~2e-5). These numerical checks are not task-quality claims.

At21:10 host anonymous memory8.55GiB LFM/5.22GiB Mellum, both zero OOM/kill counters. Recent CUDA lifetime peaks~9.8GB LFM/~63.9GB Mellum. Update times vary with complete graph geometry; a sampled zero GPU utilization at an RPC boundary does not establish idle training.

Current settings: whole-graph batch32768tokens/chunk1024, checkpoint layers, task every1000updates with exact repeated same-state/action/feedback limit3, held/checkpoint250updates and3600second checkpoint fallback. Held rotation524288tokens; full final held remains mandatory. KL is telemetry only. No current task lift or final production qualification is claimed.

## Exact active inputs and durability

Active train244407 LFM/244408 Mellum rows, held13678 each, genuine communication46899 total. Fixed initial foundation26312 remains unchanged; later communication and ordinary exposure remain mandatory through shared/RL phases.

Train manifests `/work/native-production/data/native-reasoning-all-admitted-append-20261015-v1/{lfm25,mellum21}/train-sources.json`: LF SHA `3aa6f7dab4dcb89762f0d9a6272a6fd486d224e4b87cce1b221cd6de78574d87`, Mellum `4ad05f827773b78314024df641ddc4c39e3c969303c36fe9ee92e5d9dcd6e54f`.

Recipes `/work/native-production/control/{lfm,mellum}-reasoning-all-admitted-handoff-cf7c53d6019a-v3/declared-cadence-recipe.json`: LF current resolved/file SHA `af35b4550f2f761f57aeba60847548e5fe26e4ef7fe51db55f351e748811c73e` (4230B), Mellum `15107e6ea7fa5c1e5b0a571e59ce4bf2964a6510f97db036202f3393e5e34876` (4262B). Original lineage identities are preserved separately. Outputs stay `/work/native-production/{lfm25,mellum21}/native_full_sft`.

The old good212170/212171-row prefix remains exact, with571 rejected teacher targets withdrawn in place and historical consumed visits recorded per CE frame. New32237 rows/model include admitted V5 teacher9535, typed5, REST repair18294, reviewed114+21, and Big2first100 ordinary2115+pair2153. Higher-priority older good Qwen rows precede V5; positive teacher supervision begins immediately, but V5 exposure must be measured rather than inferred.

Independent private full-object GET/SHA durability: LFM2851 is verified with the exact identity above. Mellum2500 is latest confirmed terminal verified:47787918397B SHA `76d67342c560c84f06b8f466fc3825b36a088a07b90893f2751f348b7702c4be`, independent full GET completed21:14:18. Saved2524 is pinned on cloud but not yet independently privately verified. Export services `cw-native-{lfm,mellum}-periodic-export.service` remain active; private state JSONs are authoritative as the frontier advances.

## Teacher and CPU work

Qwen3.8-Flash-Next NVFP4 RadixArk revision `7b719225242aacd3dbd3f9407468c2ee9a9d2594`, PennyRoyal3.0.1 `fa9c2673131621cafb9a527e61d97c25eac130a9`, server68164/scheduler68332. Eight collectors feed server concurrency4; native MTP/FR-Spec, FP8 dense, NVMePLE and4GB HiCache. Teacher is substantially stronger, not student/on-policy. At21:42 batch12:1665completed/1532 actual successes,24semantic/105invalid/4budget,8active,3826queued. Batch11 completed763/768 successes. Exact immutable per-batch packets/protection witnesses accompany exports; existing materializer excludes rejected intermediate targets. V5 is admitted and active. V6 immutable delta498 exact-success executions exported to DGX; conversion/admission is pending, including authenticated text-block context reconstruction.

Big2 cloud conversion resumed exact durable2300chains/52820edges at21:07:35. Driver389909/builder389913, job `/work/datajobs/native-big2-graphs-dgx-partition-20261015-v3`, four nice19 workers cores3–6; unchanged output `/work/data-preparation/native-big2-graphs-corrected-source-20261015-v2`. Correct source ID is `neuralese-e2e-cloud-big2-protected-20261015-v1`. Another1000 whole chains (source order60000:61000,10382records/11154edges) are explicitly excluded and assigned to DGX. Extractor391037 completed21:31:02; compact12,018,978B archive SHA `13f01bc4a89c57571f9f2d6fc0deba6f92f5ac220a36d477dd4da0c36be67557` privately verified and restored21:43:53 to `/mnt/external/natlang-development-data/big2-dgx-offload-1000-20261015-v1`. Exact source subset1.525GB,920train/80held chains, all21files verified; data agent owns validation/conversion. Cloud excludes these1000 and advanced2400chains/55139edges at21:31. Extraction/conversion is not admission.

Dolci/Cognitive remains live on reserved cores7,8:296/644native shards at21:31,643106closed source records,22explicit paged-evidence omissions. Evidence header V2 (338989train299held/model) seven-file SHA/size checks and private restore completed both sandboxes21:02; staged only. Corrected whole-trajectory V2 (158565train115held/model) and evidence FULL view17048train0held/model are also staged with exact file guards on both sandboxes; none of these three views is active exposure yet.

## Supervision, controller and provider

Local probes use the memory ledger and `/home/werg/.local/share/natlang/cw-venv/bin/python`. Private helpers/receipts under `/home/werg/data/dgxops`: `coreweave-native-health-20261015.py`, `coreweave-native-live-binding-20261015.py`, `coreweave-handoff-detail-20261015.py`, `coreweave-native-reasoning-release-progress-20261015.py`, `coreweave-penny-collection-health-20261015.py`, `coreweave-big2-native-frontier-20261015.py`, and CPU job poll helper/specs. Reads are bounded; health follows actual current output/control, not old diagnostic logs.

The waiting old1938 readiness controller was drained without a live joint optimizer child. Exact recipe/source/controller bindings and all nested inputs were mirrored and validated21:26:46. Canonical waiting controller guardian223035/child223054 uses frozen4820 runtime, shared output `/work/native-production/shared/native_shared_stage`, RL output `/work/native-production/shared/native_rl`, config root `/work/native-production/shared/deployment`. Missing nested LF manifests were caught and fixed before launch. Initial semantic curriculum remains exact26312train/9896held; expanded admitted46899/10369 cohort must become subsequent semantic obligations after its raw communication exposure, without shifting the initial foundation finishline. The future46899train/10369held semantic cohort is staged on both sandboxes without changing the26312 initial foundation. Full participant/checkpoint/optimizer/readiness/exit ferry and automatic rendezvous still need completed operational hookup; no autonomous end-to-end completion claim.

Actual SDK CPU fixture refresh76artifacts and round12artifacts completed; batched round ferry69seconds. Actual host SIGTERM forwarded to exact remote worker, drained stopped checkpoint at638epochs/global641, and verified all12artifacts/result last. These prove transport/update mechanisms, not full-model task quality. Actual stopped-state restart and injected launch-metadata/terminal-no-result/SDK-timeout recoveries remain to be completed. Participant-sharded shared storage implementation is under focused validation and independent review, not deployed. Exact selected participant/VAE/RNG closure, complete recovery, atomic failure cleanup and closure-aware pin retention tests use actual two-port weights/optimizers. Existing monolithic exporter must adopt full closure snapshots before any sharded production activation. Provider and async eval now share `common/runtime_inventory.py` rather than duplicate runtime/protection inventory logic.

LF3000 current evaluation at21:43 had14/16 terminal cases:1success,2semantic,2repetition,9unresolved. Four preserved first-request errors enter a produce span then hit8192tokens or EOS/server stop; requests have no forced template. Exact terminal token IDs were not persisted by failed non-RL streams. Held-only diagnostic pack SHA `20f208e2f5ed51a2344f19d0ea09b4fe6bb66c38d128bb1f5d274679baf422b3` is under `/home/werg/data/dgxops/lfm-native-task3000-held-diagnosis-20261015`. Do not score unresolved rows as model failures or alter the in-flight evaluation.

## 22:05 UTC operational update

LF3000 routine evaluation drained21:46:46 (28m03), but the old cadence failure
counter then terminated training at21:46:55 with ten unresolved results. The
completed observed counts were1success/2semantic/3repetition/10unresolved;
this is incomplete evidence, not a valid1/16 model score. The safe resume used
exact full parent3000,2068262663B,SHA
`fd7932ad14128070fbcd096f9ccf1a4a9f049222363551f96dede5080cc3dfd8`,
optimizer/RNG/exposure7575 preserved. Recipe403619 under qualifiedcf7 resumed;
actual22:01:43 step3019/loss.213293/6054.6supervisedTPS confirmed.

Current LF recipe is
`/work/native-production/control/lfm-routine-async-cadence-cf7c53d6019a-20261015-v1-v2/declared-cadence-recipe.json`,
4227B SHA`802f4cda44537b37289ee90f5cec881b48001663a7954737149bb7bac692c64c`.
Only routine task cadence1000→0 changed. Existing DGX verified-export task
dispatch supplies historical monitoring; current-weight foundation/refresh/final
and held checks remain mandatory. Canonical readiness metadata is being refreshed
to this actual recipe and ownership path, without restarting either trainer.
Mellum22:01 step2866/loss.129991/2200.2supervisedTPS remained active. Exact
checkpoint3000LF/2750M optimizer metadata shows every groupLR3e-6, including
role/thinking sparse tables in torch-Muon; no learning-rate adjustment was made.

Shared checkpoint storage dd4b689d0 passed root's independent code review and
actual SDK CPU/tiny two-port qualification. Existing signed artifact ferry moved
20files886381B from Mellum to H100; the peer payload was absent on H100. Same
globalSHA`170f1cf471a09847e6da6749994edbf57cb57445aa1eac178d6af3e7ff6334b5`
restored exact selected weights, optimizer, RNG, CE credits and VAE; one next
real optimizer update matched source loss and every resulting weight/moment.
Proof`/home/werg/data/dgxops/coreweave-shared-selected-sdk-proof-20261015.json`.
This qualifies storage/transport on real tiny models, not full trained-parent
CUDA arithmetic or task quality. Scheduled exporters must use complete closure
pins before exporting this new form; no manifest-only durability claim is valid.
Production shared activation still waits actual foundations and qualified runtime.

## 22:21 UTC controller and admitted-data continuation

Fresh22:17UTC actual progress: LF3117/loss.169125/5133supervisedTPS,
M2961/loss.287712/4302supervisedTPS. Sparse thinking masters areFP32 and
have actual nonzero gradients/changed coordinates. Private fullGET/SHA now
verifies LF3000 above and M2750,47802833061B,SHA
`56f085aefd05b9ddfb42497468e4038a5181f4247c192289ba8f54f6a0d2a00f`.
Teacher batch12:1811finished/1655success,28semantic/123invalid/5budget,
1819started/8active of3826queued. Big2 cloud2900chains/66748edges; the
reserved1000-chain DGX subset is separately converting with2workers and remains
excluded from cloud work.

Deployment guardbdadc5a20 now permits routine cadence0 when a real task spec
remains declared. Four file-bound tests and quick passed. The CPU builder
regenerated actual LF802f recipe bindings and all nested dependencies; the
qualified4820 waiting controller was rearmed22:20UTC, guardian234791/child234910,
at `/work/native-production/shared/deployment/readiness-supervisor-4820-async-cadence-20261015`.
Actual child ownership and monitoring state were verified; no joint optimizer
child or ready participant is claimed. This uses new code only for CPU metadata
construction, never silently enables new FP8 arithmetic in a student.

The all-view source append is concretely queued on qualifiedcf7. Preparation
`/work/native-production/control/{lfm,mellum}-evidence-all-views-prewarm-cf7-20261015-v1`
and release guardians at matching `*-evidence-all-views-handoff-cf7-20261015-v1`
wait for exact source-index/duplicate/group checks and plan-only before signaling
any trainer. FULL17048 is already duplicate-clean/warm; header338989 and
whole158565 are validating. Expected new rows514602/model preserve the full
244407/244408 prefix and every credit. Any duplicate-ID or held-group conflict
stops preparation before a training interruption and requires explicit view
accounting. Mellum's prepared continuation changes routine tasks1000→5000;
LF remains0+historical async. Held/checkpoint250 and mandatory current-weight
foundation/refresh/final checks remain, with unchanged26312 foundation.
No new view has training exposure until actual resumed updates prove it.

Root's actual both-model executed-FP8 joint proof passed onDGX Torch2.11/cu130/
SM121, not targetNGC2.14/cu134/SM120. New grouped-FP8/ordered-mask code remains
outside this data-onlycf7 continuation. A separate exact-parent target-runtime
check is authorized for the next useful numerical handoff; prefix sharing isOFF.
