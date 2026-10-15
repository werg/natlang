# Current supervision update, 2026-10-15 11:58 UTC

Both existing allocations continue as diagnostics while the new native shared-state trainer is integrated. No new allocation or lineage was created.

- Mellum, Blackwell sandbox `5ffa592c-5f88-4b4f-b3cc-1507ab94a5dc`: actual shared mixed optimizer updates now work. Step3585 had finite loss0.58039 and5.53seconds for30715 forward tokens; training GPU utilization was100%. Step3600 committed a47,078,033,383-byte full checkpoint, including the added mixed state, and its scheduled16-task evaluation is running. The latest independently private-verified parent remains3400: SHA `4ac8ef724e46f255c0bb13eeb2eee1705d96e4cece84139cea9268d137d2fe0e`. The periodic exporter is active. No3600 score or durability pass is claimed before completion. Last completed task2400 remains5/16 versus reference13, unqualified.
- LFM, H100 sandbox `7ee701ae-8e56-49d8-b97a-594f7eb33fe9`: exact28000 full-state resume is released under trainer203166, sourcef5e6e1726b, but CPU window staging has not yet produced28001. The preceding run stopped on the runtime-ineligibility counter after task28000 scored1/16 with one unresolved transport failure. Its diagnostic policy now retains incomplete/unqualified status and counter while permitting partial-runtime training when a majority executes and actual quality comparisons meet; total runtime failure still stops. Full private28000 is independently verified:1,149,841,883bytes, SHA `737f37ee855c68830a003e2ac6dc9591dd0ac2319e72f085bdb61f2e60dc8fbe`. No counter reset or model-negative relabel occurred.

The actual new native FP32 interface qualification ran in an isolated published133f2ce0 runtime on the H100 with pinned QAD weights and all four new markers. NGC defaults (`matmul_precision=high`, matmul/cuDNN TF32 enabled) failed unchanged tolerances: forward8.33e-4, consumerCE1.31e-3, channel-gradient3.61e-4 and incremental5.15e-4. The otherwise identical highest-matmul/TF32-disabled control passed: forward1.55e-6, CE7.63e-6, gradient6.41e-7 and incremental4.77e-7; real channel optimizer update and exact channel/optimizer restore passed,21.60seconds,1.687GB peak. Both receipts are preserved under `dgxops/coreweave-native-lfm-{qualification,precision-control}-20261015.json`. This localizes the combined precision configuration, without separating individual operator contributions. It does not qualify full backbone training, BF16, task capability or the complete new recipe. The architecture owner is preparing explicit kernel precision contracts; no canonical/live runtime was hot-mutated.

Prepared-cache investigation found that the generic prewarm helper incorrectly used held target_tokens1024 for LFM, whose actual recipe uses256. Those completed prewarms cannot be called cache hits for the live geometry. The helper now derives geometry from the resolved recipe; the wrong WT CPU worker alone was stopped, and exact256 PH prewarm runs as `r11-ph-prewarm-lfm-50e5a079`. The live correct-geometry staging continues. No GPU trainer was restarted for this correction.

Rest27 E2E admission artifacts and their lazy record indexes are fully staged/verified on both boxes:147414records/11713chains/55484edges. The two largest source archives completed member-SHA restore; their existing protected assembler is progressing. Dolci/Cognitive reached99/644 completed native shards with zero OOM kills. Search/OWL native data remains privately durable, with DGX restore/registration owned by the data helper. Full/PH/WT/new FULL inputs are staged, but no new legacy destination is introduced while the replacement architecture is built.

# Historical supervision update, 2026-10-15 11:04 UTC

The owner now directs both existing cloud runs to continue as diagnostics while the shared-state Neuralese replacement is implemented and qualified. These continuations are not the destination architecture. No third GPU or new lineage was allocated.

- LFM remains active on H100 sandbox `7ee701ae-8e56-49d8-b97a-594f7eb33fe9`, trainer138715. Update26745 had finite loss0.571,0.441seconds and38,489 supervised tokens/second. Task24000 was0/16:4invalid,3repetition,7semantic,2unresolved transport failures; no lift or qualification claim. Both actual error traces and representative semantic/invalid traces are durable on DGX. Full checkpoint26840 is privately verified by independent full readback:1,149,841,755bytes, SHA `1942307546a43b5ba78719a0b8a882b56acc5948a14ef48c4cdeaaf5d23569d6`. Current and predecessor PH prepared-window/2 caches are building on spare CPUs while the GPU continues training; the existing same-allocation handoff controller waits for their verified completion and an evaluation-safe cut.
- Mellum on Blackwell96GB sandbox `5ffa592c-5f88-4b4f-b3cc-1507ab94a5dc` is relaunching the shared mixed diagnostic from exact full state3400, preserving optimizer/RNG. Full parent46,822,799,447bytes, SHA `4ac8ef724e46f255c0bb13eeb2eee1705d96e4cece84139cea9268d137d2fe0e`, is independently privately verified. No mixed optimizer update has completed yet. Last actual plain-SFT update3417 preceded3418OOM; safe3400 is retained. Task2400 remains5/16 versus inherited13, unqualified.

Concrete mixed-startup bugs were repaired without dropping data/context: explicit layer checkpointing plus exact FP8 saved-quantized-weight accounting (603b7e0e); absent-to-absent QAT objective guard (0895de8164); exact MuonSR stacked grad_norm shape and original optimizer state dtype/value restoration (7a4bc9652d); and compatible decode-only FlexAttention32-query/64-key tiles for128-token sparse blocks (752fde9ea6). Last failed trainer59071 completed optimizer restoration, then hit the Flex compiler tile error before any mixed update. Actual Blackwell Torch2.14/CUDA13.4 cached-window and branch forward/backward qualification passed at11:03:27 with GQA32/4,dim128, cached lengths1031/8364: maximum output error0.00391 and relative gradient errors below0.00327 against math attention. Seven focused CPU tests and quick gate passed; no uncompleted full gate is claimed. Current launch source/recipe seal is c59ba570e1, with unchanged3400 parent.

R11 PH and WT input bytes are SHA-verified on both sandboxes. The newer FULL cohort `natlang-target-union-20261015-next-incremental-v2-full-ready-v1` is also staged on both. First E2E cohort is admitted; rest27v2 is now admitted with147414 records,11713 complete chains and55484 authenticated edges, and is being staged for the next useful shared continuation. The two largest mined sources finished exact restore and their existing protected assembler is running on one spare CPU. Dolci/Cognitive has82/644 completed native shards; H100 anonymous host memory was12.82GB/64GiB with zero OOM kills. Search/OWL native data is privately durable and the data agent owns DGX restore/registration.

The current Mellum legacy continuation budget22032 must not be described as comprehensive window/2 coverage: a same-data legacy-to-window-occurrence upgrade remains assigned for the next useful continuation. Training admission, archive durability and task qualification remain separate facts.

# Historical supervision update, 2026-10-15 09:18 UTC

Both existing allocations remain live; no third GPU or sandbox replacement was used.

- LFM: H10080GB, sandbox `7ee701ae-8e56-49d8-b97a-594f7eb33fe9`, trainer138715. Exact15300 model/optimizer/RNG resumed with shared Flash attention and adaptive report-only KL policy. Actual update19034 completed at28,121 supervised tokens/second; task16000 finished1/16 in575.44seconds, versus base0 and previous0. Current inputs remain the historical R8 PH snapshot, pending the protected R11 PH handoff. Full private checkpoint18480 is independently verified:1,149,840,539bytes, SHA `0c95ff6da3574ac7f8d1aef30611a8d8c30de3ae046ab43e6deefcdf8e522a6f`.
- Mellum: RTX PRO6000 96GB, sandbox `5ffa592c-5f88-4b4f-b3cc-1507ab94a5dc`, trainer37550. Exact3108 model/optimizer/RNG now continues on admitted R11 FULL, one-epoch budget,22032 total planned updates. Actual resumed update3111 completed with finite loss0.0091711 and26622 forward tokens; first-update timing7.60seconds includes startup effects. Prepared-window cache HIT. Task2400 remains5/16 versus inherited base13, unqualified. Full private parent3108 is independently verified:46,821,492,055bytes, SHA `e21f1df8312037c625ebb5497478221375362fe4594e60d7ee50477cd91792f2`.

Mellum's first R11 attempt exited before any update because the epoch-budget guard compared old-data batches against the declared new-data continuation's consumed-update placeholders. Shared fix141a4a8462 validates that continuation prefix and retains exact consumed-batch comparison for epoch-only changes;17 focused tests and quick gate passed. Recipe2361c947a7 reseals the shared repair; no weights, optimizer, RNG or task controls reset. A later shared fixb12301e486 lets continuation planning reuse the same validated prepared-window cache, with equal descriptor/schedule/held identities (17 tests and quick gate passed); it is not hot-applied to the running trainer. The separate newly diagnosed document-level continuation coverage issue is being repaired by the data auditor; original3108 state/data/schedule remain retained so omitted windows can be recovered at a future handoff.

The optional H100 native causal-conv selector is published in a68c8d5d41 with2 focused selector tests and quick gate passed. It preserves Pop's sm89 mapping. The exact sm90 wheel has separate bitwise operator qualification, but is not installed or activated on the frozen live trainer. Broad containing checks use the standing full-check service; no uncompleted full gate is claimed passed.

Data work on spare cloud CPUs: Dolci/Cognitive progressed53/644 native shards with zero OOM. Search/OWL's complete658396-decision graph/native/source archives are privately durable; registration/admission is separate. The first protected E2E cohort is admitted (`neuralese-e2e-cloud-small-protected-20261015-v2`):4221 records/340 whole chains/1180 authenticated edges. Next27-source assembly produced147473 records/11715 chains/55489 edges, but independent reauthentication caught a missing producer/provider receipt in one chain; this cohort is unadmitted while the exact source/scope issue is investigated. Its immutable source archives and explicit omissions remain preserved. All33 source archives are privately durable; both remaining large sources are now streaming restore in job `e2e-big2-restore`, PID159001, spare core5/nice19/zeroCUDA, with archive and per-member SHA verification.

All checkpoint exporters remain active on the same allocations. Next useful handoffs append the shared PH/WT/E2E phases as their protected inputs become available; Mellum checkpoint cadence600 is approved for that handoff to reduce measured checkpoint overhead. No new benchmark is a prerequisite to useful training.

# Historical supervision update, 2026-10-15 07:45 UTC

Mellum is actively training on `5ffa592c-5f88-4b4f-b3cc-1507ab94a5dc`
(RTX PRO 6000 96GB, sm120). Task2400 finished **5/16**, versus inherited base13
and previous4:5success,2invalid,4repetition,1budget,4semantic failures,
zero infrastructure/model-incomplete/unresolved. Duration828.65seconds.
Training resumed and passed2420 without errors. Full checkpoint2210 is
privately durable:46,821,491,287bytes, SHA
`d8c7352bbaa7af4e7052dc95aa812a6cc513e410e2981b0ae0a4ec5a367a27d1`.
Periodic full exports remain active. Terminal2400 prompt/output evidence and
training metadata were independently collected to DGX (bundle
`8a675d28419f1808bbed9f1f7f46e6907a1a6b7849008b1b54542c42218ddd51`).

LFM's exact15300 full weights/optimizer/RNG remain in held H100 sandbox
`7ee701ae-8e56-49d8-b97a-594f7eb33fe9`, SHA
`af52732e404cc657e67b77d7067a6dbb6cec41b63e80b1abbe53e810672f6783`.
It previously stopped at responseKL1.5195. The owner's new instruction removes
KL as an automatic stop/qualification condition; the shared policy/recipe
update is being prepared by the main supervisor. The next useful action is
resume15300 with that code and Flash attention, retaining exact optimizer/RNG
and original reference. No rollback, new research lineage or additional
benchmark is required before that authorized training release.

The exact stopped-weight Flash forensic finished all16 tasks in485.77seconds:
**0/16**,10repetition,1budget,2model-incomplete,3semantic, zero infrastructure
or unresolved. Behavior48 was15/48, report-only. RSS stayed1.71GB; previous
cuDNN serving attempts grew above35–55GB before their unfinished evidence was
saved. Actual shared cuDNN attention reproduced per-shape native-heap growth;
repeated identical shapes plateaued, and Flash changing shapes plateaued.
The v8 convolution cache environment control did not fix this SDPA path.
Full exact-model comparison on two real served prompts plus16 forced decode
steps each preserved34/34 greedy choices (maximum measured responseKL0.001396).
The separately labeled forensic changed no optimizer weights or original
failed report/certificate/task-history bytes. Full evidence survived to DGX,
bundle `3f5c21b614a8f21329dad8bf4f5eb519c71d87d30cc7fa7bed1377a53a1678be`.
This H100 finding does not explain the separate Mellum/DGX memory issue.

Search/OWL complete closed handoffs, native graph/IR,658,396decisions and
provenance are now privately durable with independent full GET/SHA readback.
The44,531,753,504-byte native tree is an11,382,584,581-byte archive, SHA
`0bb13659806aa6ffed12a4ca61c68db9caf30f6f4ccc65a7d472d953611b9a75`.
Source/native registration and training admission remain separate. Dolci/
Cognitive's existing native replay is progressing on its643,106protected-closed
train records;20/644shards were complete at the last counted frontier.

All33 mined E2E source archives and their receipt/roundtrip-proof objects now
passed private full GET/SHA verification:206,559,557,124raw bytes compressed to
959,852,823archive bytes. No raw source was deleted by this supervisor. These
archives preserve complete graph/source evidence and do not grant training
admission. The separate16 acquired external-source archives remain privately
durable, with four access-blocked source omissions explicit. Production inputs
are updated through the shared recipe as protected data becomes usable.

Earlier entries below are historical execution evidence.

# CoreWeave production state, 2026-10-14 19:05 UTC

Two owner-authorized full-weight shared production recipes were released at
18:49 UTC. This is current execution evidence, not stage qualification.

| Line | Sandbox | Actual GPU | Host request | Output |
|---|---|---|---|---|
| LFM QAD R8 PH continuation | `7ee701ae-8e56-49d8-b97a-594f7eb33fe9` | H100 80 GB, sm90 | 8 CPU, 64 GiB | `/work/runs/lfm-r8-ph-production-20261014/direct_sft` |
| Mellum fresh R8 warmup | `a7858948-0e1a-4c89-bb69-e2cc29cbc243` | RTX PRO 6000 Blackwell 96 GB, sm120 | 4 CPU, 64 GiB | `/work/runs/mellum-r8-fresh-production-20261014/direct_warmup` |

Both use no-volume root storage and NGC `nvcr.io/nvidia/pytorch:26.09-py3`.
Requested tag and official amd64 manifest SHA
`f71fd67f6d71f685e0bd08ce6c8df48807624cc5d48886370d3471bc4f89d81d`
are separately recorded in launch receipts. Actual runtime: Torch
`2.14.0a0+b2c75dd062.nv26.09`, CUDA 13.4, Transformers 5.19.0.
The LFM root disk is approximately 30.7 TB; Mellum approximately 7.68 TB.

LFM uses recipe `neuralese-direct-lfm-qad-r8-ph-coreweave-20261014-v1.json`
(commit `225b60a7`) with `--handoff`, exact qualified Pop warmup410 full
checkpoint SHA `13ebca2a4c2f2f087ce2cfc4a635114b4aa2b91a017714958bed71cd33551761`,
its optimizer/RNG, certificate/identity/monitor reference and clean raw heads.
All five handoff artifacts were SHA-verified in the sandbox. Protected PH prefix
SHA `60c69c269ac7cfc6ee26cbc5d5c78c2d72e98d14ed8370027fe9e800b5c3edf8`
was privately transferred and independently SHA-verified. Its full data scan
reached 90.15% at 19:03 UTC. First update/restore qualification remains pending. At19:25 the final merged-row
window scan was62.79% complete; startup is three streamed passes (validation,
dedup plus merge metadata, merged windows), not GPU work.

Mellum uses separately declared FRESH recipe
`neuralese-direct-mellum-r8-clean-warmup-coreweave-fresh-20261014-v1.json`
(commit `2cfa56ea2b`), identical pinned clean base/raw heads/admitted R8 full
warmup inputs. This is not a DGX checkpoint/optimizer resume. At 19:04 UTC
94 updates completed without errors. Step-0 identity reports passed=true and
logits_equal_base=true. Example update94: 3.651 seconds, 29,745 forward tokens,
CUDA peak allocated 55.25 GiB/reserved61.06 GiB. GPU utilization98–100%.
Actual full-context packed windows reach32767 tokens with checkpointing.
Planner uses actual101,975,851,008-byte device capacity. Actual runtime installs
28 FP8 expert modules using `triton-grouped-experts-v1` on sm120 and168 MuonSR
latents. These are real trainer evidence; additional optimization claims require
measured dispatch/runtime receipts. Production schedule is4851 updates at0.5epoch,
checkpoint cadence170 and task/CE evaluation400.

## Durability and lifecycle

Private project `werg/natlang` access was explicitly verified PRIVATE before
upload. PH transferred as immutable private artifact; no owner credential enters
the sandbox. A faster cloud export path uses official W&B CreateRunFiles to
issue an exact-file presigned PUT URL on DGX, then sends only that capability
and required headers to the sandbox. Actual Mellum identity168bytes was uploaded
HTTP200 in0.273s and independently retrieved with exact SHA
`e1059804617edeb679d1ffa62ea8778057c59df05c73985aa09ab1fd2bdc56b1`.
Private export run `werg/natlang/ywcj2mvb`. Full checkpoint export is pending;
small identity export does not establish full model/optimizer durability.
A ledgered watcher awaits the first atomic Mellum checkpoint, makes an immutable
root-disk copy, then privately uploads and independently hashes the fetched
object from cloud. This avoids a47GiB DGX root allocation and slow gateway transfer.

Manual release mistakenly exec'd the recipe as PID1 rather than the existing
bootstrap hold wrapper. PID1 exit can end the sandbox and file API. SIGSTOP was
attempted but verified INEFFECTIVE (PID1 remainsState=S); do not assume held.
Current recipe PID1, actual LFM trainer9887, Mellum trainer5860. Stop requests
must target the exact active trainer and wait for full atomic state commit and
durable export before SDK stop. Ordinary trainer SIGSTOP can hold fileAPI alive
before terminal exit if required; it has not been applied. Subsequent launches
must use existing `scripts/coreweave/bootstrap.sh` API-transport hold/collect
mechanism. Do not stop either pod or infer root-disk persistence after termination.
Parent retains local DGX Mellum until steady cloud training AND full checkpoint
save/export are verified; local job lifecycle belongs to the parent.

Exact private launch, progress, transfer and export receipts are under
`/home/werg/data/dgxops/coreweave-production-*20261014.json`.
Signed capabilities and W&B runtime logs remain outside the checkout, mode600
or private directories; never commit them. D40 CPU conversion shares LFM root
disk in a frozen runtime and is parent-owned. Both production allocations remain
running; all earlier seven sandbox diagnostics were independently verified stopped.

## Reader cardinality proof, 19:24 UTC

Raw656263 rows become656262 reader rows through one existing exact-key duplicate,
not missing data or a held collision. Cache skip set is exactly{250111}
(zero-based). Earliest matching row46536 is retained. Both are train,
agentic-tool-loop, source groups
`swe-rebench-instance:Clinical-Genomics__patientMatcher-75` and
`swe-rebench-repo:Clinical-Genomics/patientMatcher`.
Kept ID `harness:openhands-v0.54:chatcmpl-87de9c62149901120aee88c6287d0bba:turn:0005`;
duplicate `harness:openhands-v0.54:chatcmpl-ea839420aa7d4dbdfb1c2a5dcb079e23:turn:0005`.
Full duplicate key: split=train,
text SHA `a59f7e173070e6ebe6195b5382fdb56a735dcf49cac69a08bf8e0af8188a40b0`,
token IDs SHA `9dcaa0cb390bb2fabaa45e0ecc88e6f5a65ea1d88447f2e629f20d7d6960a561`,
supervised suffix1658, sampling_group=null. Entire corpus scan found exactly
these two key matches and zero held same-text lines. Diagnostic terminal0;
receipt `/home/werg/data/dgxops/coreweave-production-lfm-reader-dedup-proof-20261014.json`.
No admission/training change required.

Mellum first atomic full state at170 committed19:13:25,
46,821,474,583bytes. Immutable export copy completed138seconds later;
SHA `93737f4e87454a3f07c642bec2fe36e5d9fcb8fa0b8e8027d4999c363f442381`.
Actual private upload endpoint is `storage.googleapis.com`; GCS supports5TiB
objects across uploadmethods ([official quotas](https://docs.cloud.google.com/storage/quotas?hl=en)).
At19:25 scoped PUT process had read~22GB/46.8GB in370seconds (~59MB/s),
with24MB RSS, while actual trainer continued326updates/GPU100%.
HTTP completion and independent remote SHA verification remain pending.

## Current lifecycle and real task evidence, 20:25 UTC

LFM's first direct recipe PID1 exited/restarted around19:32:45, before its
first update. The new container reset ephemeral root files, including training
logs and auxiliary CPU acquisition/conversion jobs. New cgroup OOM counters
were zero and kernel logs showed no OOM kill; these observations do not prove
that the previous container avoided OOM. A recipe/continuation exception is
also possible; redirected old logs were lost. Cause remains unproven.

The SAME sandbox was fully restaged and released19:59:14 using existing
`bootstrap.sh`: central stdout plus a local tee, explicit child exit receipt,
API output manifests and30day collect hold. Current code archive commit
`33d0f44b7450508ed52192efb7a13ce6ad825afe`, SHA
`6497389dd94d0f5b75cf81272de34ef5dc7849c81558f738a3fddb8b0ae66372`.
Exact225recipe, QAD410 parent/optimizer/RNG and protected PH bindings remain.
A20second external resource sampler records cgroup current/peak/events and
process RSS from onset to central logs. No auxiliary LFM data jobs run during
startup. Parent files now also have durable PRIVATE run `werg/natlang/jer8wefd`;
owner-side checkpoint PUT took615.9seconds, followed by fast cloud fetch.

Existing row-index/prefix-map caches for this exact PH input were independently
exported20:23: `coreweave-production-lfm-ph-row-index-caches-20261014.tar.gz`,
36,031,901bytes, SHA
`679bc25486eb3d092d593b4c80b950354f293ad8b5e7bcd7775452db8d438a0e`.
Assigned consumer: active LFM restart diagnosis/relaunch, avoiding repeated
validation and prefix-map scans. No data admission or model code changes.

Mellum full checkpoints170 and400 are durably PRIVATE and independently
re-read from GCS with full SHA equality:

- 170:46,821,474,583bytes, SHA
`93737f4e87454a3f07c642bec2fe36e5d9fcb8fa0b8e8027d4999c363f442381`,
verified19:37:29; full readback201.882seconds.
- 400:46,821,477,015bytes, SHA
`f598a6598293ad14df4b18012377fb25992c794921454ee28ec55451fdf93276`,
verified20:11:26; PUT868.177seconds, full readback151.959seconds.

Run `werg/natlang/ywcj2mvb`; content-addressed run-file paths are
`mellum/<SHA>/mellum-first-checkpoint.pt` and
`mellum/<SHA>/mellum-checkpoint-step-400.pt`.
`cw-mellum-periodic-private-checkpoint-export` serializes newest atomic state
exports and stores verified progress in external dgxops. A download URL requires
owner-side authenticated redirect resolution; only final scoped GCS capability
enters cloud. The initial readback guard failed safely on an API redirect and
was fixed; no second upload was required. Parent gracefully stopped local DGX
production duplicate after first cloud full-state durability was verified.

At400, heldCE0.87935 versus1.81925base (delta−0.93990), responseKL0.2393.
Drift330's report-only probe gate37/48 compared with28/48base. Actual16task
pass is still incomplete:10finished,4success,1invalid action,3repetition,
2semantic mismatch,6unresolved. This is an assigned bug investigation; no
qualification or approach judgement. Reference13 is inherited, not a freshly
executed cloud base task reference. Actual recipe/queue temperature0.6 is
honored. A separately found generic omitted-zero bug does not explain these
specific controls. The parent is reproducing exact failures against base with
matching controls/runtime while cloud evaluation continues.

All available task400/noise evidence was SHA-verified on DGX20:02:
`task400-evidence-1792008149.tar.gz`,35,092,343bytes, SHA
`79c2361750d1ac4dc0c9d904e9f9224d84d6b4ba234ba611fc10bb9586dec030`.
Its manifest binds exact packet, evaluator/runtime and traces; snapshot is
explicitly incomplete. `first-failing-actions.json` and
`partial-failure-exchanges.json` alongside the unpacked evidence contain exact
request messages/tools/controls, actions and feedback. They are protected
assigned diagnosis inputs, never training admission. Final/noise evidence must
be collected after completion. Progress collectors now read growing logs via
bounded≤1MiB tails. Mellum root CPU runs only owner-selected bounded public
source acquisition, with no parsing/model work; raw sources remain unadmitted.

At20:37:51 the properly held LFM relaunch reached its continuation check and
failed before any update: `--continue-from training options differ: heads`.
The predecessor names `/srv/storage/natlang-artifacts/lfm25-qad-clean-head-v1/clean-base-untrained-heads.pt`;
cloud staging names the same SHA19bfa7 artifact under the Pop handoff directory.
The exact base-head content check passed; a second literal-path objective
comparison incorrectly rejected this valid relocation. Pop published the shared
content-guard-preserving fix `fad309e5`. No optimizer or checkpoint was mutated.
The holder retained central traceback, exit1 and outputs-ready, with zero OOM
counters and29,235,720,192byte peak. Prior restart cause remains unproven.
Failed diagnostic output is retained only for this assigned repair at
`/work/production/failed-heads-binding-output-20261014`; proven row caches and
verified parent remain available for a corrected same-sandbox relaunch.

At20:42 the shared scorer reports first task400 pass16finished/5success,
2invalid-action/4repetition/5semantic-mismatch; it still reports incomplete
accounting. Noise repeat1 has14finished/3success,2unresolved. These are partial
investigation results, not a final noise estimate or qualification.

Corrected v2 was released20:46:38 in the same LFM sandbox: ordinary bootstrap
PID10323 under the existing namespace-init collect hold, existing sampler7235,
no cloud reset or duplicate allocation. Exact source63b0cc280c includes fad309e5;
archive61,072,171bytes SHA
`b22ac96feddec9317f172cfd34caef0432ca15febf6e8d29ba2ec543e2877df3`.
Resolved recipe v2 SHA
`4e467b3fd0b159d7704cf28c9b1c351e2d56313e7e73e4506960ddc8cce882a1`
matches all194 staged Python hashes and unchanged parent/PH identities. Full
restore and first updates remain to be observed. The heads-failure evidence
archive is independently SHA-verified on DGX:13,564,590bytes,
`053662d98cace73099392228cee2d0f2a759293613a0f5386d2a3efce6a14f3a`.

Shared scorer accounting explicitly separates complete result export from known
terminal outcomes. First task400 pass5/16 is accounted despite `complete:false`
because all outcomes are known and infrastructure/unresolved counts are zero.
This permits its noise repeat; incomplete result files alone do not invalidate
that success count. Final noise and the matched reference comparison are still
pending.

V2 was stopped with zero PH updates at20:59:40 after static review found a second
concrete shared continuation bug: later-phase restoration copied weights and
optimizer but omitted saved Python/Torch/device RNG restoration. Exact trainer
10422 and zstd child10539 stopped; parent checkpoint remained unchanged and
namespace-init1 retained the root/caches. Shared fix `cca530d20d` restores the
predecessor RNG after optimizer restoration and before dropping its mmap.
V3 recipe published `7290d972ab` pins integrated source
`0f45f447b94cdefbc592c4ca8a48846dba02e5bd`,194 verified Python files and
unchanged parent/head/PH SHA bindings. Actual same-sandbox relaunch21:03:20 uses
bootstrap11935 under the original collect hold. V3 recipe SHA
`8de9be5af065d521c9f610078fb20ca2ec541e7ff5dc83ea7b70560aaaef1fde`;
shared plan-only and quick gate passed. First restore/update remains pending.

Final Mellum task400 first pass5/16, repeat3/16, measured noise2, zero
infrastructure/unresolved outcomes. Owner base-task qualification FAIL versus
inherited13; evaluator eligible/min0 is accounting, not promotion or a
certificate. Actual trainer resumed and reached422 at21:03. Evaluation+repeat
5244.40seconds; first-pass actor_route_replan recorded26,211completion tokens
and25fresh replies over3451.9seconds, repeat logic_proof_verifier26,102tokens
and36fresh replies over1712.4seconds. Both ended as known supervisor repetition
failures without terminal result rows. Parent's narrow matched base control
succeeded on all three tested semantic failures; trained-weight/carried-state
cause remains under investigation, with DGXsm121/cloudsm120 explicitly distinct.
Final evidence archive `task400-evidence-1792011799.tar.gz` is durably
SHA-verified on DGX,37,463,184bytes,
`53e673a25582202fbb31c561f8e11618b967a774053016c66791c63abb795873`.
Its sibling timing, first-action and partial-exchange exports preserve exact
final/noise controls and failure feedback for the assigned diagnosis.

Both full production trainers now perform real updates. LFM completed cold
windows and exact qualified410 weights/optimizer/RNG continuation at21:21;
actual H100 plan uses85,019,590,656byte capacity and switches packed/sequential
execution for current geometry. At93 it processed31,743forward tokens in0.973s,
~31,126supervised tokens/s,34.59GiB peak allocated/70.55GiB reserved andGPU100%.
Its step0 identity file explicitly inherits the previously qualified410 proof;
this is not a fresh claim that trained weights equal base. Resolved LFM cadence
is checkpoint440/eval1000, distinct from Mellum170/400. LFM521 and Mellum682
were observed at21:26 with no runtime exception.

Automatic serialized private full-state exporters now supervise both roles.
LFM440:1,149,832,219bytes, SHA
`0d508a1051685018f4203c03a35b581d04ce920257ea86cdeb8b2ee99e84b231`,
PRIVATE run `werg/natlang/7x3w4dq2`, verified21:27:16; PUT9.92seconds,
independent full SHA readback8.80seconds. Mellum510:46,821,478,103bytes, SHA
`b04223599c3c6c3fbe98801127b8d1f258f47bba44c4e718ba60d0c3fefc58dc`,
PRIVATE run `werg/natlang/ywcj2mvb`, verified21:25:52; PUT338.05seconds and
independent full SHA readback134.55seconds. Exact receipt/state files live in
external dgxops; units `cw-lfm-periodic-private-checkpoint-export` and
`cw-mellum-periodic-private-checkpoint-export` remain active. These qualify
actual training/continuation/export mechanism, not task/model promotion.

Own obsolete zero-update directories and consumed source archives were removed
only after live-FD and manifest checks: failed-heads-binding output,
zero-update-rng output, code.tar.gz and code-refresh.tar.gz under
`/work/production`,181,337,398bytes reclaimed. Active parents, protected inputs,
current runtime, caches and model outputs remain. One DGX failure evidence
archive remains for the unresolved original restart diagnosis. No production
sandbox was stopped; both allocations have ongoing production consumers.

Remaining lifecycle work: Mellum recipe still owns PID1 and has no verified
collect hold. Namespace-init ignored default SIGSTOP. Periodic private exports
protect verified snapshots; final recipe exit can remove root/API before final
artifact export. Before stage completion, use a properly held full-state resume
from verified durable state and preserve source-job durability; never assume
post-exit collect works. LFM's ordinary bootstrap11935 runs under retained
namespace-init collect hold1. No third GPU allocation or current Mellum
interruption was made.

Frozen Mellum correctly declares responseKLwarn0.5/stop1.5, but separately
inherited agreementwarn0.97 drives report-only48-probe gates every220updates
when response agreement drops below0.97 even belowKLwarn. This is an exact
utilization finding for a future operational handoff, not a hot package/control
change or explanation of the independently reproduced trained-weight failures.

## Supervision update, 23:35UTC

LFM performed the approved cadence handoff on the same held allocation/root.
After the6000task ended0/16 (1invalid,11repetition,4budget, zero infrastructure
or unresolved), a scoped trainer TERM atomically saved full state6257.
Private independent object readback verified1,149,837,531bytes, SHA
`b3a6f56a6f4f2d78cce82b126fa6a156302030d2a72c7097299efb0a09d9f1b7`.
V4 changes task cadence1000→4000 and one declared jsonio package hash (the
streaming compressed writer addition); CE/drift cadence1000, input/controls,
optimizer/RNG and final evaluation remain exact. The actual trainer emitted
`resumed step6257 code_changed=[common/jsonio.py]` and502backing shards for
483,983merged documents. First resumed updates6258/6259/6260 completed with
losses0.39349/0.79020/0.58115 and15.06/0.42/0.84seconds (first kernel startup
included); by23:33 it reached6408. No OOM/error; the CPU window reconstruction
before restore cost about21minutes despite retained row/prefix indexes.

Mellum task1200 completed4/16:5invalid actions,7semantic mismatches, no
infrastructure/unresolved outcomes,2842.69seconds. It remains below inherited
base13 and unqualified, despite lower CE0.7114 and responseKL0.22565. Actual
training resumed and reached1320. Full state1200 privately verified
46,821,480,919bytes, SHA
`ff8123a320a0e55d44d97eda939db3bb8fee2bd0c8b3e38606fb57a3191810eb`.
Terminal evaluator outputs plus recipe/identity/eval/drift/report/source
metadata survived to DGX in17,038,519compressed bytes, SHA
`cb3228350caab5fc2b5bcb083631a049cb91746fd247e694165049d4ecdc1c12`;
collect bundle75c73cf8b1e91b034568800276b56a05833ec69c63d9f8c3479edf8fc92c8ae6.
The refreshed1200 checkpoint/metadata fetch bundle is
f4c974c73109b718efc1a9200fa6e1276a3a43ce7e67a52447a7ee2a37bc3772.

Raw survival is still in progress:13/16sources,122,799,910,486raw bytes have
private full-object readback and every raw-file SHA validated; D38/D22/D36
remain. No Mellum SDK stop/replacement or thirdGPU allocation occurred.
Search/OWL public source verification and conversion completed; protected
streaming finalization is still running on one spare CPU at~95MBRSS, followed
by its already declared native pipeline. All acquired/converted outputs remain
separate from training admission.

## 2026-10-15 morning: exact held migration and source durability

Mellum's old sandbox `a7858948-0e1a-4c89-bb69-e2cc29cbc243` is completed. The approved ordinary-PID stop at the next atomic checkpoint cut step1700; PID5860 was verified `T` while recipe PID1 remained alive. The full state46,821,482,007bytes, SHA`e751438c58a67640cfae44e657fb430143448eebdb47f6a78559ef2f5d1e7597`, passed private PUT and independent full readback before SDK stop. Terminal task1600 scored4/16 with no infrastructure/unresolved outcomes, below inherited base13; this remains unqualified. All16 acquired raw sources have independent private full-SHA archive verification (311,675,771,864raw bytes). Raw restore is on demand, never acquisition-marker-only reuse.

Replacement `5ffa592c-5f88-4b4f-b3cc-1507ab94a5dc` is the sole Mellum GPU allocation: RTX PRO6000Blackwell96GB, sm120,4CPU/64Gi host, approximately7.2TB free root. Namespace PID1 remains a Bash holder; bootstrap18874, recipe18959, real trainer18967 are ordinary children. Git archive`ec5e1c84ca9d797c1c3078829b2a5b44d5e9be9e`, pinned base16-file manifest, full1700 state and exact assets were verified. Shared194-file package comparison and actual recipe `--handoff --plan-only` passed, then held-v2 released06:10UTC. Real updates advanced past1709 with GPU100%, no errors/OOM; FP8`triton-grouped-experts-v1` sm120/28modules and168MuonSR latents are active. Device capacity101,975,851,008bytes drives the actual packed32768/checkpointed planner. Exporter was restarted on the replacement ID. Task cadence1200, CE cadence400,16task controls and all original bindings remain exact. This is an exact full-state resume, not a new fresh start or a qualification.

The orchestration interruption exposed a manual staging-to-release gap. The controller now waits for explicit public-fetch, full-state-fetch and dependency exit receipts and invokes exact verification plus automatic release on success. Failed staging leaves terminal evidence. Future full-state fetches write an explicit exit receipt. No third GPU was allocated. Two failed tagged backend create attempts (`9270fe6c-0302-45e1-9fb2-cf43859c796b`, `30b4b852-10ff-4893-8822-ed5ed98f0ced`) were observed during accepted-request recovery; both were already terminal, not additional live GPUs.

LFM v4 stopped at15300 on response KL1.51951592 and the recipe remained unqualified. Final private state1,149,839,963bytes SHA`af52732e404cc657e67b77d7067a6dbb6cec41b63e80b1abbe53e810672f6783` survives, as does the held H100 root. Last below-threshold full evaluation15000 had response KL1.33162786; latest private full checkpoint before that evaluation14960 is1,149,839,835bytes SHA`8b6e2d40b0b9851daaa0aba3412489ad8ac53eaeb38ec6eeb71ada2627c3fb58`. Terminal task/probe completion was skipped by the old stop path; the owner has fixed that shared bug separately. A standalone exact15300 forensic evaluation uses the existing restore/task/behaviour implementations, original full rendering and0.6 sampling, with zero optimizer steps and unchanged failed report/certificate. Results remain pending. Terminal metadata, task outputs, drift/eval and train history were collected to DGX using the existing bundle mechanism.

Search/OWL native replay finished164/164shards:40,818cases admitted and658,396/658,396decisions approved. The7,002 long-evidence builder omissions remain required paged-program conversions. This is replay evidence, not registry/training admission. The authorized isolated Dolci/Cognitive derivative began after this CPU phase ended; both public pins totaling1,663,608,164bytes were fully verified before conversion.

The existing private exporter now accepts owner-host inputs via `CW_EXPORT_LOCAL=1`, with sealed expected SHA/byte guards, the same private-project/GCS PUT route and independent complete streaming readback. No account credential is sent to cloud and no duplicate readback file is created. The real `port-record-s3-v1.tar.zst`547,812,322byte archive SHA`8433bba402539a90b2ed70258770d90c8d65d1a67321b129c1e4800d7200f23c` passed PUT197.83s and independent GET/SHA94.26s. Its private object is in `werg/natlang/pj5bbuur`, prefix`bulk/<SHA>/`; receipt is `dgxops/coreweave-e2e-port-record-s3-v1-private-export-receipt-20261015.json`. Archive-only registration does not admit training rows. Clef and sealing-receipt exports are pending. R10 admission has been revoked by its owner; no active cloud binding was changed to it.
