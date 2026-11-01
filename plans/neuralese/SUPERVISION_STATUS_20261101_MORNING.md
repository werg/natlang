## Live supervision update, 2026-11-01 15:03 CET

This update supersedes the earlier instantaneous observations below.

- Actual whole-graph JOLT+ round11 completed from61 through63: 16 rollouts,
  seven correct outcomes, twenty training graph records and two optimizer updates.
  These records do not count as SFT visits. Actual parent is
  `sd-r1/round-000011/learner/checkpoint.pt`, SHA256
  `11a8d12453bd75ffa4130da2fc9ab4bcdc35569163f303a8cee3520989e55585`.
  Controller is at round12/collect; the rolling SFT slot still contains61.
- Actual63 protected development evaluation finished2/4, all four terminal;
  no protocol, repetition, budget or infrastructure failures. This matches61's
  small development score and does not establish general capability lift.
- The corrected307946-row successor passed fresh full CE census, source/held,
  correction, geometry and actual63 saved-frame transition checks. All420 earned
  SFT visits and optimizer/RNG custody survive;9437 additions have zero visits.
  Final controls are `ops/semantic-keyword-filter-audit-1101/qualified-successor-v8/`;
  exact63 preflight is the adjacent `earned63-correction-preflight.json`.
  Root atomically adopted63 into the rolling slot under the controller lock,
  independently hashing both parents and retaining61 as a local fallback.
  The full trainer CPU continuation preflight passed at63 (28GiB peak).
  `lfm3-joint-sft63-307946-1101.service` launched at15:01 with frozen
  `runtime-sft63-307946-28a-1101`, HIGH76GiB ledger and one-update refresh.
  Verifier passed; actual trainer PID3328751 is initializing. Full-model capacity
  qualification and a new optimizer update are not yet established.
- Paid teacher generation is paused: live process/service/timer checks found no
  active generation. The last one-case Luna run exposed an oracle contradiction
  and supplied no admitted positive data; raw evidence is preserved. Repair and
  verify public task contracts before spending more requests. Worker count is a
  ceiling; measure admitted useful yield against requests and tokens.
- Independent61 external full-state export and full readback are verified in
  `ops/export-earned61-external-1101.json`. Latest63 export is still being checked;
  an active offloader alone does not prove independently recoverable63.
- NVMe has62GiB free; external volume273GiB free. Keep the40GiB internal floor.
  Three source converters and the offloader are running; none remains frozen.
  The learner is initializing; no new optimizer advancement is claimed yet.
  The weekly read-only storage scan had run11 hours and was competing for HDD
  metadata I/O. Root stopped it at15:02; PID2780081 is gone. No data was deleted;
  active storage-floor supervision and checkpoint exports remain in place.
- A targeted stop test injected its checkpoint stop into the initial zero-update
  recovery save. It now targets an earned checkpoint explicitly; all four stop
  scenarios pass. Broader check-main failures remain under investigation.

## Earlier live supervision update, 2026-11-01 13:02 CET

This section supersedes the instantaneous observations below.

- Joint SFT completed four actual updates57→61, retaining optimizer/RNG and
  earned exposure. Full checkpoint SHA256
  `610de853e0578e240c1c085b8861468b826a767de3fd81ab62775b67a1d0abd9`;
  rolling slot `sd-r1/learner/checkpoint.pt`. There are420 earned graph visits
  against982749 required; comprehensive SFT is far from complete.
- Step61 protected development evaluation:2/4 exact successes, all four terminal
  and accounted;57 had1/4. The file-effects task improved in this observation.
  This small comparison is encouraging, not demonstrated general capability lift.
- Actual61 controller publication succeeded through main621ab7eca, with archived
  report/receipt and independently recovered57 metadata. The trainer retains53
  as its objective ancestor; authenticated publication57 proves that identity
  without confusing it with the immediate57→61 parent/update delta. Exact
  evidence: `ops/publication61-1101/publication61.json`.
- Real next on-policy round11 launched as
  `lfm3-onpolicy61-307946-v2-1101.service`, frozen
  `runtime-next61-1de6db-1101`, HIGH76GB ledger, one policy round. Actor weights
  loaded; full-prompt collection is producing fresh actual61 model replies.
  The first launch refused an imported checkout helper before any rollout;
  v2 uses the frozen runtime as actual ledger workdir and exact session custody
  now succeeds. Launch/config is
  `ops/next61-onpolicy-307946-1101/config.json`. A launch is not an RL update.
  The stale root `sd-r1/config.json` remains unsuitable for resumption.
- Prepared and bound successor recipe contains307946 graphs: actual298509
  predecessor plus9436 verified Big2 graphs and one accepted/masked CoEdIT graph.
  Train/foundation roles move together; held source stays unchanged. Unearned
  intermediate307945 is coalesced rather than claimed as training history.
  Exact controls: `ops/coalesced307946-next61-v2-1101/`. No new SFT exposure yet.
- Exact shared-producer decision factor0.01 reduces its measured effective
  objective mass from roughly31% to0.962%; producer gradients, all consumer
  targets and complete graph topology remain. No blanket communication loss cut.
- New norm backward is qualified on actual61: reference/candidate losses and
  all288 gradients bit-exact, with the optimized branch actually exercised.
  `ops/qualify61-eab-plain-fullstate-v5-1101.json`. Selection is shape-specific;
  broad unmeasured throughput claims would be wrong. The complete capacity
  geometry audit also disproved the earlier claimed five-to-one probe reduction.
- Independent stock-HF/native attention investigation isolated actual BF16
  backend output rounding with identical Q/K/V/RoPE/masks. Both are similarly
  close to the no-TF32 FP32 reference. No mask/RoPE bug or task-level harmlessness
  is established. See `LFM3_HF_NATIVE_REFERENCE_20261101.md`.
- Pop matched52/57 evaluations each observed two successes, one mismatch and
  one infrastructure-wall unresolved case. Both are ineligible for a full score;
  unresolved is not a model failure. Fresh zero-exposure instruction-baseline
  evaluation is now actually running on Pop. Future matched completion needs
  a larger nonbinding wall allowance, preserving original observations. A new
  case0-only rerun for0/52/57 with7200s allowance and unchanged count limits is
  queued behind the baseline via Pop's actual ledger waiter
  `natlang-matched-case0-longwall-1101-waiter.service`; no duplicate actor runs.
- Four reviewed medium-Luna NLIv16 workflow workers actually ran; slots01/03
  completed outputs and02/04 were still live at the last check. Full-graph
  quality and actual Sol judgment must be checked before positive admission.
  Campaign `contractnli-v16-scoped-sol-v2-20261101` on the external drive.
- Earned61 external independent export is live, approximately6.6/15.3GB copied
  at13:00. It is not yet verified recovery. Earned53 external and57 Pop copies
  retain their existing verified status. New offload owner16d722bd7 protects one
  in-flight copy and coalesces one newest pending copy independently of learner
  service exit; intermediate join/cancel implementations are superseded.
- Internal free space88GiB, external307GiB at this observation. Preserve source,
  held inputs, actual recovery dependencies and live checkpoint reservations.

## Live supervision update, 2026-11-01 11:24 CET

This section supersedes older instantaneous process, source and storage counts below.

- Production joint SFT actually advanced from earned57 to58 on the complete
  298,509-row source; the next update is running. Unit
  `lfm3-joint-sft57-full298509-1101.service`, frozen
  `runtime-sft-multiupdate-v2-1101`, rolling `sd-r1/learner/checkpoint.pt`.
  First update: 376,174 physical / 1,191,427 logical / 36,193 supervised tokens,
  833.52 seconds including an exceptionally heavy profiler. Do not use that
  observation as an unprofiled throughput baseline or capability lift.
- Earned57 full checkpoint `b3be5ac5...c8f056da` has completed independent Pop
  full-file readback. Exact receipt: `ops/current-sft-peer-export-1101.json`.
  Earned53 also has verified external full-state recovery. Runtime/source closure
  and restore qualification remain separate from these full-state copies.
- The latest57 task evaluation was 1/4 exact successes. It is not a matched
  53-versus57 comparison and does not prove improvement or general regression.
- Big2 restored data has substantial producer repetition: 93,229 graphs share
  one327-token producer. The initial effective objective census places roughly
  31% of total loss mass on that program. A is implementing an exact decision
  factor of0.01 plus measured family normalization/cap, preserving all consumer
  targets and whole graphs. It must be adopted as an explicit objective
  continuation as soon as sealed, with historical exposure preserved.
- Another9,436 authenticated Big2 producer/consumer edges passed provenance,
  graph-content duplicate, current-protection and actual loader checks; sealed
  `ops/big2-restoration-v2/qualified-delta-v1/receipt.json`. A is combining them
  with the objective correction rather than adding repeated producer CE unchanged.
- Profiling produced an11.265GB/38.8M-event Chrome trace. Terra extracted compact
  complete summaries and retired the trace with a hash/receipt, leaving a tiny
  sentinel to prevent the older running profiler exporting another trace.
  Committed5324adfe1 replaces new profiling with bounded phase timings.
  Rowwise norm backward dominates that instrumented trace. The first batched
  candidate has exact CUDA input/gain gradients but is slower in an initial
  diagnostic; it is NOT promoted. Real production-path warmed timing is underway.
  Temporary learner pauses preserve its in-flight state and always thaw in finally.
- Streaming-compressed replay runtime `runtime-port-stream-a8b-1101` is sealed;
  predecessor receipt completed and the new conversion unit
  `lfm3-cov-port-s4-stream-a8b-1101.service` is active. Driver authenticates and
  reuses completed historical shards; journals now stream directly to zstd.
  Check real new replay/output growth, not just active state. Exact launch and
  predecessor bindings: `ops/port-stream-controls-a8b-1101/`.
- Three reviewed medium-Luna CoEdIT workflows actually launched: single-document
  revision, atomic multi-document queue, concurrent-change recovery. C supervises
  whole-graph quality, Sol judgment and static conversion. One completed revision
  has task success but redundant exploratory calls; mask exact bad targets before
  admission. A failed batch is not positive data. Scripted references are not
  sampled teacher success. No pinned application files are being edited live.
- Next actual-current-policy whole-graph JOLT+ config has been prepared at
  `ops/next-onpolicy-298509-1101/config.json`; model-free resume preflight passed
  (unbound next round). It is NOT launched while SFT owns the GPU. Publish the
  genuine earned SFT boundary, rebind any newly ready objective/runtime, run final
  frozen preflight, then collect from that exact checkpoint with full prompt,
  complete schemas and sampled probabilities. Retain all unfinished SFT obligations.
  Do not accidentally resume stale `sd-r1/config.json` (old runtime/source/engine).
- MegaToken run state is absent on both drives. Additional inactive derived
  materializer caches/bytecode removed1,995,886,592 allocated bytes; receipts saved
  in `ops/recovery-cache-cleanup-1793528040757592774.json`. Unique data and actual
  recovery dependencies remain. Internal free space fluctuates with legitimate
  rolling-checkpoint reservations; keep40GiB floor.

# Supervision status — 2026-11-01 morning

Latest observation: 09:47 CET. Check live processes before acting. The goal is
still capable Natlang/Neuralese models, not completion of this document.

## Live continuation and immediate work

`lfm3-joint-sft53-a9cf-1101.service` is actually running the shared production
controller and native SFT trainer from frozen `runtime-a9cf65487`. Its parent
is earned SD-R1 step53 below; output is the single new `sd-r1/learner` slot,
never the immutable `round-000010/learner` parent. The real CPU continuation
preflight passed, preserving model/optimizer/RNG and 11 RL rounds/244 graph
visits. Main GPU capacity and six additional oversize cases passed. The first
optimizer update started at09:45; no new committed step is claimed yet.

Actual controls: `ops/joint-sft53-a9cf-20261101/config.json`, SHA
`cd279cc2056c9e599ca2d7eb0c0edc84839afd03cefcaae062486a7a8106dc7b`.
Recipe SHA `6ec5b245d3c959f399e6f3feae8891fb3191cc30af7d7525a6d99cb721c9fc3d`.
The current190154-row source is SHA
`d8185760a238cb59247b0fb5cf9db6ab800776be3f0699cbbc7b1ffbb8fe10ff`:
the quality-corrected190153 rows plus the verified complete TATQA case0 graph.
Its equality in row count with the old190154 recipe does not mean equal content.
One full source pass and seven communication visits remain required (224264
visits), with an initial four-update refresh and task-behavior check. Four
updates are an early adaptation window, not completion of this recipe. Continue
the evolving recipe promptly after inspecting earned state and behavior.

The stale unearned591181-row stage was explicitly replaced, with zero exposure
credited and its source/history preserved; see the actual parent53-bound proof
in the controls directory. This does not excuse useful data omissions. A found
108353-row Big2 communication omission has a content/protection-filtered candidate;
Native parser validation, current-source dedup, CE and geometry staging are still
in progress. Adopt the qualified successor against the actual earned current
source and checkpoint; do not reuse the zero-exposure stage replacement proof.

Both reviewed v18 Luna tasks completed successfully (sales30 turns, rate19).
Whole-graph quality review found a redundant classification action/subtree in
rate; preserve its raw history and mask the unwanted targets with bound evidence.
Sales capture failed before encoding due to an inconsistent separate-materializer
runtime binding. The builder now rejects unsupported bindings at build time
(`261e7044b`); fresh audits must pass the actual positive-row loader before GPU
capture. The failed Pop container was explicitly stopped and its lease released;
stopping a systemd wrapper around detached Docker does not itself stop Docker.

Port conversion falsely committed five completion summaries with a missing
compressed stream. Their markers were withdrawn with exact evidence under
`ops/port-invalid-summaries-1101`. Publication now checks both streams, propagates
compression failures and takes an exclusive owner lock; four regression tests
passed. Six existing SWE replays are draining with only scheduling stopped.
`port-s4-frozen-handoff-v2-1101` owns their automatic corrected handoff. It binds
actual frozen runtime/script/dependency contents and predecessor IR equality;
the old mutable `dist/.build-commit` stamp was stale despite rebuilt contents.
Verify actual successor launch rather than assuming the watcher has completed.

## Earned model and numerical repair

Clean SD-R152 completed one real JOLT+ update to global53 from ten authenticated
whole episodes (eight students and two privileged attempts). Actual update:
83.58 seconds,243234 logical forward tokens /78670 packed physical tokens,
12560 supervised tokens, about2910 logical forward tokens/sec. Student OPSD and
teacher KL terms were nonzero; outcome terms were zero. This is actual learning,
not demonstrated capability lift or native-writer readiness. Full source SFT
obligations remain unearned.

Saved full53: `/home/werg/data/lfm3-rl-dgx-20261029/sd-r1/round-000010/learner/checkpoint.pt`,
SHA `1fdbacfb2cfd55cde8d6695bb666d820b41682d5c4d3299872512c267499ff77`,
15330800291 bytes. Controller is collect/round11/trained11. The unit
`lfm3-onpolicy52-amended-jolt-9520-1101` finished normally, before an attempted
operator signal was sent. No stop signal reached it. Its finite state is saved;
its original native numerical certificate failed. Do not treat that certificate
as passed or automatically continue Native SFT on the old arithmetic.

Exact no-SGD replay reproduced three gradient failures: layer0 convolution
in-projection, input embeddings and the rank8 interface adapter up-weight.
Forward/incremental/consumer CE differences were zero. The [2048,8] parameter
is the adapter, not a convolution kernel. Layer-local captures showed the first
difference in FP32 convolution-history adjoint fan-in on imported producer rows;
a2.98e-8 difference could change later BF16 rounding. A resident A/B proved the
invocation-split FP32 gather correction passes the complete53 gradient law with
unchanged forward values and unchanged tolerances. Commit `a9cf65487` implements
that shared correction, retaining one packed model forward. Five focused tests
passed, including rejection of old arithmetic and a disconnected-import control.
A16384-token single-convolution forward/backward timing was0.0809s old /0.0843s
new (three measurements; this is not end-to-end throughput). Actual frozen
runtime qualification subsequently passed all289 comparisons with saved step53
optimizer tensors and RNG unchanged. Receipt `ops/qualify53-a9cf-fullstate-1101.json`,
SHA `b33c985fc2fe6f663d45cae367431c97e38348fc5e2e0a4d99f143f4a31f4930`.
This no-SGD certificate is separate from earned optimizer work and behavior.

Step52 remains the independently recovered fallback. Branch74 remains diagnostic;
no SFT53–74 exposure from that branch was credited to this new53.

## Actual source and scheduling repairs

The clean52 collection accidentally reused the old materialized ToolACE dataset,
although its converter and1500-case qualified successor were already ready. The
actual task exposed `sequences:any` but expected hidden object keys. A live-read
learner/seed guard stopped that invalid group before any update. Another group
had three successes and one real Native export infrastructure failure, not a
semantic negative. Original20 student/privileged rows are preserved.

Commit `9520ad202` supports authenticated completed-collection withdrawals and
rejects stale ToolACE schema contracts at admission. Actual proof
`ops/toolace-round10-withdrawal-v3/proof.json` SHA
`1e3a0418c1a82c8332be763e7ce347956a57382085e3908727cbaa320cd7be20`
retained only groups0/3 (ten rows), preserving their privileged contexts. The
zero-update transition was actually applied before53 training. Runtime9520 ran
this one round only; its9b actor/Node bindings authenticate historical recording
and must not be used for a new collection. New requests use the corrected runtime.

Withdrawn groups still polluted selector history. Commit `b957280ce` prevents
recurrence; root applied the reviewed exact round10/policy52 cleanup under the
controller lock, retaining old observations in an immutable correction record.
No weights, optimizer, exposure or counters changed. Receipt:
`ops/onpolicy52-current-20261101/selector-cleanup-applied-1101.json`.

The proposed two-graph SFT focus was NOT approved. Semantic review found a bad
qwen15 graph and invalid missing-until targets in qwen36. The combined candidate
at `ops/qwen15-qwen36-successor-v1/receipt.json` (SHA
`28f0e55e97963325ba128e29c97767af9cdf41aed3b63d0df2e29cb1a89ae8d4`)
withdraws qwen15's complete25-node graph and masks only763 confirmed bad action
targets in qwen36, preserving all12 nodes and producer/consumer edge. Current
candidate source is190153 rows, SHA
`b6ec0bbd4026afcf7e25f845b33af6e2e51de004f00f96ff4187e21880a3f7d2`.
CE/family map/geometry/recipe are bound in that receipt; actual `load_recipe`
passed under4GB LOW. Its successor plus TATQA is now selected by the live SFT
controls above. The stale LogHub focus was replaced by the good TATQA row.
Reusable quality tools/tests landed
in `83edad9eb`.

IMPORTANT: original unearned SFT plan is591181 rows (519 leaves), not190153.
Only36158 old ID+binding pairs match the current candidate exactly;555023 old
IDs are absent and153995 candidate IDs are new. This does not establish data
loss: transformation/withdrawal/variant lineage must be audited. Do not silently
mark the old debt fulfilled. `/root/luna_diverse_codebases_a` is running
`sft-obligation-map-remaining-v1` (LOW5GB), reusing/writing source-bound indexes
and preparing exact source-group/lineage/protection evidence. Shared groups alone
are not quota fulfillment. Original checkpoint52 has zero SFT visits; no old
exposure is being erased.

## Structured teacher text now reaches actual Native capture

The teacher-text readout fix is real nested typed model execution. Later fixes
support historical typed-JSON eval-finish writers through separately bound
current conversion code, without relabeling the original Luna/ac519 actor.
Materializer, Python corpus and converter preserve the original sampled producer
eval target; host-encoded JSON is context-only, not fabricated writer-body CE.
Strict source/writer/body/raw-response/protection guards and focused tests passed.

Repeated loader validation exposed three real throughput defects: global hash-log
churn, complete runtime/bound-source rescans, and repeated protected-world loads.
Shared verified caches now invalidate file/dir/symlink and nested protection
changes. A cold audit load reads about1.49GB once; subsequent loads read about
177KB, with no content rehashes. Source pins include transitive custody/hash/IO
helpers. Do not edit those pinned bytes during an active capture without stopping
and rebinding. Earlier audits v7–v11 are historical after necessary fixes.

Current stable audit:
`/mnt/external/natlang-luna-campaigns/tatqa-v16-case0-typed-json-conversion-v12-current/native-conversion/ingestion-audit.json`,
SHA `5ff0957f5bf9a9e5ebbae6b9285af004d7f956789272d6944ac95b1685b563db`.
Actual capture unit `capture-tatqa-case0-native-v12-pop52-1101` uses remote
`pop-native52-capture-json-v2-1101` (10GB host,7.5GB discrete GPU,1GB reserve),
with source encoder checkpoint52/runtime9b. The teacher remains Luna/ac519.
Output: `/mnt/external/natlang-luna-campaigns/tatqa-v16-case0-native-capture52-v2`.
The capture completed all19 turns, and NativeGraphSource verified the assembled
20-node graph with six authenticated consumer imports, original producer-action
CE and zero host-encoder CE. Graph SHA
`1df930a0c598b2871f34b4161f6824925068820105fd5c4e8d9a55fa87f593e3`.
Actual reader states exist as5.6MB safetensors, source positions[1,37), with
exact writer/typed-body receipts. Sampled `native_writes=0` counters exclude
these separate host encoder bindings; do not infer a missing source from that
counter alone. The current recipe binds the separate original Luna teacher and
student52 encoder through corrected role receipts. Verify actual exposure in the
next earned checkpoint. Prior v1 failed after a source-helper change;
its partial capture remains diagnostic and its actor/lease were actually closed.

ContractNLI final2 generation was actually launched on the correct source after
rejecting stale final/final4 plans. All three Luna episodes completed but Sol
judged them false; no positive admission. A drain marker left the fourth row
unclaimed so its new requests can use fixed seed/custody/runtime code. TATQA18
sales/rate references passed while movement and unsupported-USD references
were excluded for semantic repair. The two sound cases were launched with
medium reasoning and full prompts and both completed. Current v20 source work
also exposed a false alias conflict: comma/semicolon record syntax was compared
as raw text. `d744d64a1` compares parsed canonical types; four focused tests passed.
New teacher freezes use the fix; historical actors retain their actual bindings.
Use actual source/child evidence and repair feedback; do not count broken task
contracts as model negatives. Up to six Luna generation workers are authorized.

## Recovery, storage and active supervision

Step53 independent file export is active as `export-earned53-independent-1101`,
PID589782 at last check. Destination:
`/mnt/external/natlang-independent-recovery-20261101/jolt53/1fdbacfb2cfd55cde8d6695bb666d820b41682d5c4d3299872512c267499ff77/`.
It has copied about15GB and is still finalizing, not a verified recovery. Preserve its source until
complete independent subprocess readback. The complete runtime/source53 draft
`ops/recovery-onpolicy53-v2-draft.json` SHA
`5a06d6104f79e626c6e9eda7156c2e5c329a152c5e2a99cb6fe5f35317b74456`
binds730 entries and separately retains future SFT obligations. No strong queue
activation or complete53 restore qualification is claimed.

The supplemental diagnostic69 restore uses NVMe for tiny runtime/Python/assets
and external hardlinks for source objects/checkpoint. Its bounded pause for53
backup ended and the exact invocation thawed. Root's three-converter180-second
window `checkpoint-external-write-window4-1101` ended and all three converters
were directly verified thawed. Diagnostic-only pause gave <1MB/s for53; pausing cold converters improved
it to roughly14MB/s, identifying disk contention. Internal NVMe was mostly idle;
the external HDD had about1s read latency. Do not call this an internal SSD fault.

MegaToken runs are deleted on both drives. Disposable recovery staging/caches and
merged worktrees were removed; unique work/recovery refs remain externally.
Internal free space was89GiB, external339GiB before ongoing exports.40GiB internal
floor remains enforced;52/53 remain protected in their original SD-R1 slots and
new SFT writes only to the separate rolling slot. Actual external I/O priorities
now favor the earned53 export over cold converters/diagnostic69, without freezing
them. Measured efficacy is separate from a successful ionice setting.
Pop SSH briefly timed out, then was actually reachable again; never infer terminal
failure from one timeout. CoreWeave monitor's latest request was not entitled;
its stale-launch hold remains.

Root supervises operations. Existing helpers: Luna relational application (C)
finishes Native teacher data, Luna diverse_codebases_a audits source obligations,
Terra joint_sft_v4_execution fixes numerical law and recovery. Do not spawn more
strands by default. Targeted tests passed; broad check-main is not fully green.
Continue actual source admission, corrected-runtime qualification and comprehensive
joint SFT/JOLT+, then program adaptation/metalearner when the backbone is suitable.
