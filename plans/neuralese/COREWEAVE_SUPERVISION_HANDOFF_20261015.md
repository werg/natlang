# CoreWeave supervision — 2026-10-16 05:58 UTC

## Current state; supersedes the earlier observations below

- LF checkpoint5224 SHA4459907ba91fa45dda65b09d5370c5b6820dd649b3dfb57c932a35d3076d5976
  is the current durable parent. Checkpointed CPU-input-offload recovery made six
  updates, then hit the actual64GiB host cgroup limit; those uncommitted updates
  must replay. No-offload recovery ACTUALLY reached5233 at05:52, GPU100%/~37.7GB.
  Keep checkpoint_layers=true/inputoffload=false; no-checkpoint max graph OOMed.
- M safely saved5385, SHA0d467f4c3249817e1b39795e24329fad3929f8b0c73b7136514ccd42052cea18,
  48,041,003,941bytes,811971boundrows/16906completedvisits. Its attempted rollback
  failed before SGD because continuation metadata still named V7 although the
  checkpoint already includes V8. Cloud is binding the actual saved V8 source
  as predecessor and restoring5385, not5000. cgroup oom_kill=0. Future safe stops
  signal the recipe parent so it can forward and record interruption properly.
- Shared optimizer fixcbcbb223b independently qualified67root tests+quick;
  f1630b52d telemetry68root tests+quick. Native ordinary sparse rows/adapter now
  AdamW, backbone MuonSR unchanged. Explicit native-ordinary-to-adamw-new-history-v1
  migration preserves exact original Muon buffers/groups as provenance, weights,
  RNG/exposure/unaffected state, and honestly starts new Adam moments/age atzero.
  Both cloud compositions passed45focused+quick; next recovery deploys this fix.
  Initial declared NN input/output rates.0003 forBOTH; LFreasoning.0003,
  Mreasoning/backbone/adapter3e-6. These are adjustable operational choices, not
  demonstrated optimal rates. Both future recipes noise1/native transporttrue.
- M5001–5250 never had an ordinary-only streak longer thanone; that hypothesis
  is ruled out for this interval. Actual FP32 updates occur, including inherited
  Muon momentum on zero gradients. Routing correctness is not a task-lift claim.
- Corrected native LF5000 task unit remains active. Five early completed cases
  had semantic failures despite actual native writes (three also actual reads);
  two returned a state reference as plain text. Native storage/import evidence
  does not establish successful control read-out or downstream quality.
- M24,556,281,665byte inference-only view fetch is advancing (~13GiB by05:57).
  Shared resume verifier passed24root checks; private future-retry disk guard now
  counts retained pending bytes. Do not restart the active useful fetch. Final
  full SHA/base inventory/seal/queue admission still required; no M DGX run yet.
- V13 ACTUALLY admitted1156train/model in d70605681; staging is separate from
  exposure. Local D12 both-port conversion has ten complete pairs; source agent
  is preparing a verified first complete cohort admission, not waiting D16/D18.
- Search final-output forecast exceeds root capacity alongside the M ferry.
  Data agent owns incremental SHA/size-verified relocation of the immutable128file
  selection (11,587,363,760bytes) to external storage, retaining original paths
  with per-file symlinks. Never remove source bytes before verifying their copy.
- Standing fullf9c1b00d2e FAILED one real process-start identity race:2113Python
  passes,15CUDA/2010TS/22conformance passes. Root reproduced25/100 transient
  identity misses; exact retry repair passed11focused tests. Next standing full
  must verify the integrated fix. Separate CPU fullcheck OOMed12GiB; focused16GB
  CPU learning-file investigation progresses, not another broad rerun.
- RootcoordACK05:53, nextdue06:43. Existing source workers, Qwen teacher queue,
  private ferries/evaluators, storage and both learners require fresh checks.
  Whole-foundation completion and useful cross-model readiness remain unclaimed.

The following detail records earlier measurements and dependencies. Where a
runtime, checkpoint, rate, queue count or check status conflicts, use the current
state above and the live bound receipts, rather than these historical samples.

DGX owns all execution; Pop closed. Exactly two student sandboxes and one
self-hosted Qwen3.8-Flash-Next NVFP4 teacher. Comprehensive evolving production
training remains the objective. Neither original 26,312-graph communication
foundation is complete; expanded 57,144 tracked separately. No trained joint
semantic dialect, production RL, or task-lift claim.

## Training and immediate handoffs

- Mellum restored exact full step5000 parent f966fca5dd68d07e0a7cf435e1403df411280f6fbdc989d129f554797ea48f9e
  (47,975,737,061 bytes), preserving optimizer/RNG/exposure. Runtimef521a6163116,
  recipe335617/native335628. Actual5045 at05:00, ~5296supervisedTPS. Four role
  input/output rows now .003; backbone/reasoning3e-6. Actual FP32 parameter
  updates verified, not gradients alone. V8 resident adoption atstep5019 in
  SAME trainer:811971train/15827held. Adoption is not per-new-row exposure.
  Big2 and Search actual-parent capacity checks PASS, including112008physical
  and173589totalKV limits. Big2+9972held resident preparation continues, then
  Search/V9–V12 and incrementally admitted Search shards; no47GBreload.
- LF actual5121 at04:57, ~4601supervisedTPS/GPU99–100%. OldV7 runtimef030c90,
  recipe/native471131/471368;804180train/15827held. Next resident handoff has
  convolution-history index reuse (43CPU/CUDA exact-output/all-gradient tests),
  one routine held pass, per-boundary diagnostics and explicit native transport.
  Original foundation~8582/26312 at5088; Mellum~7156/26312 at5012. Refresh exact
  fixed counts from current per-row credits before readiness claims.
- LF prewarm strict source/maxima scan reached78k/~105k at05:03; it is CPU
  work, not a stalled learner. Guardian521234 paused until checks finish.
  Root approved using SAME mandatory actual-parent maximum cases to qualify
  checkpoint_layers=false against checkpointed bitwise loss/all-gradients;
  explicit exact-parent checkpointed fallback on numerical/capacity failure.
  Shared helper9c3dbcd14 passed22focused+quick, cloud composing it. No activation
  or throughput improvement claim until actual child proof/SGD.
- Readiness/controller waits genuine completed original communication exposure.
  Bind actual resident recipe/current.json at each adoption. VAE/joint/RL
  implementation is integrated but must not run ahead of foundation readiness.

## Task evaluation and a confirmed transport configuration gap

M5000 first complete16-task pass:10successes. Redundant second pass was stopped;
its partial results remain separate. Production recipes now declare noise1.
Current M child inherited olddefault2; next useful policy continuation must set1
before10000 without reloading solely for this metadata change.

Actual LF5000 responses include native_state multipart blocks, but the evaluator
had neuralese_server=false: collector.ts then uses the plain client, omitting
native store/value runtime. Those historical results cannot establish functioning
native transport. Rootf9c1b00d2 requires native-aware transport for live native
TaskEval and new native async seals; shared productionbase+LFspec explicitlytrue.
Generic legacy defaults/historical sealed requests remain unchanged. Independent
cloud review clear;71actual async/SFT/HTTP/Node both-port/receipt tests+quick PASS.
Model-facing measurement debt: inspect48actual model turns per student in the
next corrected held window, including stored/imported blocks and task outcomes.
M old10success trajectories contain no nativeparts so far; partial audit ongoing.

LF4750 historical result unscored:0success,6semantic,3budget,2model-incomplete,
5unresolved; no task-quality conclusion. LF5000 plain-client task unit stopped
05:17 for the demonstrated transport mismatch; original request/partial evidence
retained. Root rearming SAME verified5000 weights with explicit corrected sampler
and immutable runtime into native-lfm-async-eval-native-transport-20261016; rearm completed05:18:19. SINGLE dispatcher restarted05:19:16 on the new outpath;
actual corrected task unit taskeval-dbb7eb81-0005000 active. No old sampler edit.

M inference-view derivation ACTUALLY passed exact bitwise verification:24,556,281,665
bytes SHAe897b6854839735c09ea3324e3e6bf0e48c3155ab49d5f0b9da9aaab4b40b6b2.
Only nested training.optimizer removed; full training parent untouched, origin
separately bound. Peakhost~4.25GB, no optimizer update. Private PUT + independent fullGET ACTUALLY verified (fullGET78s). DGX single-stream
ferry started05:19:56; actual~5MB/s. A40MB ranged-transfer probe confirmed206
exact Content-Range but four streams share the bandwidth, so no parallel fetch
was adopted. Root492e8626e adds verified pending-prefix Range resume for future
interruptions without stopping the current download; focused7+quick passed,
root independent7+async checks running. Full ferry still in progress, not enqueued. Root4GB ingestion-only service waits on
coreweave-native-mellum-inference-export-state-20261016.json, uses sharedpoll_latest
and existing SINGLE dispatcher queue. Private Mspec mirrors actualcloud16case
indices/workers2/quotas, native transporttrue+noise1 explicit. Localbaseinventory
must match; rootdisk~29GB, require actualviewbytes+3GB before singlecopy/hardlink.

## Urgent sparse vocabulary optimizer repair

M5250 new held communication20graphs: produceropen mean9e-19/NLL43.52,
producerclose4.9e-16/NLL37.45, contextconsumeopen8.1e-17/NLL42.50 despite
thinkingopen.918/close.9998 and aggregateCE.369. Different5000cohort means no
matched temporal claim; currentabsolute control boundary failure is real.
Root and independent reviewer BOTH found doubleclassification: latent_split
sends minside<16 matrices to ordinary/AdamW per its contract, but PortMuonAdamW
reclassifies4xwidthrole/2xwidththinkingrows into vanilla torchMuon(minside>1,
match_rms_adamw). Thus .003 interface rate used shape-scaled orthogonal updates,
not the intended AdamW. Actual FP32 row updates prove it is not frozen rows.
Reviewer owns shared eligibility fix + honest explicit named-optimizer migration;
unknown historic Adamsecondmoments must not be fabricated or silent-reset.
Cloud directed to safely save current M progress and continue SAME exact state
with legacyMuonkind retained but interface rate rolled back to3e-6 pendingrepair;
fold noise1+nativeTrue in this useful continuation, never reset to5000. LFpending
handoff keeps existing kind until migration qualified; assess rate with evidence.
Root/cloud inspect target weighting, batchcomposition and gradients independently;
ordinary-only negative pressure is a hypothesis, not a proven behavioral cause.

## Teacher and evolving admitted data

Qwen/PennyRoyal3.0.1/SGLangSM120 dedicated teacher:04:50 batch12 3559/3826terminal,
3125success,8active,259notstarted. Fresh512cases/128worlds/8domains were qualified
512/512 by actual isolated reference runtime; authored inputs are not teacher
training outputs. Atomic next-source switch04:37, NEXT1024trajectories use fresh
packet (two variants), currentbatch continues. No new sandbox/teacher allocation.
V11 admitted1347train/model; V12 ACTUALLY admitted2641train/model and stagedboth
(41797ee71). V13 harvest70successes127.8MB is now materializing with all V1–V12
exact-dedup inputs; noV13admission claim. Exact duplicate/action omissions
and protection proofs remain required. V8 actualM adoption; V9=3154,V10=2028,V11
and Search0000/1/2 admitted but no new-row training claim until actual exposure.
Searchshards0000=5177PH+1696whole,0001=5340+1728,0002=5033+1653 permodel,0held.

Search fullselection164/164files658396rows,4workers/32GB live; outputs0003–0005
advancing. SQLite is per-active-shard TemporaryDirectory: observed completed M0003
6.5GB removed after durable outputcommit. Scratch is bounded concurrent work,
not monotonic retained DB accumulation. External37–42GBfree; root29–30GBfree.
Do not migrate huge scratch onto root while Mellum view needs24.6GB.

## Source conversion and remaining work

All20new acquisitions terminal:16acquired, D09/D20/D24/D39 actually403GatedRepoError
with existing authenticated credential, not license holds. D04 timing/hashmetadata
has no textSFT. D34Dolci existing work not duplicated. Big2cloud5800chains/
133938edges/131443ordinary permodel, Dolci381/644 at04:50; refresh actualworkers.

Shared adapters now cover all acquired structured sources: D22/D25/D35/D17,
D26/D27/D28/D38/D23/D36/D37, D13/D14/D32/D19. Completecontexts, actualreasoning,
failedruns, exactneighbor/evalSHA and benchmarkgroups retained. D14fullpreferred
plusagent3/4fallback, D32Harmony blocks, D19exactusage joins; no inventedoracle.
Rootc52e97c80 adds D12safeTAR/fullstream-final views, D15releasedtwoactorviews,
D16/D18completeobserved-log/source-label diagnosis supervision.33rootfocused+
quick PASS; sourceagent57focused+quick. Localnormalized export COMPLETED9624sessions/74MB under immutable660b662e8.
Prepared-tokenizer shared seam7fdee9785 passed39rootchecks45.74s+quick, including
actualbothHFtwo-sourcefresh/reusebyte+receiptparity; explicitsource/expanded
fingerprints/inventory/rendererconfiguration and every-invocation lease. Frozen
3259 native bulk CPUworker active6GB; firstD12/LFarchive652sessions→362train
graphs+290explicitnonpositiveomissions, Mellumactive. Globalheldclosure16,455
groups includes322new;83D18heldrows protect98relatedD16sessions before rendering.
Those source data are not yet admitted/exposed. Three genuinely truncatedD12
streams and empty/error artifacts have explicitomissions, no fake completeness.

Fullnormalizers D22/D35 and D17/D25 lastobservedactive/completing; rootmustrefresh
terminalreceipts. Additional five-source normalized worker is STAGED onLF under
/work/cpu-runtime/acquired-full-v2-18e8beb21, jobacquired-full-v2-root-lfm-20261016;
cloudownslaunchwhen CPUslotfree so training/prewarm not starved.

Root ACTUALLY replaced the sole old all-or-nothing native converter at05:12:
exactPIDs503139/503143/503145 checked/stopped, uncommitted74,294,690B inventoried
beforedelete; normalizedsources untouched. Newrunner525307/driver525311 on
/work/cpu-runtime/full-session-native-82bc5b377, jobacquired-full-native-durable-root-lfm-20261016.
Exactreasoningcoverage avoids redundant Mellum prefixes, keeps all missing thoughts.
100source-record fsynced shards, completefileheldclosure, SHA/runtime/holdbinding,
singlewriter lease, verifiedresume and cleanupjournals.20focused+10cleanup+quick
PASS. Actual05:24 durablefrontier100source records committed in one shard;
sourceworker alive despite a transient missing-/proc driver query. Full per-file
completion/admission not yet claimed; inspect latest durable-progress receipt. Final
admission still needs GLOBAL crossfile/group heldclosure; never register both
aggregate and constituent graphs. M whole-session conversion + further bulk
sources remain work; share prepared-tokenizer seam after validation.

Native coefficient RL production route, completepolicy+optimizer/RNG/exposure,
atomic publication+lease, base migration/recollection and hygiene integrated;
rootactual17CUDA/Node/CLI/controller+5lease tests PASS. Controlledrewards only,
no productionRL/tasklift/targetFP8 claim. Cloud must transfer policy_dependencies
before enablement, after current useful SFT continuation/foundation checks.

Standing fulla6b8d5beb7 ALLGREEN05:16 (TS566s, actualGPU suite included); prior
0b5also allgreen. Standing service will advance main; no duplicate broadcheck.
An agent's already-running acquired-session fullcheck remains active and monitored.
RootcoordACK05:07, nextdue05:57; monitor all cloud/train/data/eval/export/storage
processes at least every50minutes. No hour sleep while these actions remain.
