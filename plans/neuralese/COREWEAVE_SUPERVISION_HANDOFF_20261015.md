# CoreWeave supervision — 2026-10-16 06:47 UTC

DGX owns execution; Pop is closed. Two training sandboxes and one self-hosted
Qwen3.8-Flash-Next NVFP4 teacher; no further allocation. Comprehensive adaptive
training of both models remains the objective. Neither original 26,312-graph
communication foundation is complete. Expanded 57,144 is tracked separately.
No trained joint dialect, production RL or task-lift claim.

## Actual training and recovery parents

- Both learners now make real AdamW updates for native ordinary parameters;
  backbone MuonSR is unchanged. Explicit migration preserves weights, RNG,
  exposure, unaffected optimizer state and historical Muon buffers as provenance.
  New Adam moments honestly start at zero. Shared fixes cbcbb223b/f1630b52d;
  native transport=true, noise=1. Input/output role rates .0003 for both;
  LF reasoning .0003, M reasoning/backbone/adapter 3e-6. These rates are adjustable.
- LF runtime /work/native-runtime/6b7284057078, recipe538107/trainer538125.
  Last measured5445, 69 post-migration updates; 102 communication graphs with
  positive producer-marker gold support. Actual5378:6593 supervisedTPS/GPU100%,
  16.33GB peak. checkpoint_layers=true/inputoffload=false: CPU input offload
  previously hit the real64GiB host limit. Required exact5376 parent:
  2,241,135,623 bytes, SHAed2c99000b86c3aeb3b5f7d03026e03e2e879186e8644bbef6c9faad08b935e7.
  Original foundation9037/26312 at5376. New-source capacity requires a real
  current no-offload certificate; never rekey the old offload=true proof.
- M runtime /work/native-runtime/28432eb5e0c8, recipe353591/trainer353599.
  Last measured5504,119 post-migration updates; 70 positive-supported comm graphs.
  Actual5399:5528supervisedTPS/GPU99%,54.52GB peak. Checkpoint/inputoffload=true.
  Exact5385 parent48,041,003,941 bytes,
  SHA0d467f4c3249817e1b39795e24329fad3929f8b0c73b7136514ccd42052cea18.
  811971 bound rows; original foundation7371/26312 at5385. b98abb0f2 fixes exact
  V8 resident predecessor recovery without resetting exposure. Big2/Search
  resident CPU prepare355075 scans97600new train+9972held; do not restart it.
- M5500 held producer-end argmax22/22, meanp=.9783; producer-begin remains weak
  (.000935/NLL7.535). Different held cohorts prohibit matched temporal claims.
  Keep actual task, held loss and per-row gradient/update telemetry.
- Measured capacity continuation24cb71bbf bounds ordinary Adam storage delta
  against exact old numerical/capacity proof and approved runtime inventories.
  M current derived cert e95aa1b0084455d64524fcea98135002c518fa3cfc794aecd4044b98691c3df4
  at control/mellum-native-adam-recovery-20261016-v1/inherited-capacity-resident-predecessor.json;
  limits112008physical/173589KV. New-parent measurement is not claimed.

## Teacher and data admission

- Teacher batch12 closed3826:3354success,95semantic,366invalid,11typed model-budget.
  Root actual3826 closure qualified with teacher_batch_completion.py (50004e610);
  typed budget closure advances scheduling, never relabels failures as usable data.
- Fresh batch13 started06:15:59 parent584911,512 reference-qualified fresh cases,
  two variants/1024queued,8workers. Successor guardian588486 waits exact parent
  identity and drained workers, validates closure, launches canonical v6 feeder.
  No extra sandbox. Fresh packet schema support currently blocks V15 harvesting:
  cloud fixes exporter nested protection; source agent fixes converter admission.
- V14 ACTUALLY registered72f78f1d1:2250train/model, source134trajectories244790997B
  SHAd8c628457eeab0860bb54329837d3087a00c3177fb0e230c03dd194496e89ada.
  Staging2219 completed both; registered/staged is not exposure. V8 actual M
  resident adoption; later teacher batches await actual per-row exposure.
- D12 full cohort ACTUALLY admitted2945f45af:5025 original sessions,2826 usable
  whole train graphs/model,2199 nonpositive omissions/model,0held. All6archives
  both tokenizers. Global held closure16455groups. Source audit plus4 exact
  source-render examples; no oracle/native execution claim.
- D12 capacity partition preserves all5652 original row bytes and provenance:
  receipt59632e5e1a253925ac4730144bcf8b2673db4f5175c3c1f1bd3a2b1aa6d640f5.
  LF993fit/1833long at32768/32768 UNQUALIFIED operational limits;
  M2526fit/300long at112008/173589. Partition is not yet registered/staged/adopted.
  Never append both full aggregate and its fit subset. Shared integration1be58289e carries mandatory
  exact pending-row obligations in continuation/SFT/recipe/readiness, including
  zero-update declaration admission and later activation without double credit.
- Local9624normalized sessions/730actual export receipts coverD12/D15/D16/D18.
  Frozen3259 original converter remains active onD15. Parallel D16/D18 low-priority
  units use same frozen converter, exact source/hold binding and disjoint leases;
  original worker will verify/reuse their receipts. Sources not yet exposed.
- LF durable full-native worker525307/525311, runtime82bc5b377, output
  /work/data-preparation/acquired-agent-full-native-20261016-v2-durable:
  seven100-source-record shards/frontier700 committed06:02; full completion
  and admission unclaimed. Full normalizersD22/D35 andD17/D25 need refresh.
  AdditionalD28/D27/D26/D38/D23 normalized worker stagedLFv2; cloud owns actual
  launch at available CPU slot. M full-session native handoff remains required.
  All16acquired sources have adapters;D09/D20/D24/D39 actual authenticated403.

## Evaluations and checks

- Corrected actual-native LF5000 unit taskeval-dbb7eb81-0005000 remains active.
  Latest file discovery shows10 job results, refreshing semantic counts. Earlier
  five completed were semantic failures despite native writes/three reads;
  investigate runtime/control read-out, never infer inability from aggregate loss.
- M inference-only view verified24,556,281,665bytes,
  SHAe897b6854839735c09ea3324e3e6bf0e48c3155ab49d5f0b9da9aaab4b40b6b2;
  exact full5000 training parent unchanged. Fetch/baseinventory/seal/queue ACTUAL
  completed06:28. taskeval-c2be9511-0005000 ACTIVE, now actual job result files,
  ~53.4GiB combined CPU/GPU use. Single existing dispatcher serves both.
- Standing broad check24cb71bbf hung at41% on root's unbounded process identity
  retry with fake/nonmatching PID. Explicit unit stopped; not verified. Fix
  e48425811 bounds identity observation, retains ownership token and child
  supervision, tests absent PID and actual shebang wrapper recovery. Actual37
  readiness/semantic/RL focused tests PASS22.44s +quick. Standing timer runs
  latest main; keep one broad suite. Last fully green a6b8d5beb7.

## Storage and retired operations

- Search four existing workers retain earned receipts/runtime SHA. USB rotational
  external scratch causes heavy I/O stalls. Relocation writers paused after M
  ferry completion; root~20GiB/external~34GiB free. Source bytes remain until
  SHA+file/directory fsync. Tiny exact dedup watchers remain. No live scratch move.
- SQLite optimization a01b16d34: defer secondary indexes until bulk insert and
 64MiB cache;3focused parity tests PASS. Agent preparing subset execution and
  mixed-version final manifest before deploying to unstarted current shards.
  Never relabel old groups as produced by new renderer code.
- Root pruned only verified redundant transfer TARs:1534506981B then876794139B;
  original sources and both cloud verified copies retained. Required diagnostic
  parents/source/frozen runtime receipts stay until dependencies are resolved.
- Found retired natlang-clef-pools-daily.timer still externally generating.
  STOPPED/DISABLED timer+service. Guard52c671cc0 forbids new API generation before
  writes; existing historical render/register retained.2focused tests+quickPASS.
  Other generator timers audit found no active unauthorized generator.

Root coord ACK06:43 (next due07:33 latest); monitor all learners, source workers,
queues, exporters, evaluations, guardians, storage and actual exposure. No hour
sleep while these implementation, admission and diagnosis tasks remain.
