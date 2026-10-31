# LFM3 supervision handoff — 2026-10-31 (Claude → Codex)

Written by the DGX supervising Claude session (coord identity `dgx-claude-5c7cf3fb`) at ~08:30 CET.
Live numbers go stale fast: **re-verify every claim below against processes, files and logs before acting**
(AGENTS.md "Verify, don't assume").

Authoritative logs (read these first, newest at the bottom):
- `/home/werg/data/lfm3-rl-dgx-20261029/ops/decisions.log` — every RL decision, cutover, incident (146+ lines).
- `/home/werg/data/dgxops/claude-supervision-actions-20261029.md` — hourly supervision entries.
- `/home/werg/data/dgxops/claude-deletions-20261029.txt` — every deletion/move I made (with reasons).

## 1. Where things stand

### Decisions in force (owner-approved or made under owner's full authority, 2026-10-31)
- **SFT step608 line dropped.** SFT-aligned paired eval (program-header rendering, base vs step608): base 32 vs
  step608 17 of 51 paired, gained 6 / lost 21, McNemar p=0.006 → regressed. Result:
  `/mnt/external/natlang-development-data/archive/data/lfm3-eval/decision-dgx-20261030/sft-aligned-partial51.json`.
  RL starts from base (`fresh-init`). Future SFT only on the repaired broad mixture (v4) with the serving prompt.
- **RL from base on DGX, two lineages:** `jolt-r1` (JOLT+ GRPO) and `sd-r1` (self-distillation/RFT).
  RL-vs-base after ~6 rounds: 4 vs 5 of 9 paired (no signal; lineages barely moved).
- **Neuralese bonus (owner amendment in AGENTS.md "Whole-graph RL"):** group advantages centered on task outcome
  only; +0.1 added afterwards to correct episodes whose graph actually passes Neuralese state producer→consumer
  (338c5c2ba). Failures never get it; correct non-Neuralese episodes never go negative. SD weights such rows 1.1×.
- **Broad RL task source v2** `runs/natlang-rl-broad-external-20261031-v2/cases.jsonl` (32,947 cases, 23 external
  sources + inline-curriculum slice; held dev/confirmation packets in the protection union). **v1 is withdrawn**
  (45 held-SFT-related + 55 unloadable DataBench cases); never train on it. sd-r1's v1 round-8 attempt was
  abandoned untrained (files kept in `sd-r1/round-000008`, `ABANDONED.json`).
- **Learner `--max-graph-forward-tokens` stays 180000** (120000 deferred 39% of episodes). `checkpoint_seconds`
  raised 300 → 1800 in both lineage configs (15 GB saves every 5 min + external offload filled NVMe).
- **Cloud continuation lineages (`jolt-r1c`/`sd-r1c`) not launched** (no RL lift); cloud starts 4 fresh lineages.

### DGX RL (dir `/home/werg/data/lfm3-rl-dgx-20261029`)
- Runner: ledger unit named in `ops/runner-unit` (currently `lfm3-rl-par-9`, 100 GB, high), running
  `scripts/alt_par.sh 6` (both lineages; updates serialized by `update_lock.sh`; per-lineage logs
  `logs/par-jolt.log`, `logs/par-sd.log`). Budget file `ops/runner-budget-gb` = 100.
- State at handoff: jolt-r1 round 6 learner update running (~64 of ~77 episodes; ~3 h per update);
  sd-r1 round 8 (v2, `round-000008-attempt-1`) collected, its update waits.
- **Pending automatic action**: ledger unit `lfm3-rl-restart-after-jolt6` (script
  `$W/scripts/restart-after-jolt6.sh`; survives the Claude session) waits for jolt-r1 `trained_rounds >= 7`, stops
  the runner, runs the cloud helper's GPU measurement `ops/gpu-window-command.sh` once as ledger unit
  `rl-gpu-window-1` (≤25 min; output `logs/gpu-window-1.log`, result `debug/gpu-window-1/result.json`), renames it
  `.done`, then relaunches both lineages as `lfm3-rl-par-10` and writes `ops/runner-unit`. Verify it fired (decisions.log
  line "runner ... restarted as lfm3-rl-par-10"). Manual equivalent: stop the runner unit, relaunch with
  `python3 scripts/memory_ledger.py run --unit lfm3-rl-par-N --budget-gb 100 --priority high --wait 3600 --detach -- /usr/bin/bash -c "exec $W/scripts/alt_par.sh 6 > $W/logs/par-N.log 2>&1"` and update `ops/runner-unit`.
- Runtimes (immutable git-archive copies, `scripts/build_runtime.sh COMMIT /abs/path/FROM_RUNTIME`):
  `runtime-5b9a0a6c5` (jolt r6, bound), `runtime-6e756f19e` (sd r8 attempt-1 was collected on it — keep until that round
  trains), `runtime-afb13bb47`, `runtime-cd5416516`, `runtime-d6ccac819` (main with put_bytes resume, typed-text-result
  deferral and the prep-speed fixes db204a9aa/22eb6a8ed). Both lineages have `pending-switch.json` → `runtime-d6ccac819`
  + v2 cases, 16 rows/round, 32768 context. Switching: `scripts/cutover.sh RUNTIME 'CONFIG_JSON'` (preflight per
  lineage; a bound round that fails preflight finishes on its old runtime).
- Bugs fixed this session (all on main): put_bytes removed by 23627b32f broke resume (afb13bb47); uncaptured
  typed-text-result Neuralese writes failed whole rounds → learner defers those groups and preflight skips them
  (c1c929610, cd5416516). **Open engineering gap:** differentiable text→state lowering so those episodes train.
- Memory incidents: global OOM 04:52 and ledger-floor stop 05:17 when jolt's update (~63 GB CUDA + 14 GB RSS) overlapped
  sd collection (~30 GB) plus other jobs. Keep both lineages in ONE runner unit; don't split them into separate
  units (claims don't fit). `measure_runner_peak.sh` now refuses to write an empty budget.
- Disk incident 07:30 (NVMe 6 GB): checkpoint_offload copies of deleted checkpoints held space on the saturated
  external disk. My watch script killed such copies; checkpoint interval raised. Keep NVMe > 40 GB.

### Episode-length/throughput findings (owner asked)
- Learner packs causal prefixes (round 6: 31.7M unshared → 7.2M packed tokens, 4.4×); system prompt + tools ≈
  6–7k tokens/episode (~7%): **the full prompt is not the cost.**
- Cost = long trajectories on the old curriculum: jolt r6 30% success, median 23.5k sampled tokens, failures 33k,
  35/83 budget exhaustion, 59/83 hit per-turn length. Broad v2 (sd r8): 74% success, median 4.6k tokens (5× cheaper);
  clutrr/babi ramble to length limits; spider/bird 100% (no GRPO signal → down-weight/raise difficulty for JOLT+).
- Learner ~5.4k tok/s GPU-bound. Levers: cross-episode (group) prompt prefix sharing (~6%), prep speed (11 min
  single-core before GPU), long-graph memory (cloud helper measuring checkpoint-input file offload), cloud GPUs.
- Analysis script: `/home/werg/data/lfm3-rl-dgx-20261029/scripts/ep_length.py ROLLOUTS...`; restart script copy: `scripts/restart-after-jolt6.sh`.

### Cloud (CoreWeave)
- Accounts `cw1..cw4` (`~/.config/natlang/coreweave-accounts/cwN.env`; never print or pass keys into sandboxes).
  Default account FROZEN. Cross-account test passed. **Every GPU create is refused `CWSANDBOX_GPU_NOT_ALLOWED`
  ("organization is not entitled to create GPU sandboxes") — owner/CoreWeave must enable GPU entitlement.**
  Zero sandboxes running. Spend is tracked in GPU-hours only (prices unknown): `cwctl.py accounts`.
- Owner policy: no caps, no early retirement, balance spend; on the first explicit quota error evacuate all
  accounts' state to DGX; normal export cadence.
- Tooling: `scripts/coreweave/rl_cloud.py`, `rl_cloud_sandbox.py`, `rl_cloud_autopilot.py`; plan
  `plans/neuralese/LFM3_RL_CLOUD_20261031.md`; run file `/home/werg/data/lfm3-rl-cloud/run.json` (v2 cases).
  Units `rl-cloud-autopilot-5` (one GPU probe/hour, then smoke + jolt-c1/sd-c1/jolt-c2/sd-c2, urgent coord on
  entitlement) and `rl-cloud-export-4` (exports + blob retention via retention_leases). Inputs are content-addressed
  W&B blobs (cases `54ad7d9e`, assets `e442206c`). SGLang fork rebased on Liquid's LFM3 PR (`8f2fd5894`, pin
  `8be682c94`); **GPU parity checks still owed** once a GPU exists.

## 2. Work in flight (helpers were Claude subagents of this session — they stop when it ends)
Pick these up from their files/units, not from the agents:
1. **Broad RL v3 + natlang codebase RL tasks (owner priority; Claude session 01WJvxPP handoff, 2026-10-31).**
   - **Done (on main):** v2 `natlang-rl-broad-external-20261031-v2` registered (32,947 train; unchanged, lineages read
     it). 66a9b1374 `number` accepts integers beyond 2^53. 3ae1bf1fa oracle `puzzle-solution`. Builders c6c18fa5c +
     6d46e259f in `scripts/rl_tasks/` (`SOURCES` lists all). 6d46e259f adds `semantics.codebase_check`
     (`ts-host/src/teacher/codebase-check.ts`, wired in collector: the edited natlang package must load with the
     actual loader, other files byte-identical, signatures/uses/export headers preserved) and `sources/natcode.py`
     (100 design-brief apps with their release split + codebases/ programs; explore tasks with the package as the
     root's callable folder, Neuralese `read_function` helper on half; repair tasks = directory reducer over the
     package with one planted load fault). RL runtimes: ts-host at 6d46e259f+.
   - **Verified** (`/home/werg/data/broad-rl-tasks-20261031/build-v3/<source>/verified/`, kept/built): tabfact
     1800/1800, wikisql 1798/1800, msr-sqa 1799/1800, convfinqa 1332/1800, multihiertt 1147/1800, toolace 1794/1800,
     nestful 1342/1358, tau-retail 585/598 (382 train), tau-airline 27/46 (held only), apigen-mt-retail 1497/1500,
     apigen-mt-airline 1245/1285, countdown 1500/1500, rg-multi-solution 2046/2320, databench (fixed runtime)
     3030/3054, codebase-natlang (TS-module exploration) 1441/1441.
   - **Running (ledger units):** `broadrl-v3-chain-f` (verifying codebase-py, 8,957 cases, secondary Python-package
     exploration); `broadrl-v3-verify-natcode` (natlang-codebase, 7,608 cases, 6,458 train). Output:
     `build-v3/<source>/verified/verification.json` (resumable via partial.jsonl); `journalctl --user -u <unit>`.
   - **Next:** (a) check natlang-codebase verification: repair references must load and all wrong_code controls be
     rejected; if explore cases fail to load, check the `explore_program/` `uses` prefix rewrite in natcode.py.
     (b) assemble v3 into a new dir (v2 untouched), small ledger budget:
     `.venv-neuralese/bin/python scripts/rl_tasks/assemble.py --build /home/werg/data/broad-rl-tasks-20261031/build
     --build /home/werg/data/broad-rl-tasks-20261031/build-v3 --assets /home/werg/data/broad-rl-tasks-20261031/assets-v3
     --out runs/natlang-rl-broad-external-20261031-v3 --cap-override reasoning-gym=4000,rg-multi-solution=2000,natlang-codebase=7000,codebase-py=2500
     --exclude-ids <the v2 held-set exclusion JSON>`; then held-set union (`union_groups_fast.py` in that data dir)
     and ProtectedWorlds; register cases, held packets and assets-v3 in `training/neuralese_corpora.json`.
     `NATLANG_RL_TASK_ASSETS_V3` overrides the assets-v3 location.
   - **Open:** codebase share with those caps ≈ 11k of ~50k train (~22%); reach 25-35% with more natlang-codebase
     questions per package or lower caps elsewhere. rg-multi-solution: 274 n-queens cases had a control accepted
     (dropped; inspect). codebase-natlang alias FN 6.9% (inspect the reversed-list alias). `.nl` authoring (missing
     child) is judged by signature + load only, not by execution. Not built: the 180 train design-brief exact gates
     (`/home/werg/data/luna-generation-20261026/design-briefs-pr4-review/packet-v3/`), directory reducers over
     generated data folders, iterateOn folder reducers. Not converted: big math (decontaminate), KodCode
     (Python runner), RLVR-IF. Notes: `plans/neuralese/BROAD_RL_TASKS_20261031.md` (v3 section).
3. **Successor SFT mixture v4 / coverage repair** (Claude data-coverage agent; audit and full history in
   `plans/neuralese/LFM3_DATA_COVERAGE_AUDIT_20261031.md` sections 6+, build commands at its end).
   - **Done (on main):** register crash fix a08e9eae6; per-document exact subset incl. parts renders 892fb2986;
     reference-free verify 3e4263c2d; gemini registry reason dd0eac29f; un-awaited false positives 78e232aed;
     policy v4 + builder (component carve, family caps 4%/cluster 24, one prompt view per decision, bounded
     external-agent tier) d66f2ce5a; logical-family declared/1 objective from exact own-output census
     (`build_logical_family_objective.py`, targets 40/40/12/8, true cap 8 with redistribution) a1dc24a4b+bb61df6fd;
     refine-judge crisp labels verified a1dc24a4b; `--serving-entry mix` default + full-sample 1.0 for serving
     renders bb61df6fd; prompt-view verify of every train leaf 86ef270e6; LFM3 acquired-session re-encode
     (`convert_acquired_sessions_student.py`, follows the 5,526 admitted LFM2.5 receipts incl. history-keep;
     LFM3-unrepresentable call names counted as render-pending) bb61df6fd; S3 subset resumable 9885d7c91.
     ts-host/dist rebuilt at 5f8eae420 (schemaOf). Render queue: 92 renders registered + ~24 graphed awaiting the
     loop's 30-min register pass.
   - **Running (ledger, low):** `lfm3-data-queue-loop5` (render queue driver, `runs/lfm3-renders/queue.json`;
     serving re-renders of v13-r2 [from the recovered source], static-s7k, Search/OWL, 10 gemini `-r2`);
     `lfm3-cov-coding-v2` (coding sample per-repo 12 + render → queue item `swe-rebench-coding-mix-render-20261028-v2`;
     then set the v1 item to a retired status: v2 is its superset); `lfm3-cov-acq2-0/1` (acquired sessions →
     `/mnt/external/natlang-lfm3-data-20261028/acquired-sessions-lfm3-20261031-v2`, logs `convert-{0,1}.log` there,
     resumable); `lfm3-cov-port-s4-chain` (S3 port-record subset at 100k/family into
     `.../port-records-s4-20261031/records` [12/45 families done, resumes], then `port_record_calls.sh` →
     `.../port-record-calls-s4-20261031`). No plain-bash drivers remain.
   - **Remaining v4 build:** (a) rerun `build_lfm3_joint_sft.py score-renders` and `score-teachers` (scores dir
     `runs/lfm3-mixture/scores/idiom-<sha12 of idiom_scores.py>`, currently `idiom-717f2f18e004`, 187 files) once the
     renders above are graphed; (b) `convert_acquired_sessions_student.py summary --exports
     /home/werg/data/acquired-sessions-lfm3-20261031-exports-v2.json --out-root .../acquired-sessions-lfm3-20261031-v2
     --out S.json`, write its train outputs as a `--bounded` JSON list `{mix: external-agent, path, family: Dxx}`;
     (c) register the port-record-calls-s4 corpus and render it with `student_rerender.py add` (serving mix;
     single-shape template, downweight roots); (d) `build_lfm3_joint_sft.py build --out runs/lfm3-mixture/joint-sft-v4
     --include-teachers --bounded <list>` (policy v4 default; runs prompt-view verify, CE, logical-family objective,
     geometry, learner inputs); (e) protection-union check, registration/manifests, composition report by source
     and task shape (target ≥ ~900k docs, ~360M supervised tokens).
   - **Open holds/problems:** D12 oracle-regrade hold (6,713 rows, not relaxed); D24 acquired exports only on a
     CoreWeave `/work` volume (blocked on cloud); communication own-output mass far below 40% (objective caps at 8,
     depends on the Neuralese graph helper's builds); native-sft-v13-r2 original source lost — rows recovered to
     `/mnt/external/natlang-lfm3-data-20261028/native-sft-v13-r2-recovered-20261031/` (`provenance.json`); keep
     `stale-render-state-20261031/natlang-native-turns-v13-r2-render-20261011-v1` until that render registers.
     External disk `sda` I/O pressure ~50% slows everything; an unbounded `bfs /` crawl was killed.
4. **Broad Neuralese graphs + D7 chain-graph conversion (Claude session 01WJvxPP broad-graphs agent, 2026-10-31).**
   Design/status: `plans/neuralese/NEURALESE_BROAD_GRAPHS_20261031.md` (section 7 still to fill). Root
   `R=/mnt/external/natlang-broad-graphs-20261031`.
   - **Done (on main):** case adapters `scripts/neuralese_data/broad_graph_cases.py` (11 sources, held-relation rule,
     custody resumable+parallel 45298ef37/caf9ba9e5), IR builder `ts-host/scripts/inline-curriculum/build-broad-graph-calls.mjs`,
     pipeline `scripts/neuralese_data/broad_graph_pipeline.sh`, parity check `check_broad_graph_parity.py`
     (FP32 one-pass vs cached: h/logits identical, grads <=4.3e-6), general D7 converter
     `scripts/neuralese_data/build_chain_graphs.py` (one graph per chain, reader cap 64 / result-reader 512, target
     dedupe 64, serving prompt mix, per-output cap-state; resumable phase 1+2 cde5e1a24/49b061636/6a2e1e9a2,
     kill/resume byte-identical), diversity report `chain_graph_diversity.py` (ab40a78b3).
     Converter authentication fixes (all teacher trajectories with inline instructions, handover history or direct
     Neuralese<string> results previously failed e2e_mine authentication): 945cd9ed1, 2aa17665b. Custody speedup
     c92be9ca8 (ProtectedWorlds.digest cached, 16x).
   - **Cases:** `$R/cases/cases.jsonl` 43,930 (train 40,930 / test 3,000; hotpotqa 6000, 2wiki 3478, squad 5000,
     fetaqa 5000, multi_news 4000, spider 5000, bird 5000, xlam 5244, qmsum 652, loghub 1556, musique test-only).
     Custody (`$R/ir/custody.json`): 17 train omitted (spider 5, xlam 12). Held-set group closure
     (`union_groups_fast.py`): 0 conflicts / 40,930 (`$R/cases/held-set-union.json`).
   - **Running (ledger units, low, resumable; check `journalctl --user -u <unit>` and the logs):**
     `bgraph-pipeline-3` (replay 173/441 shards in `$R/turns`, then e2e convert + chain graphs per source into
     `$R/graphs/<source>/`; converter build `/home/werg/data/broad-graphs-tmp/convfix` = 945cd9ed1);
     `bgraph-rest27-full7` (rest27 D7 graphs, 11,713 chains -> `$R/rest27/graphs`, log `$R/rest27/log`; input copy
     `/home/werg/data/broad-graphs-tmp/rest27` 14.5 GB on NVMe, delete after receipt);
     `bgraph-big2-d7-1` (`$R/big2-d7.sh`: tranches 010,011,012,009 then 000-008 when records exist ->
     `$R/big2/tranche-NNN/graphs`, cap-state `$R/cap-state-big2.json`);
     `tmine3-a/b/c` (`$R/teacher-mine/scripts/rerun_v3.sh`: 26 teacher groups re-mined with converter build
     `/home/werg/data/broad-graphs-tmp/convfix2` = 2aa17665b into `$R/teacher-mine/work-v3/<group>`, `<group>.done`
     markers; old `work`/`work-v2` being deleted by a nohup rm). v2 check on qwen-boot2-21: 13,890 records,
     5,598 edges, 166 omissions (old converter: 4,282 / 601 / 10,108).
   - **Not running:** big2 tranches 000-008 records re-extraction (paused for disk I/O):
     `python3 $R/big2-reextract.py 0 1 2 3 4 5 6 7 8` as ledger unit (low, 3 GB; writes ~140 GB, verifies each
     tranche sha against its extract-receipt); `big2-d7.sh` picks them up on its next run.
   - **Next commands:** (1) teacher chains per group once `work-v3/<g>.done`:
     `build_chain_graphs.py --work $R/teacher-mine/work-v3 --source <g> --out $R/teacher-mine/graphs/<g>
     --corpus-id <teacher corpus id> --cap-state $R/cap-state-teacher.json --workers 4` (ledger, low, 12 GB).
     (2) diversity: `chain_graph_diversity.py --out $R/diversity.json $R/graphs/* $R/rest27/graphs $R/big2/tranche-*/graphs
     $R/teacher-mine/graphs/*`. (3) spot-check producer spans/imports (decode `nodes[i].producing`, `imports`) and
     one-pass parity on ~10 full-prompt graphs (`check_broad_graph_parity.py`, GPU medium). (4) register in
     `training/neuralese_corpora.json` with manifests in `training/corpus-manifests/`:
     `natlang-broad-neuralese-graphs-20261031-v1`, rest27 D7 graphs, big2 D7 graphs, teacher-mined graphs; the big2
     D7 output supersedes the old big2 communication graphs (audit D7) for future mixtures. (5) fill design doc
     section 7, delete NVMe temp (`/home/werg/data/broad-graphs-tmp`) and `$R/cases-closure-v1`, `$R/rest27-trial`.
   - **Open problems:** ledger starvation (RL par-N reserves ~105 GB; low units preempted every RL restart; all
     stages now resume). MuSiQue train blocked: current protection quality groups cover all musique_hops questions.
     Residual teacher omissions: soft-state argument reads whose producer text the converter cannot resolve
     ("unresolved exact source references", ~0.6% on boot2). `e2e_mine.py convert` does not reconvert parts split by
     `--max-mb` in the same pass (worked around by the 3-pass loop in `teacher-mine/scripts/mine_group.py`).
     Queued sources: TAT-QA/FinQA, NarrativeQA, QuALITY, KodCode, CodeSearchNet (reasons in the design doc).
     Learner append goes through the resident source-update mechanism (not done here).
5. **Learner efficiency (cloud helper):** profile/cut the ~11-min single-core prep in `rl/native_stage.py`;
   train long graphs at lower peak without deferral (checkpoint-input file offload; exact gradient parity).
   GPU window result → `debug/gpu-window-1/result.json`.

### 2a. Cloud RL and learner efficiency (state at handoff)
**Cloud readiness.** Everything except GPU entitlement is in place and checked: the CPU pipeline check on cw3 passed
(env build, runtime, custody, imports, parent fetch from W&B, export -> W&B -> DGX GET -> sha-verified hub copy).
Run file `/home/werg/data/lfm3-rl-cloud/run.json` (v1 copy `run.v1-withdrawn.json`): lineages smoke-jolt,
jolt-c1, sd-c1, jolt-c2, sd-c2 on broad source v2, all fresh from fresh-init `5ff07638`; jolt-r1c/sd-r1c are
continuation entries not to launch unless the supervisor decides so. First steps once an account is entitled (the
autopilot does 1 and 3 by itself and sends an urgent coord on entitlement and on smoke failure):
1. Smoke: `rl_cloud.py launch RUN smoke-jolt --rounds 1` (autopilot, after a rebundle at HEAD); passed when its
   export is DGX-verified (`rl_cloud.py status RUN`).
2. SGLang GPU parity on the cloud GPU before trusting actor samples: `scripts/neuralese/check_sglang_lfm3*.py`,
   producer/adapter/decide parity and actor capture against fork `8f2fd5894` (never run; DGX GPU was busy).
3. Launch jolt-c1/sd-c1/jolt-c2/sd-c2 (autopilot balances accounts). Rebuild the bundle at a main commit that
   contains the learner fixes below (db204a9aa, 22eb6a8ed) and runtime >= 542df3c08 (v2 checks, node:sqlite).

**Exporter and blob retention.** `rl-cloud-export-4` GETs each lineage's newest export, verifies sha, keeps a hub
copy, and runs `rl_cloud.py prune` each pass: it keeps the newest DGX-verified export per lineage and anything newer,
seed blobs, current and cached inputs, and every blob committed in the last 24 h, and never a blob leased through
`scripts/coreweave/retention_leases.py` (namespace `.cloud/standalone-*`; launches lease parent+inputs as
`launch:<job>`, released on sandbox end). Intent is durable in `.cloud/rl-blob-retirement.json` before deletes;
absence is checked against the full W&B listing (W&B `files(names=)` echoes absent names; deleting an absent file
is an opaque internal error). The 24 h grace fixes a race that deleted the just-uploaded v1 cases blob (8e79f8eca).

**Learner prep speed (done, on main).** cProfile of jolt-r1 r6 admission: `prepare_objective` 353 s for 72/83
episodes (deepcopy of per-token captures 165 s, sampler top-k decode 66 s, compile_graph 42 s, duplicate canonical
hash 32 s, parse_graph 29 s) -> 184 s (db204a9aa: one turn list + one hash per episode, pickle copies, numpy top-k
validation). The objective (1.4 GB at the 120k cap, 2.8 GB at 180k: per-action base64 top-k) was parsed 4-5x and
canonically re-hashed ~5x per update (`binding()`: ~38 s plus a second ~9 GB copy each) -> one parse per process
shared by file hash, `binding()` = strict-stat file sha (db204a9aa, 22eb6a8ed); deferral runs in memory before a
single write. Remaining host-side costs: compile_graph/parse_graph per episode (~70 s), top-k decode (~25 s),
the objective's JSON size (a binary top-k sidecar would cut most of it), and the per-round read-only policy
capacity check (130-500 s of GPU per case, up to 3-4 cases; could reuse a passed proof across rounds of equal
runtime/geometry). The profiled run stopped at episode 72 on a JOLT custody mismatch caused by today's protected
union change, not by the code. Not yet in a runtime: build one at main >= 22eb6a8ed.

**Long-graph memory (measurement pending).** Approach: existing exact file storage of activation-checkpoint inputs
(`--checkpoint-input-offload --checkpoint-input-offload-directory DIR`; CheckpointFileStore: fdatasync +
FADV_DONTNEED per tensor, so no page-cache growth on unified memory). Estimate: layer checkpoint inputs are
24 x tokens x 2048 x 2 B (~17.7 GB at 180k packed tokens) of the ~63 GB peak. GPU window script
`/home/werg/data/lfm3-rl-dgx-20261029/ops/gpu-window-command.sh` (inputs copied to `debug/gpu-window-1/inputs`;
parent = current jolt-r1 checkpoint at run time, with the trainer's actor-identity check rebound for measurement
only) runs the real trainer set-up on the largest packed r6 episode twice (device vs file offload) with production
retained gradients, bitwise named-gradient + loss parity, CUDA/host peaks and time, no optimizer step; result with
a `conclusions`/`verdict` block in `debug/gpu-window-1/result.json`, log `logs/gpu-window-1.log`. Needs >= 30 GB
free NVMe (checked). If the verdict is ADOPT: set `checkpoint_input_offload: true` and
`checkpoint_input_offload_directory` (NVMe scratch) in both lineages' learner trainer parameters, then raise
`--max-graph-forward-tokens` by the measured headroom (fewer deferred episodes). Not started: smaller recompute
peaks (sequence-chunked MLP in the checkpointed layer) if offload alone is not enough.

**Open: text->state lowering.** Child calls declared `Neuralese<string>` that return computed text get a block
written by the runtime writer (graph `block_write`, `source_kind: typed-text-result`, `producer: null`) that is not
in the sampling capture. These episodes are now deferred whole (c1c929610/cd5416516; 6/96 in sd-r1 r8). The fix
is either a differentiable text->state lowering in the whole-graph learner or recording the encoded state as an
explicit stop-gradient external origin with its exact consumer bindings.

## 3. Owner-pending items
- GPU entitlement for cw1..cw4 (blocks all cloud work).
- Delete `/mnt/external/outputs/proto1` (24 GB, Feb checkpoints)? `sudo rm -rf
  /mnt/external/natlang-native-directory-20261002/model /mnt/external/natlang-vllm-qwen36` (root-owned).
- AGENTS.md still has stale Mellum/Qwen/VAE sections. Memetic authoring: self vs JOLT+ teacher.

## 4. Gotchas learned this session
- Ledger: `run` without `--wait` fails fast when headroom is short; with `--wait --detach` it queues as
  `<unit>-waiter` and preempts lower resumable units. Requeued stopped units can be re-admitted later —
  `memory_ledger.py cancel --unit X` when retiring one. Low jobs are guard-stopped first under pressure.
- Large copies to `/mnt/external` must use the bounded copy (`archive_to_external.py` / `checkpoint_offload`
  verified_copy); an unbounded rsync built dirty cache and contributed to the OOM. External disk is I/O-saturated.
- `archive_to_external.py` leaves symlinks; mark live dirs with `.keep-on-nvme` (done for both RL dirs). A `find`
  for deletion must not traverse into the archive (I deleted the archived base GGUF that way; regenerated via
  `claude-regen-base-gguf-1` to `/mnt/external/natlang-development-data/archive/data/lfm3-eval/LFM3-1.7B-BF16.gguf`
  — verify it completed).
- A runtime is in use while any unfinished round was collected on it (resume preflight resolves it), not only
  while a lineage's controller-runtime names it. I deleted runtime-6e756f19e too early and had to rebuild it.
- Never edit a running bash script in place (restart it). `build_runtime.sh` second arg is a runtime *path*.
- py-spy needs sudo here; use timing logs/cProfile.
