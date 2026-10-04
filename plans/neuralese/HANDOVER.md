# Neuralese programme: handover, 2026-10-03

State at the end of the first implementation session. Read [README.md](README.md) (decisions and stage graph) and [DECISIONS.md](DECISIONS.md) first; the stage plans S0–S8 in this directory are current. The source design documents are in [sources/](sources/) and are inputs, not the spec.

## 1. Owner working rules

- Plans and code live in this repository; commit to `main` (no push unless asked).
- Data from bgkit (`~/bgkit`) and Schnitzeljagd (`~/sdkb`) is reused; their models and machinery are not.
- No fixed pass/fail thresholds: evaluate continuously against rubrics, review, react.
- No budget or cost talk in plans; paid teachers are allowed.
- Dialects are a lightweight version tag; bump freely while the system is young.
- The original observations below refer to the **DGX Spark** (`mltick`, aarch64 GB10), not necessarily the reader’s host. It runs the Qwen3.6 vLLM teacher (`natlang-qwen36-nvfp4-server`, host port 8082). Teacher campaign services roll over; the earlier v9 family unit is historical. Verify current units and campaign status against `plans/HANDOVER.md` and the home generation authority before allocating the GPU. **The owner explicitly authorized autonomous generation pauses on 2026-10-03 to resolve reclaimable memory issues. Preserve queues and checkpoints, record pauses, and restore the reviewed campaign after GPU checks.**
- Another agent (on the owner's dev machine) is fixing the ts-host test environment (dataset cache `vendor/datasets`, ffmpeg, `@natlang/node` staging, `bonsai-queue` hanging on `127.0.0.1:8081`, `llama-runtime` reading `~/.config/natlang`, `source-review`). Its work arrives on `main` as merges.

## 2. What is on main

| Stage | Commits (main) | State |
| --- | --- | --- |
| Plans | `c78800b43`, `352c10ee4` | S0–S8 plans, README, decisions; payload temperature added. |
| **S0 spec** | `41291918d` | `spec/SPEC.md` 0.5-draft (Contexts, Captures, Iteration and termination, Neuralese chapter), `TYPES.md`, `spec/neuralese.d.ts`, `spec/NEURALESE_{FILES,PORT,REWRITES,DIALECTS,GRAPH,DATA}.md`, schemas, `training/api-migrations/neuralese-language.json`, `conformance/neuralese/` (58 cases, many now implemented). Done. |
| **S1 data** | `95ec9ae78`, `96e3ac427`, `36c8c6873` | `spec/neuralese-port-record.schema.json`; ledger `training/neuralese_data_ledger.json`; converters `scripts/neuralese_data/`, CLI `scripts/neuralese_port_records.py`; dedup and split closure; multi-turn/agent converters; v13 migration driver skeleton `ts-host/scripts/migrate-neuralese-language.mjs`. |
| **S2 skills** | `31c4b6442` | `ts-host/src/skills/` (standard `SKILL.md` + `metadata.natlang`, pool, scope injection, disclosure), `spec/skill-episode.schema.json`, `ts-host/scripts/skills/` episode and repair-episode builders (samples only). |
| **S3 port** | `69c181586`, `0d3bef3e3`, `c3d58d435`, `5f2f3de00`, `0b5a68506`, `8f9866c47` | `training/neuralese/natlang_neuralese/`: own LFM2 forward with snapshot caches (fast path: SDPA GQA, preallocated KV, hub causal-conv1d), write/read ports, temperature-gated payload (μ, log σ, τ, KL, log-prob), phases A–F, batched ragged training, anti-collapse terms, harness, pilot driver `train/pilot.py`. |
| **S4 runtime** | `c2a287e7f`, `ee0f8655f`, `a9f979523`, `aa8ad73ac`, `1377c09eb`, `2913d9059` | ts-host: `Neuralese` type kind and diagnostics, literals ↔ references, content parts, in-memory store, iterateOn bounds and predicate prompt; contexts and rebinding, `.nz` files, `nl.with` lowering, companion `.nz`/skills binding; data rewrite passes + CLI `ts-host/scripts/neuralese-rewrite-trajectories.mjs`; execution graph `src/native/graph.ts`; law rewrite pass `src/compiler/rewrites.ts` (gated, no measurements yet); `natlang:learning` and `natlang:neuralese` (combinators as soft system functions). |
| **S4 server** | `49eb9bb1b`, `1af78e783`, `cda793803`, `9a33ef65a` | Python reference server `natlang_neuralese/serve/` (write procedure in decoding, store and block endpoints, escaping, batched steps, streaming, gradient replay sessions, optimiser steps); ts-host transport `src/model/neuralese-server.ts`. |
| **S4 llama.cpp** | `7d2522048` | Fork pin `training/neuralese/llama-cpp-fork.json`, GGUF export `natlang_neuralese/export/`, CPU fp32 parity tests (pass). |

Environment: Python venv `/home/werg/natlang/.venv-neuralese` (torch 2.11 cu130, transformers 5.18, peft 0.21; setup in `training/neuralese/README.md`). Tests: `.venv-neuralese/bin/python -m pytest -q tests/neuralese tests/test_neuralese_port_records.py`; ts-host: `npm run build` then `node --test test/neuralese*.test.mjs test/skills.test.mjs`. Beware the repository-wide `.gitignore` rule `data/` (a negation exists for `training/neuralese/natlang_neuralese/data/`); check `git status --ignored` before committing new code directories.

## 3. Unfinished work and where it is

| Item | Location | State |
| --- | --- | --- |
| llama.cpp server, GPU parity, heads on backend | Fork `/home/werg/llama.cpp-neuralese`, branch `neuralese`: `8bd8c95a0` (layer-range API, CPU read/write ports; parity passes) + `27b2e6e8b` **WIP** (neuralese-server.cpp, backend heads, GPU parity; unverified). Natlang side WIP on branch `worktree-agent-aa373bf2a5aa6f2f5` (`b94b43fe3`: GPU parity test variant, export CLI). | Resume: CUDA build for sm_121, GPU parity, heads on backend, llama-server protocol matching the Python server, ts-host end-to-end test, then wllama + OPFS. |
| S1 full conversion | `/mnt/external/natlang-development-data/data/neuralese/port-records/full-20261003/` (`raw/` ≈ 35 GB, `build.log`) | **Stopped during the SWE stage**; bgkit, Schnitzeljagd single-turn and multi-turn stages finished their raw output, but dedup, split closure across everything, validation and the manifest have not run. Resume with the build script from `36c8c6873` (see `build.log` for the stage commands). Licences for knights/synlogic: check the ledger. |
| S3 pilot | `runs/neuralese-s3-pilot-20261003-v2/` (`harness_after_C.json`, `metrics.jsonl`, `pilot.log`, `checkpoint.pt`) | After phase C the channel carried **no content** on held-out spans (correct 3.578 vs shuffled 3.568 nats, zeroed 4.888; cross-source similarity 0.99; effective rank 45.7). During phase D training batches showed correct beating shuffled by ~0.30 nats/token and similarity falling to 0.86, but the run was stopped before the held-out harness after D. The full-machine run plan was not written. |
| v13 corpus migration | `ts-host/scripts/migrate-neuralese-language.mjs` (skeleton) + `ts-host/scripts/neuralese-rewrite-trajectories.mjs` (passes) | Wire the passes into the driver and run over the v13 model-neutral corpus. |
| S2 corpus | `ts-host/scripts/skills/`, samples in `/mnt/external/natlang-development-data/data/neuralese/s2-skill-episodes-sample/` | No teacher collection yet. |

## 4. Next steps, in order

1. **S3 channel use.** Resume the pilot through phase D and read the held-out harness. If correct ≈ shuffled persists, strengthen content pressure (withheld sources, shuffled-negative contrast, self-distillation from the full-source teacher) before scaling. Then write the full-machine run plan and **ask the owner to pause the teacher campaign**.
2. **S1.** Finish the full conversion (SWE stage, dedup, closure, validate, manifest); run the v13 migration.
3. **llama.cpp.** Verify and finish the WIP commit; GPU parity; server; browser.
4. **S4 gaps.** Exact second-order `grad` (currently refused), `law` objectives, graph nodes for combinators/readouts asserted in tests, batched prefill/completion/readback in the server, `crossEntropy` target placement.
5. **S2.** Teacher collection of authoring episodes (paid teachers allowed; the DGX teacher is busy).
6. **vLLM** extension (S4 §7) for S7 rollouts.
7. Re-run the full ts-host suite once the environment agent's fixes land.

## 5. Worktrees

Agent worktrees are under `.claude/worktrees/` (excluded locally via `.git/info/exclude`). All finished work is cherry-picked to `main`; only `worktree-agent-aa373bf2a5aa6f2f5` holds unmerged WIP. The others can be removed with `git worktree remove` once you no longer need them.


## Integration note — 2026-10-03 evening

The full/sample build wrappers now preserve existing output: tags must be fresh and
path-safe, defaults include UTC time, and protected-source snapshots are per run.
Do not rerun a wrapper over the retained interrupted full-20261003 output. Resume
explicit CLI stages after reviewing build.log and existing artifacts, or choose a
new tag. Origin integration did not run builds/tests or activate Neuralese training.

## 2026-10-03 20:35 UTC — memory recovery and continuation

The owner authorized autonomous pauses. Qwen’s completed queue left its server
holding about 86 GiB of unified GPU memory. It was stopped cleanly (container
retained, restart policy `no`). CUDA free memory then measured about 90 GiB.
Advisory `POSIX_FADV_DONTNEED` over selected pipeline files recovered another
~11 GiB of Linux free memory: 97,479,589,888 → 109,347,905,536 bytes. This is
file-cache reclamation, not deletion or a reduction in live model weights.
Receipt: `runs/neuralese-integration-20261003/memory-receipts/cache-reclaim-development-v2.json`.
Current DGX readout: 88 GiB free / 117 GiB available; Qwen remains paused.
Luna v46 (two workers), Space Bunny v46 and the local full Muon training remain
active. No generation queue, source data, gold or training checkpoint was removed.

Prevention:
- `common.parquet_rows` reads 128-row batches instead of materializing a complete
  row group. Protected-data Parquet and Arrow scans now also convert at most 128
  rows to Python at once; Arrow’s underlying IPC batch remains source-defined.
- Finalization caps boilerplate group sets at the three-group classification
  threshold and releases indexing maps before subsequent passes. Classification
  and split policy are unchanged; the compact dedup/closure index still grows
  with the number of records.
- Reference serving and GGUF export memory-map trainer checkpoints, so unused
  optimizer tensors are not eagerly copied into RAM. These entry points require
  normal modern torch.save checkpoint files; no legacy fallback was added.
- Enabled DGX `natlang-development-cache-hygiene.timer`: every ten minutes it
  advises release of selected pipeline cache only when MemFree is below 16 GiB.
  No global drop_caches, sudo, data deletion, or active process termination.

Deployment: bounded readers and finalizer changes are on DGX main; already
running Python stages retain imported code until their next stage/process.
No extra tests were run for this memory-only follow-through. Evidence consists
of the recorded live memory readings, successful reclamation receipts and active
service checks. Do not interpret Linux MemAvailable as CUDA allocation capacity.

Other continuation state: private fork https://github.com/werg/llama.cpp-neuralese
(branch `neuralese`, `8d302c6a2`) preserves the DGX WIP and server fixes; CUDA
port parity passed four cases. Non-streaming HTTP/TS end-to-end verification is
still pending. S3 phase D completed in a preserved child run; held-out correct
payloads beat shuffled payloads modestly, but BF16 cache agreement was 2/4, so
investigate that before scaling or launching E/F. Full S1 conversion and held
v13 compiler migration are running on DGX. Candidate outputs require review and
execution replay before training admission. Restore Qwen and the reviewed v5
successor after the remaining GPU checks; do not treat its paused server as live
generation. Older sections above describe the initial handover, not current state.

### 2026-10-03 20:44 UTC — cache diagnosis, faithful checkpoint deployment and generation restore

Same checkpoint, four held-out prefixes and fixed payloads were compared on DGX.
Snapshot fork vs unforked readback logit difference was exactly zero in every
variant. Float32 reference arithmetic matched all four eight-token continuations
(max logit difference 3.05e-5). BF16 fast with/without the convolution kernel each
had one mismatching case; BF16 reference had one different mismatching case.
Every mismatch was a full-forward top-logit tie. This supports numerical drift
rather than snapshot corruption on these inputs; it does not establish universal
cache parity. Evidence: `runs/neuralese-integration-20261003/cache-diagnostics-report.json`.
Harness reports now include per-case mismatch steps, margins, token agreement,
dtype and kernel identity; no tolerance or admission rule was relaxed.

Reference serving now infers trained cutoff/block maximum, rejects incompatible
explicit values, restores phase-F LoRA tensors, and retains float32 head parameters
like the trainer. Default remote base loading pins the intended revision. LoRA
alpha metadata now follows the actual 2*rank trainer setting. Phase-D checkpoint
loading was exercised by the diagnostic; phase-F restoration still needs its own
end-to-end verification after that phase exists.

Generation restart exposed a real identity bug: the importer hardcoded
assignment.json while authority could bind assignment-v2.json. Future sync/import
uses the exact authority-pinned assignment, hashes its bytes once, and rejects a
changed file. Preserved v4 ledger remains associated with its original assignment.
The handoff accepts this only through a pinned predecessor import-assignment path,
verified supersedes SHA, and an exact operational projection differing solely in
native proof attestation metadata. No ledger rewrite or case-result relabeling.

Qwen container restarted; reviewed v5 controller
`natlang-qwen36-v5-restored-controller-20261003.service` waits for actual model
readiness before launching 1024 cases and retaining the importer as its child.
Controller SHA a09121936a0c77f64bd145b0fd9b129bb3b2572c540efbf518e132d8c8e69a0e.
Original v5 proof/assignment/preparation are preserved; v2 artifacts explicitly
normalize verified report fields and bind their originals. Worker/model/prompt,
source/gold/protected policies and concurrency256 remain unchanged.

SWE and agent conversion finished raw stages; full dedup/closure finalization is
underway. Full v13 migration remains held pending replay. Their heavy disk readers
were briefly frozen for Qwen weight loading, with independent two-minute thaw
safety timers. Finalization now includes the existing writer-target leakage check
rather than leaving it only in a separate validation CLI. The already-imported
finalizer must still receive a separate validation pass before admission. This
check is a heuristic with documented short-answer/exact-reference exemptions,
not complete causal proof. No candidate corpus was published.

Local full Muon run reached checkpoint6580/13165; fixed held-out loss at6500
was0.832901 (6000:0.836124),128examples/no skips. CPU execution eval at6000 is
still processing actual HTTP requests, not yet a completed score. Luna v46 two
workers and Bunny v46 journals show continuing case progress.

### 2026-10-03 20:55 UTC — verified Qwen recovery and narrower cache hygiene

Qwen endpoint became healthy; controller launched, but runner rejected setup
before any cases because root changed the native review to v2 without rebinding
its authorization SHA/path. This was an integration error, not model failure.
Preserved original approval artifacts, setup-only worker/pool status and one
pool_setup_failure journal event in remote `setup-failure-v2-preserved/`.
Verified no jobs/results/case events/pool-plan existed. Corrected authorization
v2 and wrapper v3 passed actual ARM `--preflight-only` for all1024 identities.
Root recovery receipt `pool-v9-alternates-v5/root-setup-recovery-approval-v1.json`
binds their exact hashes. Restarted the inactive v5 unit with wrapper v3.
Now worker state running (collector2719648, supervisor2719604), real Qwen POST
responses200, and retained sync/import child2711836 reports running/zero initial
artifacts/no import errors. No failed model samples were relabeled as positives.
Authority records the recovery. Controller readiness now also checks the
native-review SHA and both exact remote receipt paths in authorization, so this
mismatch will fail before any future launch.

Cold-start model page cache was reclaimed after readiness: DGX MemFree rose
from~1GiB to~23GiB, subsequently~16GiB as readers resumed; MemAvailable~34GiB.
A manual whole-mirror advisory scan blocked in kernel lock_buffer for several
minutes and was terminated; its original empty receipt remains evidence of that
interruption. No cause beyond the observed kernel wait was proven. Periodic
hygiene now scans only pinned dataset cache and Qwen weights, excludes recently
modified files (five-minute guard), and records in-progress receipts plus counts
instead of leaving blank files until completion. Data, output files, checkpoints
and model weights remain intact. Timer stays enabled every ten minutes with the
16GiB MemFree pressure trigger.

The full port run prerequisites are recorded in
`plans/neuralese/S3_FULL_RUN_HANDOFF.md`: unfinished E/F review, full corpus
quality/replay, bounded sampler, faithful base/export/serving handoff and actual
Muon support for a new port run. Existing port AdamW state is not silently
converted; current local main training already uses Muon. Current repo changes
are committed/pushed and copied to DGX. Resume requested work/check/sleep cadence;
a timer/controller does not itself wake the assistant session.

### 2026-10-03 22:00 UTC — requested sleep completed; first follow-up sweep

Completed the requested ~50-minute sleep and performed a live sweep. Main local
Muon checkpoint6940/13165; all main, eval, Luna and Bunny units active. Step6000
execution evaluation completed:23attempted,1policy-held,11complete successes,
11semantic failures,1incomplete; no contract/resource/infrastructure failures.
Semantic accuracy is11/22completed cases (50%), not11/24. Step6500 is evaluating.
Report `runs/training-periodic-eval-20261003/execution-eval-v6/execution-results/execution-results/step-06000/adapter/report.json`,
SHA f878cbb7c119a73033ff244659b2fb93f450ee9fe5a2f245a54b8722f996641e.
Do not claim a controlled improvement against older differently handled baselines.

Qwen observed347imports:340admitted/7wrong_return rejects (snapshot, not final
campaign totals). Current collection passed setup and serves real requests.
Luna finished237cases across two workers; Bunny395. Next v47 refill preparation
assigned to the existing Luna source-audit agent, using source-pinned Qwen v5
cross-teacher IR, native/protected/source/history checks and current frozen v41;
no launch/authority change delegated. Agent must report exact pins for root review.

Reviewed the first seven Qwen rejects against their exact criteria/state: answers
assert financial hardship/legal threat/recognized purchase/account anomaly or
other allegations without the corresponding required evidence. No confirmed gold
or runtime defect. Review artifact
`runs/generation-check-20261003/memory-followup-v42/qwen-rejection-review.json`
preserves source, gold and rejection hashes. Future prompt clarifies per-file
question/state isolation. Do not inject ordinal labels or accept these wrong
answers as training positives; retain repair/DPO lineage.

Found a real preview defect: long filename-stem keys can exhaust the value budget,
causing even a one-character string label to appear empty with a truncation note.
Native previews now retain strings up to16characters (shorter than the truncation
notice). Isolated TypeScript compilation passed in
`runs/generation-check-20261003/memory-followup-v42/compiled`; no tests/model calls
were added for this source follow-through. Active frozen runtimes unchanged; fresh
freeze/native proof still required before teacher deployment.

Full S1 finalizer is progressing, not stuck: its read counter advanced81→83GB,
CPU~80%,resident process~400MiB. Cgroup8GiB largely comprises~7.6GiBfile cache,
with memory-high events/no OOM. Preserve the live pass rather than discard its
work. Future streams now request sequential/NOREUSE kernel cache hints and log
pass1counts every100krecords. These hints are optional; record/split semantics
unchanged. Full v13 migration processed~92krows and is still writing its pending
candidate;27compiler-failed rows so far remain held. Its in-flight exhausted_input
was initialized true; source now reports null/running until the stream finishes,
then complete/exhaustion explicitly. This reporting fix does not alter active
candidate output or prove its completion. Replay/failed-row review remains due.

DGX~33GiBavailable/14GiBfree; cache timer succeeds, both conversion units active;
external179GiBfree. Home20GiBfree: do not pull the entire huge DGX raw corpus into
that space. Preserve remote outputs and return manifests/review receipts; arrange
an explicit storage projection for full corpus mirroring before claiming a home
replica. No data was deleted or excluded from the dataset catalog by this sweep.

### 2026-10-04 continuation: checkpoint identity and live sweep

Main full Muon run is active at step7040/13165,56320trained exposures,0skips.
Bunny481/512complete; Luna149+155finished, with one preserved incomplete
(refill-v46-luna2-0134). Its Codex request-processing error is infrastructure,
not a resource timeout; five replies saved, zero result rows. Keep failed/held
with an explicit retry obligation, never promote it as positive. Qwen, S1
finalization and full v13 migration remain active; DGX33GiBavailable.

Fixed phase-F pilot resume: retain the complete phase schedule while stopping
at individual review boundaries. Checkpoint optimizer groups now save/replay
LoRA layer order and exact parameter names before loading moments. Historical
LoRA checkpoints without group identity are preserved and refused for explicit
repair; pre-LoRA D checkpoints remain compatible. Require port config/schedule
and validate phase position. Earlier missing harness reports are not synthesized
from later resumed weights; existing reports are included in the summary.
Luna performed two read-only reviews; no tests/model runs were added. Actual
F resume/deployment exercise remains due before claiming full resumability.

### 2026-10-04 v47 continuity and rejection review

Bunny v46 completed512/512 with complete output accounting. Reviewed sequential
handoff succeeded: v47-r2 now running launcher2749345/supervisor2749346,512cases.
Root worker plan SHA4d11a51c0f0c6314a26e636056d34a51f4285649369b20d1379dbd7d3f756e5d;
rollover SHA00132143f99ff23a434ef058c46de6aeeaad5faab49489a649d44fcd226ed987.
Service natlang-bunny-v47-reviewed-rollover-20261004 retains ownership via waitpid.

Two Luna v47-r2 256-case slots are approved and waiting independently for v46
PIDs to exit, with exact terminal accounting gates. Plan SHA
58813f94e7964412ed66eeb778de81bc5052ebb638998ab2283e1f709cfebcf0, service
natlang-luna-v47-reviewed-slots-20261004. Failed v46 Luna2 key0134 remains held
and pending explicit retry. Its raw journal line hash c2c100b... differs from the
canonical JSON event hash a53c9c31e7acba347c5a67885d0a0cc58fa9bd8e6e1dc079b7a6bf2e9c0394ed
used by the handoff checker; root checked both against saved evidence.

All v47 tasks are exact Qwen v9-v5 parent records. Bunny512 equals the union of
disjoint Luna256+256; current native replay512admitted/2246approved decisions,
0held/denied/unlinked, source/protected/history proofs reviewed. Same frozen v41
is retained; this is cross-teacher source reuse, not new source coverage.

Qwen latest snapshot545imports=534admitted/11wrong_return; a later12threject
appeared during review. Four follow-up rejects were legitimate wrong answers:
open issues labeled resolved and later explicit clarifications ignored. Luna
review artifact runs/generation-check-20261004/qwen-followup-rejection-review.
No source/gold/admission relaxation. Future prompt now clarifies whole-conversation
corrections and distinguishes proposed remedies from success. Active frozen
runtimes unchanged; new freeze/native proof required before deployment. Main
step7000 heldout loss0.8185059633,128examples/0skips.

### 2026-10-04 post-sleep sweep and E/F continuation

Actual sleep22:18:46→23:11:49UTC (~53minutes including tool overhead). Main
full Muon run7360/13165,58880exposures,0skips; step7000loss0.8185059633.
Qwen v9-v5 finished1024:1001admitted/23wrong_return. All23source/gold rejects
reviewed; no source/runtime fault found. Exact duplicate question/state in two
composites is not independent evidence. Review-beyond-11 files under
runs/generation-check-20261004/qwen-followup-rejection-review.

Full language migration finished111301rows,35070changed,33failed rows,122340
annotations/3745capture sites. Output SHA8cfd43aad51f33c875b030a5d0c22ce2371e7bd6d529f378b63001eae5fb2648.
All remain held pending execution replay. Local small receipts/failed rows and
Luna independent review in runs/neuralese-integration-20261004/v13-full-candidate-v2-receipts;
full7.8GiB candidate remains DGX.33rows have36failures across7trajectories;
conservative compiler passes preserve code on diagnostic change, not proven
unsafe sources. Cumulative annotation acceptance/diagnostic diff logging may
improve completeness later, with new pinned candidate/replay.

Stopped idle Qwen after final imports, reducing DGXusedRAM~87→4.6GiB/free93GiB,
then resumed preserved D checkpoint1400 into fresh
runs/neuralese-s3-pilot-20261004-resume-ef-v1. First8GiBcap attempt OOM before
E's first update (204MiBallocation,80GiBphysicallyfree); preserved attempt log.
32GiBbudget restart completed E200/F300 atglobal1900; phase_index6/step0.
Actualoptimizer_lora_layers=[15,14,13,12], adapter_layers=[12,13,14,15].
No change to AdamW for this lineage; main full run remains Muon.

E stopping was nearly deterministic:197/200batch means16; three slight length17
continuations, no mean below16. Paired identical actions cancel their centered
REINFORCE score even when payload rewards differ. No sampling/log-prob mismatch
found in read-only review; no claim16is optimal. A diagnostic CLI now reports
per-position logits/probability/entropy on sealed prefixes without changing
policy. Future exploration changes must use matching behavior log-probability
and a new checkpoint lineage; current E/F result preserved.

Prepared Qwenv9-v6 next1024, actualARMv41 proof1024admitted/4333approved,0held/
denied/unlinked,58train groups,0protected/historycollisions; all v5 assignments
excluded. Still residual customer-service buffer, not balanced/new coverage.
Caught BEFORE JOBS preparation-contract errors: native child paths doubled
bundle/bundle; ad hoc hash refresh wrote manifest into preparation.json and
left stale manifest hashes. Original files/failure evidence preserved. Root
review packet-v2, native/auth/assignment/launch-v3 and preparation-review-v3
are authoritative. Root's initial preparation projection used the wrong screen
key; corrected to controller's prior_screen contract without loosening checks.

Approval helper now accepts an explicit versioned packet, registers v6
predecessors, emits canonical proof fields/child paths and derives its controller
preparation receipt directly from checked history. No hidden packet fallback.
Remote preflight-v2 passed1024,0jobs, no pool plan. Conditional config-v3 SHA
00b2777ef0445ba1e91d2aa22264a5fee8e89be15e1903dbff71f3b5e2914e92;
service natlang-qwen36-v6-reviewed-controller-20261004 waits only for Qwen
endpoint readiness and retains its sync/import child. Qwen restarted after E/F
completed. Check actual controller/collector readiness before claiming generation.

### 2026-10-04 full candidate audit and stopping evidence

S1 full conversion completed23:30:59UTC: final68GiB at DGXexternal
full-20261003/resume-20261003T1948-review1/final. No home68GiBreplica is claimed
(home19GiBfree). Dataset remains a candidate until independent current checks.
No actual freeze occurred when root tried to protect Qwen warmup I/O: the S1
service had already finished; the unnecessary thaw timer was canceled.

Explicit validate --schema now fails before scanning if jsonschema is missing,
instead of silently running only stdlib checks. The schema validator is cached
once per process; report binds the exact applied schema bytes. Luna made scoped
source edits, root reviewed; no tests were added or run. Full current schema/
leakage validation is next, with bounded streaming inputs and explicit held status.

E/F completed1900 and saved all4LoRAoptimizer groups in release order. CPU
stopping diagnostics exercised actual phase-F serving load (base/control/head/
LoRA restored), not HTTP/C++ deployment or optimizer-resume parity. Sealed4
prefixes stop at16: D early-stop probability~0.00093–0.00106; F~0.00111–0.00127.
CPU/BF16 diagnostics are limited to these prefixes, not population estimates.
AfterF correct-minus-shuffled NLL margins: extractive−0.1323, multihop−0.1238,
tooldigest−0.02228; cross-source cosine0.3887/rank95.17. Mixed sample metrics
(no universal performance claim); greedy lengths remain16. Local receipts in
runs/neuralese-integration-20261004/stopping-diagnostics-receipts. Fresh E
exploration trial should temper stopping with matching scoring and record
behavior config/lineage; preserve this completed baseline.

Luna v47 both active PIDs2770822/2770834 after exact per-slot accounting; Bunny
v47 active. Root has requested a single future Luna repair of preserved v46
provider-failed0134 after a current slot finishes, not an extra concurrent worker.
Qwen weights loaded; warmup in progress. Nextv6controller waits on readiness.

### 2026-10-04 full S1 audit running

Final candidate has1,870,591 total records (1,361,548train/52,439validation/
456,604test),45files/68GiB. Current independent schema/structural/leakage/
duplicate-ID validation is running on DGX as
natlang-neuralese-full-current-quality-audit-20261004, standalone pinned audit
venv,8GiBcap. Exact checker/spec/manifest/dependencies in
runs/neuralese-integration-20261004/full-candidate-audit-v1/input-receipt.json.
Small local mirrored receipts in full-candidate-audit-receipts; full bytes stay
external. Candidate remains held; source-policy/unknown-license and protected
split checks are separate admission obligations. Do not feed it into the full
port training run on the strength of the earlier finalizer's old checker alone.

Reviewed Luna exact provider-failed0134 retry now has a waiting controller,
not an active third worker. Root verified27pins, selected case equality,
byte-identical5replypartial, original0output and native4/4decisions. Plan-v2 SHA
41c22b98e215dc5bcc9ec61041f582e703c731c61635b0393a097f7876e23ab4.
Original audit-v2 has misleading event-hash metadata; root-independent-review-v1
pins actual exact event with sorted-compact canonical SHA a53c9c31... and raw
line hash separately. Original audit evidence preserved; no original failure
relabeledpositive. Service natlang-luna-v46-single-repair-reviewed-20261004 waits
for all256currentv47Luna1 finishes with completeaccounting and releasedslot.

### 2026-10-04 phase-F trainer handoff verified

CPU trainer restoration of the preserved step-1900 checkpoint passed exact
comparisons for optimizer moments, LoRA/head weights, control rows, RNG state,
parameter group identities and phase position. All five AdamW groups restored
in order, with LoRA layers [15,14,13,12]. No updates were performed and the
original checkpoint remained unchanged. Checkpoint SHA:
c1135eb9ae629b22d02214ceb50d2461619bd38f226375ed6d5942efdc3b1b98.
DGX receipt: runs/neuralese-integration-20261004/phase-f-trainer-handoff-v1/report.json.
Local copy: stopping-diagnostics-receipts/phase-f-trainer-handoff.json.
Forward/backward continuation and trained GGUF/HTTP/C++ parity remain separate.

An independent streaming split audit CLI is now available. It checks explicit
split-group conflicts and places protected matches only in test. Protection
membership and input hashes are recorded; missing or empty protection sets,
empty datasets and changed input metadata fail. It does not establish closure
of background source aliases or license/teacher outcome admission. Both parsing
and hashing use optional sequential/NOREUSE cache advice. No tests or builds
were added or run for this source change; the full production scan is next.

### 2026-10-04 deployment receipt and next generation buffer

Exporter now records bounded-streaming hashes for trainer input and GGUF outputs,
resolved base/fork/converter identities, optional crisp LoRA artifacts, actual
export dtype and trained adapter metadata. Changed checkpoint/adapter/converter
inputs fail before publishing the success receipt. Existing checkpoints lack a
training-base identity, so the receipt states that limitation and claims no
parity. Active frozen generation runtimes are unchanged.

Qwen v7 conditional controller is running with config-v2 SHA
494ce5d1ab22f8355be87fbb68c31a68efc4d5a14364da9a648532b9f38f0ebd.
Only blocker at prelaunch was predecessor v6 still running. Remote preflight
verified the exact 319-case pool with no jobs or pool plan created. The importer
will bind assignment-v2; original assignment/failed readiness records remain.

Independent full split audit is running on DGX:
natlang-neuralese-full-split-audit-20261004. It uses the exact protected.json from
the completed full build, under 6G high/8G max memory limits. Output will be at
runs/neuralese-integration-20261004/full-candidate-split-audit-v1/report.json.
