# Neuralese programme: handover, 2026-10-03

State at the end of the first implementation session. Read [README.md](README.md) (decisions and stage graph) and [DECISIONS.md](DECISIONS.md) first; the stage plans S0–S8 in this directory are current. The source design documents are in [sources/](sources/) and are inputs, not the spec.

## Current status — 2026-10-04 transition to crisp skill self-improvement

This section supersedes historical state tables below. The owner requests a rich
corpus of real self-improvement through crisp skill editing, with measured
improvement on separate query tasks. General task generation should finish at
reviewed queue boundaries after self-improvement cases and collection are ready.
Do not keep creating general refills automatically. Preserve all negatives,
partials, journals, checkpoints and failure/success relationships.

### Neuralese delivered

- S1 full conversion: 1,870,591 records / 45 families / 68GiB, exact manifest
  2bd25a842abf8e715272cec15e221234a8200e6a4e5c7a083bc573645aa961f9.
  Structural, group-split and protected-index audits passed. This is a candidate,
  not an admitted full training set. Train gold/checked654999; teacher705705 and
  failed844 remain separate.
- S3 resumable A–F experiment completed at global1900. E had almost no stopping
  exploration; F has mixed small-family gains. Original AdamW lineage preserved.
- S4 trained F export, ordinary CPU HTTP, controlled block write/read and two-turn
  TypeScript typed-store transport work. Missing BOS fixed in private llama fork
  https://github.com/werg/llama.cpp-neuralese at023131332.
- DGX clean cache reclamation now covers finalized data and immutable exports;
  active model allocations, mutable runs and queues are untouched.

### Work still required

1. S1 concrete outcome/checker and source admission, exact document/episode alias
   closure, heldout preservation and v13 compiler migration plus execution replay.
   Existing group audit alone misses content aliases. Under-three overlap scan
   found611 two-group source digests; fixed-group membership scan is running.
   Higher-frequency/semantic aliases and shared-background policy remain open.
   Upstream verification can be valid; distinguish it from local execution and
   do not blanket-exclude null revision metadata or teacher sources.
2. S2 implement and exercise authoring collection, paired starting/revised query
   evaluation, transfer and skill-ablation checks, journals/resume, and publication
   to the training builder. Current skill runtime and episode/repair builders exist;
   sample episodes are not a collected improvement corpus. This is the immediate
   priority. Author sees support evidence only, never sealed query/transfer data.
3. S3 fresh stopping-exploration experiment with matching behavior probabilities,
   ordinary Natlang replay of F, autonomous task quality and family-level review.
   Full-run loader must stream with bounded lengths and deterministic resumability.
   Add new-run Muon/AdamW parameter policy and optimizer-state checkpoints; do not
   reinterpret old AdamW checkpoints. Select/merge exact crisp base using execution
   evaluation as well as loss. See S3_FULL_RUN_HANDOFF.md.
4. S4 complete trained Python/C++ numerical and GPU parity, SSE/gradient/optimizer
   protocol paths, remaining runtime conformance/second-order-gradient limitations,
   browser wllama/OPFS and vLLM integration. Controlled transport is not autonomous
   quality or complete parity.
5. S5 program-level graph replay training and core operators; S6 soft skills and
   learned updaters; S7 task RL; S8 final student selection/publication. These have
   plans and partial foundations, not completed end-to-end training runs.

### Active compute and stopping policy

Pop-OS trains the full crisp LFM2.5-350M Muon run, step9020/13165 at04:43UTC,
zero skips; estimated ~14hours left at recent speed. DGX Qwen v10 generates;
Luna v50 two slots and Bunny v51 run locally; approved Bunny v52 is waiting.
Before redirecting any provider, review current authority and terminal accounting.
Stop general generation by draining reviewed queues; cancel a waiting successor
only after verifying it has not started. Reuse freed capacity for skill episodes.
Current main training continues; checkpoint selection waits for execution evidence.

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

### 2026-10-04 01:35 UTC deployment fix, split audit, and worker continuity

The independent full split audit passed 1,870,591 records: 985,434 source groups,
zero cross-split conflicts, and zero protected matches outside test. Its 81,471
protected matches are in test. Local receipt:
`runs/neuralese-integration-20261004/full-candidate-split-audit-report-v1.json`.
This does not settle source-policy, background alias closure, or teacher outcome
admission; no automatic promotion of the full candidate corpus occurred.

Phase-F checkpoint step 1900 exported successfully to DGX external storage at
`data/neuralese/exports/phase-f-20261004-v1`. Model GGUF SHA
56ba15f94301717a5e9ae9ed1a0f15310108346fb63b088ce2938a1d9463c8b0;
heads GGUF SHA a2c8facf7cdb27fab050eb3e28e6fefc03a75ad7c7f67f40ed67a5a5ccd5cdbe.
The first ordinary C++ HTTP response was empty. Diagnosis reproduced the failure
in Python by omitting BOS: the merged F model predicts EOS without BOS and `ok`
with it. The server now adds vocabulary-required BOS once before first prefill,
without duplicating an explicit BOS. Private fork commit 023131332 is pushed and
present on both hosts. CPU and CUDA server builds completed; actual HTTP checks
used CPU only. The same ordinary request now returns `ok` with 15 prompt tokens.
A controlled write/read request exercised a 16-position, width-1024 block and its
safetensors download. These checks establish only the exercised deployment paths,
not autonomous quality or full Python/C++ numerical parity. TS transport remains
unverified. Diagnostic server was stopped to release RAM; receipts are under
`runs/neuralese-integration-20261004/phase-f-http-v1`.

Bunny v49 root review independently verified all artifact pins, exact complement
to v48 within Qwen v6, zero history ID/payload collisions, and the native proof.
A stale runtime manifest hash in the proposal was corrected only in a new approved
plan; originals are preserved. Approved worker SHA
6bd382145613c033290055ef6bbe8d9837f5bb9b8743284aed842a185dabc2ad;
rollover SHA c0dd266b0c174035c6f617bf2824f314da1fb6f603ed4618f93a9995647ba7bf.
`natlang-bunny-v49-reviewed-rollover-20261004` is waiting for exact v48 completion
and released worker ownership. It does not add a parallel provider worker.

The 32-case balanced single-source wrapper prototype retains exact input/gold and
source metadata, eight cases per workflow family. Initial prompt tokenization fits,
but full file-observation token budgets, native admission and fresh history review
are still being prepared. Do not promote initial-prompt fit to a full budget proof.
Main local Muon training reached step 8040/13165 with zero skips; the fixed 128-case
held-out token loss at step 8000 is 0.8045394. Execution quality remains separately
measured. Qwen, Bunny and both Luna workers remain active.

### 2026-10-04 01:37 UTC remaining reclaimable cache fixed

The timer covered raw datasets and Qwen weights but omitted the completed 68 GiB
final corpus and immutable phase-F export. Added those two exact directories to
both the checked-in systemd template and deployed DGX service. No mutable queues,
training outputs, unfinished finalizer state, broad volume roots, or global cache
flush were added. Previous deployed service saved in the local DGX receipts.
The existing age guard and per-file POSIX_FADV_DONTNEED remained unchanged.
An actual cleanup increased DGX MemFree from approximately 3.6 GiB to 19 GiB,
reduced cache from 31 GiB to 15 GiB, and left Qwen generation active. MemAvailable
remained about 33 GiB and swap about 414 MiB. This reclaims clean file cache;
active Qwen model/KV allocations remain. Future finalized immutable snapshots
must be added explicitly when their audits create substantial cache pressure.

### 2026-10-04 01:39 UTC trained TypeScript handoff exercised

The existing DGX compiled `neuraleseServerModelTurn` and `NativeToolAgent` completed
a two-turn controlled session against the trained phase-F C++ CPU server. The
server-written literal became a typed `Neuralese<string>` return reference; local
and remote stores agreed on its content ID and 16 x 1024 payload (65,536 bytes).
Two transport exchanges were recorded. Receipt `phase-f-http-v1/ts-handoff-report.json`
pins the compiled artifacts used. This is controlled transport/runtime verification,
not autonomous output quality or complete numerical parity. Diagnostic server was
stopped immediately afterward; Qwen serving continued. No test suite was run.

### 2026-10-04 02:30 UTC cadence sweep and corrected source expansion

Actual interval 01:40:26 to 02:30:28 UTC, with a brief rejection-review interruption
followed by return to sleep. Qwen v7 completed 319 imports: 306 admitted and 13
semantic wrong returns. All 13 are reviewed separately; file/runtime checks pass,
with one recovered filename typo. Bunny v48 completed 512 and v49 is generating.
Both Luna v47 slots subsequently completed 256 each; the gated single repair
started in slot 1. Root is preparing independent successor slot plans.
Main local training was at step 8360/13165, zero skips, GPU fully occupied.
DGX cache remains under control: around 21 GiB free, 13 GiB cache, 412 MiB swap.

Full S1 outcome aggregation passed exact file/manifest/protection identities across
all 1,870,591 records. Labels: 983,324 gold, 95,292 checked, 790,876 teacher,
1,099 failed. In the train split, gold+checked total 654,999; teacher 705,705;
failed 844. Labels are not synonymous with final source-policy admission. Optional
missing upstream revision metadata is not itself a blanket exclusion reason.
Exact source/input identity and background alias review remain distinct checks.

Sealed v42 has only seven source/compiled changes relative to v41: per-file/key
and whole-conversation guidance plus bounded parsed provider diagnostics. x64
manifest 8593e73a210d64f21ddf21008a83d68435078e20da500c9d981334d5845f3e8c;
ARM manifest 3a507ea98deb4c7fd8f1c1ff4fd1e04dff15495c4476b44a91ae48ef9001ecb1.
ARM local and deployed physical closure checks passed (20,706 files, 22 internal
symlinks). Old dependency versions/admission implementation are retained; TypeScript
compile passed, no test suite run. Active queues have not been changed in place.

New balanced one-source directory supply is 860 cases, 215 per workflow family,
limited by security source eligibility after train/protected/type checks. These
are new task variants of prior sources, not fresh source coverage. Full observation
budgets fit context 32768 with actual 8191-token response reserve; worst total
22475, margin 10293. Initial planning-only budgets were insufficient for seven
invoice samples at context 16384, so they were not treated as launch proof.
The first pool draft invented unsupported curriculum.slice=folder_one_source;
runner join caught it before launch. Failed v1 is preserved; v2 preserves the
source's valid slice and is still awaiting final native runner/root approval.
Do not claim admission-only proof establishes runner identity validation.

### 2026-10-04 future input provenance

Future finalizer manifests record exact normalized input JSONL SHA/bytes/rows,
protected-index hash and scoped finalizer/CLI code hashes. All three existing
passes must consume matching bytes; stat/hash changes abort output publication.
No old candidate was rebuilt or retroactively admitted. Full metadata scan of
1,870,591 rows matched all 45 output identities; absent optional upstream
revision/ID fields alone are not blanket holds. Background source membership/
transitive alias receipts remain missing and need explicit policy/evidence work.
See runs/neuralese-integration-20261004/s1-provenance-alias-closure-review-v1.md.


### 2026-10-04: soft discovery metadata

User decision: descriptions and summaries of soft skills should normally be Neuralese values; IDs and routing metadata stay crisp. Implemented `SkillDescription = string | NeuraleseRef`, optional `summary`, strict `Neuralese<string>` validation (dialect-qualified allowed), and sentinel/content-part rendering in the progressive discovery listing. Text remains available for crisp skills. Tests verify actual runtime delivery to a Neuralese-capable driver, explicit rejection on a text-only driver, immutable metadata, invalid types/IDs, and the `.nz` host-hook path. Future soft-skill generators must produce/train soft discovery values by default and persist their blocks with the asset. Complete `.nz` context/body loader wiring remains pending; the new API does not claim that full integration. Markdown skill descriptions can already carry real stored soft references.


### 2026-10-04: optimise discovery as well as skill bodies

User requirement: self-improvement must tune descriptions/summaries for both crisp and soft skills so agents use the necessary and helpful skills, neither missing useful ones nor using irrelevant/redundant ones. Crisp authoring now explicitly permits and prompts metadata-only improvements, applicability/non-applicability descriptions, and consistency with bodies. No universal skill-count penalty is appropriate. Remaining collection/evaluation work: mixed/distractor libraries and metadata-only cases; distinct discovery/body-read/helper-call accounting; paired metadata/body ablations and disabled-skill contribution checks; cost comparison at preserved quality; soft-description gradient integration. Query remains sealed, and current positive quality gates do not prove optimal skill selection. Transfer admission also now requires transfer gates to pass, not merely nonnegative quality effect.


### 2026-10-04: graded families, frontmatter types, S3 trainer prerequisites, laws

- **Memory incident.** 08:16 global OOM (unified memory: vLLM ~80 GB plus an extra Neuralese server) killed the
  Qwen v10 pool (655/860, resumable with `root-launch.sh --resume`; restart left to the owner), pop skill pilot v2,
  the optimisation-v4 crisp collection and the soft-skill SQL run. Rule: one memory pool; check MemAvailable,
  cap torch with `set_per_process_memory_fraction`, avoid CPU→CUDA duplication.
- **Graded skill families** (`ts-host/src/skills/graded.ts`, S2 §5.1a): answer token F1, ranking NDCG,
  assignment accuracy, call F1, choice Brier, besides SQL F1 and Python tests. Corpora on the drive under
  `crisp-skill-self-improvement-20261004/`: `graded-v3` (HotpotQA answer/support, knights, xLAM, WorldTree; 290
  episodes), `sql-v1`, `code-v1`. `graded-v1/v2` are superseded (targets do not load). None collected yet.
- **Soft-skill baseline** (`ts-host/scripts/skills/soft-skill-baseline.mjs`): 8 SQL train episodes, support-only
  Adam tuning cut sealed-query NLL 2.33 → 0.94 (text-init 2.38, no skill 2.33). Transfer NLL and sampled graded
  quality (`--sample true`) are implemented but not yet run; that control decides whether this is a family skill.
- **`.nl` frontmatter**: `args`/`returns`/`types` are TypeScript text read verbatim (SPEC.md); all 930 distinct
  frontmatters in the v13 teacher data parse identically.
- **S3 prerequisites** (S3_FULL_RUN_HANDOFF.md): phase-E exploration with importance weighting, Muon/AdamW port
  optimiser policy, streaming loader with `--stream`. A CPU smoke of stream+Muon+exploration at scale 0.01 runs;
  its gradient norms explode from phase C (untrained heads, immediate full unroll at that scale) — compare
  against the same smoke with AdamW before trusting Muon for a real run.
- **Law objectives**: `objectives.law(name, …)` for the six S0 laws, measured through `read` as cross-entropy of
  the right side's readout. `lib.empty()` blocks are now registered constants that upload paths can supply.


### 2026-10-04 afternoon: length supervision, full-depth stop, decision data

- **Block length and stopping** (S3_PORT.md §3.1): the pilot's stop head learned the count (all A/C spans 16 → every
  D–F block 16). Now: variable span lengths (`--span-lengths`), phase-D lengths from source size
  (`--tokens-per-vector`, teacher-forced, stop BCE), and a full-depth stop source (`--stop-source final`) deciding on
  completed states with lookahead, exact by causality, in training, `write_block`, the server step writer and grad
  replay. Not yet in the C++ fork. Not yet trained: the next S3 run should use all three.
- **Soft-skill control** (`soft-skill-sql-v4`, 16 Spider episodes with transfer): support-only tuning lowers query NLL
  2.33 → 0.97 but transfer NLL 2.33 → 1.18, so most of the gain is generic answer-format priming (≈0.2 nats is
  family-specific); sampled SQL quality stays ≈0 in every arm with the 350M phase-F model.
- **Graded crisp pilot** (`graded-pilot-v1`, Qwen3.6 executor): Spider episodes keep their baseline (already 1.0);
  KodCode has headroom (0.83) but searches end `incomplete` when the author's per-step turn/token/wall budget is
  exhausted (72 of 400 model calls used) — the step budget, not the search budget, limits code episodes.
- **Decision data** (`runs/dgx-development-generated/decision-data-20261004/`): 58,500 typed cases, 624 episodes;
  Decider 2B labels in progress; Clef-flash downloaded for a second teacher pass (needs ~19 GB; run when memory allows).
- Gradient replay differentiates terms one at a time (same gradient, a fraction of the memory); the reference server
  takes `--memory-gb`.

### 2026-10-04 evening: system improvements 1–7

1. **Memory admission** (`plans/MEMORY_ADMISSION.md`). `scripts/memory_ledger.py run` admits heavy jobs against
   MemAvailable minus unspent budgets of running claims, launches them as user units with kill priority by class
   (experiment 900, collection 600, service 300) and `NATLANG_CUDA_MEMORY_GB` (read by the labeler, the Neuralese
   pilot and server). `natlang-memory-guard.service` (enabled) stops an admitted unit over 115% of its budget, CUDA
   included, or the lowest-priority newest unit when free memory drops below 8 GB. `adopt` registers units started
   before the ledger. `scripts/systemd/natlang-campaign-resume@.service` resumes a campaign after an abnormal end;
   installing it is the owner's call.
2. **Headroom screening** (`ts-host/scripts/skills/screen-episode-headroom.mjs`, S2 §5.1b): starting context on
   support cases only, per executor, concurrent, with a family probe that stops screening saturated families.
   First screen running for Qwen3.6: `crisp-skill-self-improvement-20261004/headroom-qwen36-v1/`; knights
   kk-people2/3 are saturated (support quality 1.0 on every episode so far).
3. **Decision readout** (spec/SPEC.md "Decision readout"). `readout: decision` (or model config
   `decisionReadout: 'finite-returns'`) answers a finite-typed call by scoring every value as the whole reply to a
   compact opening; the distribution is the trace event `decision_readout`. Drivers: vLLM via `prompt_logprobs`
   (0.5 s per call on Qwen3.6 with the compact prompt, 2.2 s with the tool manual), the reference server's
   `/v1/neuralese/decide` (prompt once, options from its cache) and the fork's same endpoint.
   `objectives.decision` trains it with log loss, Brier or RPS; the grad term `decision` and
   `GradSession.decision_backward` (one option's graph at a time) back it. Distillation:
   `export-decision-prompts.mjs` (the runtime's exact prompts) → `natlang_neuralese.train.decision` (LoRA on
   LFM2.5-350M; base, gold and Decider arms queued as `natlang-readout-distill-v1` under
   `decision-data-20261004/readout-distill-v1/`).
   Result (26 families, 40 held-out cases each, 600 steps, mean): base log loss 4.97 / top-1 .262; gold 1.086 / .579;
   Decider distributions 0.981 / .593. The Decider itself: 0.695 / .719. Distilling the teacher's distributions
   beats gold labels slightly and the 350M readout closes most of the gap to the 2B teacher's log loss.
4. **Skill-specific soft skills** (`ts-host/scripts/skills/soft-skill-decision.mjs`): arms none, text-init,
   generic (pooled over families), tuned, specific (generic plus cross-entropy against the generic readout off its
   family, a bounded hold term); specificity = query gain over generic − transfer gain over generic. Found and
   fixed on the way: concurrent learning objectives could take each other's recorded turns. Objectives now
   capture per output when given a function and refuse overlapping promises; the SQL baseline passes functions.
   Earlier SQL soft-skill results ran objectives concurrently over promises: any mismatch would have surfaced as a
   `learning-no-turns` error, which none of those runs reported, but treat their per-case numbers with that caveat.
5. **Shortcut checks**: `natlang_neuralese/data/shortcuts.py` (length profile, count-only stop hazard and its stop
   recall) runs at each trainer phase (`shortcuts.json`, warnings, `--fail-on-shortcut`) and logs
   `stop_bce_count_baseline` next to `stop_bce`. `scripts/audit_decision_shortcuts.py`: prior and length-quintile
   baselines per decision family (`decision-v1/shortcuts.json`): spam and toxicity are prior-dominated (a constant
   matches 90%/91%), trec-question's length carries label information.
6. **Episode library and gate** (S2 §5.1c): `scripts/episode_lib.py`; graded and decision builders reproduce their
   packets byte for byte and gate themselves; `audit-episodes.mjs` adds manifests, target loading, gold at its best
   score, required transfer and constant-answer warnings. Found: 11 BIRD gold queries over the scorer's timeout
   (builder now checks under scorer conditions; `bird-v2` building), 3 WorldTree episodes with one answer letter per
   query set (`graded-v4` rotates labels).
7. **Fork conformance**: the fork's writer implements the final stop source and count-free stop heads (projector
   keys `neuralese.stop_source`, `neuralese.stop_position`; parity CPU and CUDA), the fork server serves
   `/v1/neuralese/decide`, and `tests/neuralese/test_server_conformance.py` runs both servers on the same weights
   (info, forced write, read, decide, refusals; both stop sources): 10 pass. Fork at `c0313100a`.

## 2026-10-04 night: learning continuum M4 starts; results in

- **Adapters (M4 first deliverable; LEARNING_CONTINUUM §13 status).** `model/tiny_adapters.py` (`xs`, `tiny`;
  spec = block dialect `adapter/1;base=…;kind=…;r=…;u=…;layers=…;targets=…;seed=…`), reference server: per-row
  hooks, `x_natlang_adapters`, `POST /v1/neuralese/adapters`, adapter leaves and Adam steps in grad sessions.
  TS: `Adapter` type, `withAdapters`, `adapters.create`; `.nz` round trip. Tests: `tests/neuralese/test_tiny_adapters.py`
  (zero = base, equals merged LoRA, mixed batch = alone, Adam lowers loss) and the adapter case in
  `neuralese-learning.test.mjs`. Fork `d03aa6b65` refuses adapter requests (501); conformance checks it.
- **Soft-skill decision v3** (`decision-data-20261004/soft-skill-decision-v3/results.jsonl`, 24 query cases per
  family; query quality none / generic / tuned / specific): specific improves on generic in 7 of 14 families and
  ties in 3 (sms-spam .745→.961, toxicity .824→.961, helpfulness .641→.796, trec .674→.806, ag-news .735→.811,
  sst5 .805→.859); it is lower on boolq, subjectivity, app-stars, paws and vitaminc. Specificity (query gain over generic − transfer gain over generic) is positive in 12/14;
  the tuned arm without the hold term goes negative on subjectivity (.418) and vitaminc. The hold term is doing
  its job. Prior-dominated families (spam, toxicity) need the prior baseline beside them before claims.
- **Headroom screen v1** finished knights and xlam only; the other families failed on a mid-rebuild `dist/` and
  Node 18 under systemd. `headroom-qwen36-v2` (queued through the ledger, frozen dist snapshot, Node 24) runs
  worldtree (graded-v4), hotpot ×3, bird-v1 and decision into `screen-2.jsonl`.
- **bird-v2** gate: 6 transfer gold queries failed `reference_executes`. Five of them time out at 20 s now,
  although each passed the 2.5 s build check. That suggests external-disk contention. Re-gate when the disk is
  quiet; if they still fail, drop slow gold at build with a lower bound.
- **M0 done** (LEARNING_CONTINUUM §13 status): improvement-step records, converter, direct writers, the method-arm
  runner and `objectives.conditionedDistill` with the privilege rule. `natlang-method-arms-v1` (14 decision
  families × 7 arms, Decider-2B teacher) is queued through the ledger against the restarted 8094 server, which now
  serves adapters. Results go to `decision-data-20261004/method-arms-v1/`.
- **Decider-2B teacher baseline** (`decision-data-20261004/baseline-decider-2B-v19.json`, all 58.5k labels, held-out
  quality): ≥ .90 on ag-news, dbpedia, language-id, sms-spam, sst5, toxicity, yelp, formality. It is weaker on
  app-stars .758, newsgroups .712, emotion .721 and sarcasm .751. It beats the 350M soft-skill arms on most
  families. Clef labeling is still waiting for 26 GB of admission.
- **Guard incident (15:56).** My unadmitted test runs and probes, which spawn CPU model servers, pushed free memory
  below the floor. The guard then stopped the admitted runs: method-arms-v1 (one family done), readout-distill's
  teacher arm (step 390 of 600, lost; `train.decision` now checkpoints every 50 steps and takes `--resume`) and the
  8094 server. Rule now: anything that loads a model goes through the ledger. Relaunched: the 8094 server,
  `natlang-method-arms-v2` and `natlang-readout-distill-v1b` (queued; `run2.sh`, teacher and gold arms with resume).
- **Adapter learning rate.** On sms-spam, `xs` rank-4 adapters reach query .959 at Adam lr 0.05, against .288 at
  0.01. method-arms-v2 uses 0.05. Deltas: `deltas.learnMerge` and `deltas.interference` are in and tested.
- **BIRD on NVMe.** The 79 BIRD databases the bird-v1/v2 episodes use (32 GB) are now at `/home/werg/data/bird-sqlite`.
  Gold SQL from the full external disk timed out nondeterministically: 6, then 43 gate errors on the same packet. From
  NVMe the bird-v2 gate passes with 0 errors (`bird-v2/sql-episodes.audit-nvme.json`). Score and collect with
  `--database-root /home/werg/data/bird-sqlite`; builders still read the archive, since they select among all databases.
- **Whole-codebase test pass.** TS: 1013 pass, 0 fail, plus native conformance 22/22. Python: 528 pass. Fixed on the way:
  - a forced compaction turn lost its notice when automatic shortening ran;
  - the audit tests depended on the working directory;
  - student-improvement rounds exec'd a bare `python`, but the host has only `python3`.
- **Robustness.** The headroom screen and the memetic author wait out executor restarts; `train.decision` resumes from
  checkpoints. The Qwen container was resized by the owner at about 16:20, which freed memory.
- **Adapters bound by context.** An `Adapter` context item applies to its own function's turns only and is not rendered
  in the scope. There is also a crossing test (`serve/crossing.py`). With untrained heads the block cannot change (the
  content projection is zero), so the check matters for trained heads.
- **Method arms v2** (adapter lr 0.05), first families: sms-spam: soft .955, adapter .959, joint .959; sst5: soft
  .835, adapter .826, joint .846 (teacher .928). Memetic v1 is running on 8095 with 6 families.
- **Method arms v2 complete** (`decision-data-20261004/method-arms-v2/`; 14 families; 16 support, 24 query, 8 Adam
  steps; artifacts in `artifacts.nz`). Mean query quality:

  | none | soft-init | soft-gold | soft-teacher | adapter-gold | adapter-teacher | joint-gold | Decider-2B |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | .516 | .515 | .774 | .766 | .759 | .781 | .792 | .857 |

  Joint soft skill plus adapter is best on average. Where gold training on 16 cases overfits (paws, vitaminc, where
  adapter-gold falls below no skill), distilling the teacher's distributions holds quality: adapter-teacher .716 and
  .596, against .507 and .490 for adapter-gold. Every arm stays below the teacher.
- **Memetic v1, first family** (sst5): best .766, below soft-gold .835. Its plain seed is already worse than no skill
  (.672 against .780), and its long guidance texts may confuse the 350M model. To assess once all six families are in.
- **Adapter projection P, stage 2** (`projection-e2e-v1`; P trained end to end on 10 families; held-out families
  adapted for 16 steps). Code through P / direct adapter / code through random P / none:

  | family | code via P | direct | random P | none |
  | --- | --- | --- | --- | --- |
  | sarcasm | .705 | .758 | .385 | .379 |
  | helpfulness | .735 | .812 | .422 | .426 |
  | trec-question | .704 | .718 | .714 | .696 |
  | ag-news | .365 | .829 | .740 | .682 |

  P's code space transfers: +.3 over none where a random P gains nothing. It is still below direct adapters, and
  ag-news (4-way) collapses. The training families are mostly binary, so the next run needs more and more varied
  families (topic and intent choices), more steps, and codes written by the model rather than free codes.
- **Soft system prompts and Neuralese conversion (owner direction, decisions 40 and 41).**
  - Runtime prompt pieces have stable IDs (`ts-host/src/native/system-prompts.ts`): interpreter and its depth-limit
    variant, function tools, directory reducers, generation guidance, approach, decision, the predicate prompt's fixed
    lines, compaction notice, handover frame, automatic note, last-turn notice.
  - Under a Neuralese driver, a bank (`neuralese.systemPrompts`, or the `withSystemPrompts` scope) replaces each
    piece's text with its block. `withSystemPrompts` is differentiable, so self-improvement can tune the prompt.
  - `system-prompt` is now an improvement-record artifact kind, and `run-method-arms --arms prompt-gold` tunes the
    soft decision prompt per family.
  - Text-initialised bank: `/mnt/external/natlang-development-data/runs/neuralese-system-prompts-20261004/bank-text-init-v1.nz`,
    21 pieces, built by `scripts/neuralese-system-prompt-bank.mjs`.
  - Base training: `train.decision --soft-prompts BANK --prompt-lr` trains the pieces in its prompts as leaves and
    saves `system-prompts.nz`; `--rank 0` trains the prompt alone. Python helpers are in `natlang_neuralese/prompt_bank.py`.
  - Converter: `scripts/neuralese-convert-trajectories.mjs` (format in `spec/NEURALESE_DATA.md`, "Neuralese conversion").
    Over the v13 teacher corpus (111,301 records, written to `/home/werg/data/neuralese-converted/v13-20261004/`, 2 min):
    - every system message is soft: 141,488 prompt sites; 66 older system-prompt versions plus 9 current pieces in
      `pieces.jsonl`;
    - 4,051 handover writes and 3,100 pinned-note reads; every read has its write;
    - counted, kept exact: 557k tool outputs (no consumer trace), 110k instruction sites and 4k `nl` literals (later
      curriculum steps), 678 turn-count notices.
  - Not yet consumed by a trainer. The S5 replay trainer must still turn `soft` parts into leaves and `$write`/`read`
    into producer and consumer views, with the crisp note as the teacher's view.
- **Projection e2e v2** (15 training families, 5 held out, 1500 steps). Code via P / direct / random P / none:
  sarcasm .46/.68/.39/.38, helpfulness .74/.80/.42/.43, trec .42/.69/.71/.70, ag-news .53/.85/.74/.68, pubmedqa
  .70/.75/.07/.05. More families did not fix the multi-way collapse: on trec and ag-news a free code through the
  trained P is worse than no adapter. `projection-e2e-v3` (running) trains P on codes the model writes from the
  family's instructions and cases (`--codes written`), followed by the same run with token embeddings of the prompt
  (`--codes embedded`, control). Its arms are zero-shot, shuffled (another family's code) and code-init.
- **Delta projection D, stage 1, fails** (`delta-projection-v1`; 70 recorded soft deltas from method arms and memetic
  refine and merge). Relative reconstruction error: train .81, held-out .96, random-D control .93. Adam deltas of soft
  skills are close to sign noise in all 1024 dimensions per position, and a shared D with a 256-wide bottleneck cannot
  reconstruct them. Reconstruction is the wrong target: D should be trained functionally (stage 2: decision loss after
  applying `D(code, base)`), as P is. Also, the stage-1 fit runs on CPU and took 6 h; use the GPU next time.
- **Memetic v2** (quality fitness, 60-word guidance, generic seed, 8-step refine, soft-gold reference on the same
  split). sst5: memetic .718, seed .679, soft-gold reference .671. The same-split reference is far below the
  method-arm soft-gold (.835, which used different support cases), so 24-case query scores are noisy at ±.05. Judge
  memetic only across all six families; the run continues on 8095.
- **Conversion v2 and the trajectory trainer.**
  - `/home/werg/data/neuralese-converted/v13-20261004-v2/` replaces v1. It adds the `compacted-result` piece (22
    pieces; bank `bank-text-init-v2.nz`) and splits tool outputs: 357k copy an exact value into a later turn and stay
    exact, 200k are model-only and are encoded under `--convert tool-outputs` (curriculum step 1).
  - `train.trajectories` trains the soft parameters of converted records (optionally with LoRA) on target
    cross-entropy and reports crisp / soft-init / trained held-out. Handovers are rendered crisp in this version.
    Smoke: soft-init 1.35 against crisp 1.46 on 4 held-out records. A real run needs a budget of about 12 GB: host
    memory plus CUDA reached 9 GB at 6k-token prompts.
  - `grad.py` prompt passes now project logits at the last position only. They used to allocate full-vocabulary logits
    for every prompt position, which ran out of memory on long prompts.
- **Correction: projections need trained heads.** `projection-e2e-v1`/`v2` ran without `--heads`. That is fine for
  free codes, which P reads directly. The first `projection-e2e-v3` written run also lacked them, so every write ran to
  the 64-vector maximum (untrained stop head) with degenerate content. It is kept as `*.untrained-heads` and was
  restarted with the S3 pilot checkpoint's heads. `delta-e2e-v1-written` (D stage 2, running) has heads.
- **Readout distillation, teacher arm done** (`readout-distill-v1/teacher`; LFM2.5-350M + rank-16 LoRA on Decider-2B
  readout distributions, 600 steps of 8, 0.09 epoch). Mean held-out quality over 26 decision families:
  base .453 → .797; Decider-2B itself .848. The student improves on every family, e.g. sms-spam .06 → .90,
  pubmedqa .04 → .77, language-id .15 → .96 (on pubmedqa the base puts .98+ on the same option for nearly every case: collapsed, confident readouts). The
  gold arm is running (step 90 of 600) for the gold-against-teacher comparison.
- **Conversion by use, encode and digest (owner direction; decisions 42, 43).**
  - **Encode** (`/v1/neuralese/encode`, `serve.grad.encode_text`): text into Neuralese in one forward pass through
    the port. The text's token embeddings are supplied at the block positions as in phase A, giving one vector per
    token, with no summarising call. It is now the default initialisation of soft artifacts: the system-prompt bank,
    the method arms and memetic (`--init embed` keeps the old raw token embeddings), the combinator library,
    trajectories and delta_e2e. projection/delta e2e have an `--codes encoded` control.
  - **Converter v2** (`/home/werg/data/neuralese-converted/v13-20261004-v5/`; v1–v4 removed). A site converts when
    it is reused, handed between agents, or large:
    - Instructions: 2,400 of 4,835 instructions texts serve two or more calls, covering 90% of calls. They are shared
      soft parameters (99k sites), plus a 10% coverage share of single-use ones (1k sites).
    - 55k cut-off listing values become digest sites (3.6k nested ones lack the full value).
    - 443k single-use tool outputs stay text. 2.5k printed child-call results are counted as handoffs needing graph
      records.
    - Read parts carry their note (`source`), since the compaction call can lie outside the record.
  - **Digest operator**:
    - runtime: `neuralese.digest` with `serverDigester`;
    - server: `/v1/neuralese/write`, the write procedure at a write site;
    - the `digest` prompt piece (24 pieces now, with `progress-judge`);
    - a Python write-site mirror pinned to `tests/fixtures/digest-site.json`;
    - `train.trajectories --digest written`.

    A large argument is listed as its digest block with "<name> holds all of it"; small arguments stay literals.
  - **Smoke** (S3 pilot heads, 300-record handover subset): 2 notes and 2 digests written by the model and read by
    their consumers. The stop head chose 16 vectors for each digest (the maximum is 32); loss went 3.1 → 1.6 over 3
    steps. Model tests 19/19, including encode, write and the digest fixture. The full TS suite is still to rerun: it
    needs a 16 GB ledger budget, and the earlier attempt never started because of a unit-name clash.
  - **Gaps:**
    - ~~The writer gets no gradient from its readers~~: fixed (differentiable writes, below).
    - Child-call handoffs need graph records.
    - ~~Digests of values over 48k characters are cut~~: fixed (chunked plan, below).
    - The learned updater and the `improve` operator (LEARNING_CONTINUUM §9–10) are not built yet; when they are,
      their bodies start encoded from text like the combinators. `compose` is delta arithmetic and has no body.
    - ~~The C++ fork has neither `/encode` nor `/write`~~: fixed (below).

### 2026-10-04 late evening: the writer learns from its readers; digests have no length cap

Owner: "why does the writer get no gradient? that defeats the entire purpose", and 48k characters was too low a cap.

- **Differentiable writes in `train.trajectories`.** Handover notes (`--handover written`) and digests
  (`--digest written`) are now written by `train.execution.unroll_write` with gradient: the producer's prompt (the
  soft-rendered record up to the write site) is prefilled with gradient, the write is unrolled, and the payload enters
  each consumer as a GradSession leaf. Consumer cross-entropy flows back through the payload into the backbone path,
  the port heads (`--heads-lr`, saved as `heads.pt`) and the soft parameters of the producer's prompt.
  `--stop-pg λ` adds a policy-gradient term for the sampled stop decisions with reward −(loss + λ·length) against an
  EMA baseline; `--detach-write-context` cuts the gradient at the producer's prompt.
  `--distill` adds self-distillation to the crisp record when a payload replaces text.
- **Digest plan, no cap.** `digest.py` holds the plan used by both the server (`POST /v1/neuralese/digest`) and the
  trainer. The window is the model's context (128k for LFM2.5-350M) less the site's own text. A longer value is split
  into token chunks: one part digest per chunk, then a combine site reads the part blocks and writes the final digest.
  `serverDigester` (TS) now sends the whole value with the instructions (text, or the soft `digest` piece) to
  `/digest`; `DIGEST_SOURCE_CHARS` is gone.
- **Smoke** (S3 pilot heads, handover subset, budget 16 GB): writer gradient norm 44 → 35, loss 4.9 → 2.8, peak
  12 GB. Every write had length 16 of a maximum of 32. This is the known pilot artefact, not the new path: the pilot's
  stop head learned the count from all-16 spans (see "Block length and stopping" above; p(stop) before 16 is about
  0.001), so the stop policy gradient sees no length variation until heads from a variable-length pilot are used.
- **Tests:** model 20/20 (`test_digest`, `test_serve` with `/encode`, `/write`, `/digest` and a chunked plan at
  window 16, render, grad, prompt bank); TS 9/9 (`system-prompts`, `neuralese-conversion`).
- **C++ fork** (`8a056e4d0`): `/v1/neuralese/encode`, `/write` and `/digest` (same plan and window rule; the window is
  bounded by the served `-c` as well) and template readout. Conformance on the CPU build, both stop sources: 16 passed
  at `d8068643a` (adds encode with and without context, write at a site, the fixture digest and a chunked digest);
  template readout (written and decoded values) 4 of 4 at `8a056e4d0`. The full 18 at `8a056e4d0` is queued behind
  memory admission (`natlang-conformance-full`).
- **Template readout (decision 44).** Combinators no longer decode freely. A `readout: template` call keeps its
  ordinary opening, and its first reply is forced to `return_result(status='success', value=` cut from the model's
  own chat template (`chat.call_reply`; the C++ server cuts its own). A Neuralese result is written at the value and
  the call closed; other results are decoded from the value on. The reply parses into an ordinary tool call, so the
  agent's return checks apply, and a rejected value falls back to ordinary turns. The trajectory trainer's note
  writes now use the same cut: the LFM2 template quotes `note='`, where the old hard-coded prefix had `note="`. `.nl`
  files may declare `readout: template`.
- **D stage 2, functional, written codes** (`delta-e2e-v1-written`; 15 training families, 1,500 steps; held-out
  readout quality on 5 families, mean): none .446, base skill .473, code adapted through D for 16 steps .692, direct
  skill optimisation for 16 steps .755, the same code adaptation through an untrained D .476, zero-shot (the family's
  own written code) .559, another family's written code .584.
  - D is a usable adaptation space: 16 steps through it nearly reach direct optimisation (.692 against .755), and an
    untrained D gets nowhere in the same steps.
  - The written codes carry no family-specific information yet: zero-shot is no better than a shuffled code. D learned
    a generic decision-improving delta (+.09 over base). That is expected from the pilot writer: its stop head is
    fixed at 16 and its content is barely trained (see the conformance and pilot notes).
  - Per family, the code beats direct on sarcasm (.779 vs .695) and pubmedqa (.706 vs .673) but is far below it on
    trec-question (.571 vs .786, below none).
- **Arms server killed by the memory guard** at 19:31 (5.8 GB over a too-small budget). That ended memetic v2 after
  3 of 6 families and the prompt method arms after 5 of 14. Memetic v2 results so far: sst5 best .718 (seed .679,
  soft-gold .671); emotion .525 (seed .596, soft-gold .491); sarcasm .691 (seed .382, soft-gold .736). The remaining
  families rerun into `memetic-decision-v2b` and `method-arms-prompt-v1b` with a 10 GB server budget.
  - **Memetic v2b** (query readout quality, best / seed / soft-gold): helpfulness .826 / .454 / .550 (transfer to
    app-stars .743 vs seed .510); app-stars .750 / .511 / .714; trec-question .537 / .512 / .539. Best individuals
    came from refine-baldwin and merge; searches took 19–35 min against ~2 min for soft-gold. Across the six
    families memetic beats soft-gold on four (sst5, emotion, helpfulness, app-stars) and ties or trails on two
    (sarcasm, trec-question).
  - **Method arms v1b** (9 families, query quality, 8 steps, encode init): mean none .536, soft-gold .661,
    prompt-gold .662, prompt-teacher .679 (teacher itself ≈ .85). Soft prompts optimised against gold and against the
    teacher are equivalent in mean; the teacher target helps on vitaminc (.698 vs .600). trec-question is the
    exception again: every arm is below none (.694), as with D stage 2. Checked: the labels are the TREC coarse classes,
    correctly mapped. The cause is label shift between two tiny samples: the 16 support cases (first in file order)
    are 5/16 "human beings", 4 description; the 24 query cases are 10/24 description, 2 "human beings". Tuning moves
    mass toward the support distribution and the untuned prior wins on this query set. Class-stratified support and
    larger query sets (≥ 60 for 6 classes) before comparing arms on multi-class families.

### 2026-10-04 night: sizing, guided-generation measurement (owner: proceed in order: sizing, guidance, browser)

Owner on sizing: simplify training by sizing writes from what they stand for; a size may be fed at inference to
generate block-wise with the sketch system, but sizes must never be required.

- **Training** (`train.trajectories --tokens-per-vector R`): a note or digest write is sized from its crisp text (the
  note, or the listing preview the digest replaces) as ceil(tokens / R) vectors capped at the heads' maximum. The write
  has no stop decision, and the stop head is trained on that boundary (`--stop-weight`). Smoke (R = 4, pilot heads):
  lengths 16–32 following the source, loss 5.06 → 1.60 over 3 steps, peak 8 GB.
- **Inference hint** (optional everywhere): `neuralese_length` on chat requests and template readout, and
  `length`/`passes` on `/write`, in the reference server, the fork (`c8d92f02b`) and the TS contract
  (`template.length`). With a hint there are no stop decisions. With `passes` the block is written block-wise
  (`execution.blockwise_sketch`, C++ `nz_write`): all positions go through layers [0, k) at once and the sketches are
  refined by fixed-point iteration. Pass p fixes position p exactly, so passes ≥ length reproduces the sequential
  write; fewer passes is an approximation, to be trained for. Without a hint the stop head decides.
- **Bug fixed in both servers:** at temperature 0 the payload equals its mean, and the content-addressed store
  returned the earlier "payload-mean" block. Every greedy write had therefore lost its write record (stop logits,
  seed). The payload block is now stored first.
- Fork conformance: 20 passed (both stop sources), adding length hints sequential, one pass and all passes.
- **Syntax measurement** (`ts-host/scripts/measure-turn-syntax.mjs` over reduction traces). Post-trained 350M student
  (`student-posttraining-20261004/final-gpu-eval-v3-single-bos/final`, 107 turns):
  - 85% ok;
  - 3.7% eval code that does not parse as TypeScript (mostly `':' expected`);
  - 5.6% unclosed call markup, all runaway line repetition;
  - 0.9% malformed markup;
  - 4.7% plain text.

  The earlier v1 eval had 12% unclosed. The Qwen teacher (2,269 turns) is 100% ok.

  Implication: TypeScript syntax proper is the smaller share. Repetition runaways and missing calls are larger, so the
  check-and-backtrack layer should also treat repeated lines as an error, and the envelope grammar should require a
  call when the turn must make one.
- **Guided generation** (`serve/guidance.py`; C++ `tools/neuralese/guidance.{h,cpp}` with vendored tree-sitter 0.26.0
  and tree-sitter-typescript 0.23.2, the same versions as the Python packages). The request field `guidance` (true or
  settings; the reference server's `--guidance` sets a default; the TS runtime option `model.guidance` sends it only to
  natlang's servers) does three things:
  - **Envelope:** with `require_call` (default when `tool_choice` is "required") the reply is forced to open a tool
    call, and call names are checked against the offered tools or `guidance.tools`.
  - **Line checks:** inside `eval(code=…)` every completed line is checked for repetition (`repeat` occurrences),
    redeclaration (`const`/`let`/`var` of a name already declared at the same indentation in the same block) and
    TypeScript syntax. A tree-sitter error inside the completed text counts; one reaching its end is only
    unfinished. The unfinished last line is checked for a run (one 3–60 character chunk repeated `run`=4 times at
    its end, `x||x||x||x||`), which never completes a line and so escaped the line checks.
  - **Backtracking:** a rejected line rolls back to its start and the token chosen there is banned, with `retries` per
    point, after which the line stands.

  Masking would force unlikely tokens; rejection keeps the model's own distribution except where it is definitely
  wrong. Rollback uses cache snapshots at line starts (Python: immutable caches; C++: copies of sequence 0 to spare
  sequences 3–6) and recomputes the few tokens between. Responses report `x_natlang_guidance.rejections`.
  Conformance: `/v1/neuralese/guidance/check` (the checks over fixed replies) and `/v1/neuralese/render` (rendered
  prompts, tools included) agree; guided generation rolls back the same way on both servers.
  - Prompts render identically in both servers. Greedy free text still diverges between them on near-ties (float
    differences), so free-text equality is not a conformance criterion.
  - Found and fixed on the way: the reference server's streaming sent raw call markup and no `tool_calls` deltas.
    Every streaming client (the eval runtime) saw unparsed calls. The first guided A/B was invalid for that reason.
  - A/B after the streaming fix (student adapter, reference server, 23 cases): TS syntax errors 3.6% → 0.8% of
    eval turns, repeated lines 5.4% → 0; task successes 9 in both arms. The remaining failures were in-line runs
    (now the run check), redeclarations (now checked) and calls with JSON literals (`false`, `null`) in nested
    pythonic arguments, which both servers' call parsers now accept. Fork `cf01643c0`; conformance plus guidance and
    serve tests: 50 passed (native and wasm).
  - A/B rerun on these fixes (same 23 cases; per eval-call turn, unguided → guided): well-formed .856 → .945, TS
    syntax .038 → .009, repeated lines .058 → 0, unclosed calls .058 → 0, malformed 0 → 0 (was .11 in both arms
    before the parser fix). Tasks: 11 complete successes in both arms (9 before the parser fix, 11 on the original
    server); incomplete 1 → 0, so the guided case that now finishes fails semantically. Guidance fixes form, not
    task quality: the remaining failures are wrong answers, which is the model's job. Keep guidance on for
    serving (no cost in successes, no stuck turns); don't expect quality gains from it.
- **Browser runtime with Neuralese** (owner: "We absolutely need to implement the browser runtime").
  - The fork's server engine is now a transport-free service (`neuralese-service.{h,cpp}`, `nz_service_handle`),
    shared by the HTTP server and a WebAssembly build (`tools/neuralese/wasm/`, Emscripten 4.0.20, wasm32 for
    Safari, CPU SIMD, one thread, wasm exceptions).
  - The build is vendored at `ts-host/vendor/neuralese-wasm/` with a provenance file.
  - TS: `fetchModel` has in-process endpoints (`registerLocalEndpoint`), so the Neuralese driver, block store,
    decision readout, encode, write and digest reach the wasm service unchanged. `startBrowserNeuralese` runs it in a
    Web Worker with the model and heads mounted from Blobs (WORKERFS, no mmap); `startNodeNeuralese` runs it in
    process (NODEFS).
  - Conformance: `[final-wasm]` runs every check against the reference, 12 passed.
  - Headless Chromium (`scripts/browser-neuralese-pilot.mjs`, `test/browser-neuralese.html`): load 6 s; a plain
    turn; a forced write whose block lands in the runtime's store with its write record; and a `readout: template`
    call returning a `Neuralese<string>` (95 s on one thread: the full system prompt is prefilled on the CPU).
  - Threads (`neuralese-wasm-mt`, cross-origin isolation) and quantised Q8_0 model and heads: see the provenance file
    (8 threads: template call 10.5 s; 615-token prefill 2.2 s, 32 tokens decoded in 1.1 s).
  - WebGPU (`neuralese-wasm-gpu`, `GPU=1 tools/neuralese/wasm/build.sh`: ggml-webgpu through emdawnwebgpu with JSPI, so
    load/handle/unload return Promises and the TS service awaits them for every build). Correct end to end in
    Chromium (SwiftShader and the NVIDIA adapter). Not yet fast anywhere we can test: on the DGX, headed Chromium
    (145 and 154, under Xvfb; headless only offers SwiftShader) sees the GB10 but without `shader-f16`, which
    ggml-webgpu requires, so no GPU device registers. `chooseNeuraleseBuild` (browser export) therefore picks the
    WebGPU build only for an adapter with `shader-f16`, else threads, else one thread; hello reports the backend
    devices so a page can confirm. Next: measure on Chrome with `shader-f16` (macOS/Windows); an f32 shader path in
    ggml-webgpu would cover Linux.
