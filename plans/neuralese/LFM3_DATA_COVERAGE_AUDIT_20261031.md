# LFM3 SFT data coverage audit (2026-10-31)

Question (owner, 2026-10-31): we acquired many diverse sources, so why is the LFM3 SFT data narrow?

Answer: three causes.
1. The pipeline never delivered most of the broad sources to the run. Conversion, rendering, verification,
   registration and mixing each dropped whole sources. Usually this was silent, and in several places it was a
   concrete bug.
2. The sources that did arrive come from a handful of program templates.
3. The Neuralese share (intended, and to be kept) is about 90% one repeated program, because a single static
   port-record template feeds it.

Scope: read-only audit, plus this report. No registry or training data was changed.

The audited training source is `/home/werg/data/lfm3-sft-20261029-v6/on817200/train-sources.json`:
- sha `d2c469dd…`, 533 leaves, 1,072,735 rows;
- the restart source of the fresh LFM3 run (`LFM3_FRESH_RESTART_INTEGRATION_20261029.md`).

Counts are rows (native graphs) unless noted. "Exposure" means what reached this source list. Learner exposure maps
live in the cloud checkpoints and were not read. The run had earned only a few hundred updates when the cloud
stopped, so actual exposure is a small prefix of everything below.

Scratch evidence (scripts and outputs) is on DGX:
- `/tmp/cov-audit/`: `compose2.json` family composition, `sample2.json` decoded sample;
- `/tmp/cov-audit-A/`: public datasets;
- `/tmp/cov-audit-B/`: Neuralese diversity;
- `/tmp/cov-audit-C/`: builder replay and row ledger;
- `/tmp/cov-audit-D/`: view/ask, S1, e2e.

## 1. What was actually trained (on817200)

### By logical family

From `on817200/census.json` (own-output mass, unscaled):

| family | rows | unscaled mass share |
|---|---|---|
| native-communication | 349,693 | 64.0% |
| ordinary-source | 583,138 (incl. 255,535 `:prompt-full` variant copies) | 22.7% |
| natlang-curriculum | 138,012 | 5.4% |
| coding | 1,892 (SWE-rebench OpenHands) | 7.9% |

### Duplicate views

255,535 rows (24%) are `:prompt-full` renderings of decisions already present in their program-header form. The
rule is `/home/werg/data/lfm3-sft-20261029-v6/admit_full_variants.py`, receipt `../receipt.json`. They add prompt
robustness, not tasks.

### By underlying source

Teacher program families were mapped through the idiom score files, which hold the `program` and `curriculum_family`
of each `teacher-program:` trajectory (`/tmp/cov-audit/compose2.py`). Counting unique non-variant, non-communication
rows (468,298):

| source family | rows | share | programs | origin |
|---|---|---|---|---|
| fresh_service_recovery | 105,134 | 22.5% | 4,018 | `authored-service-recovery-20261016`: 8 services x ~16 worlds (128 worlds) |
| workbench_tickets / applicants / reviews | 72,109 | 15.4% | ~2,900 | authored workbench worlds |
| teacher_decision_labels clinc_intent_choice | 31,317 | 6.7% | 1,865 | CLINC150 labels |
| people_chain + people_lookup | ~25k | 5.4% | ~1,000 | schnitzeljagd synthetic people |
| decision_extract_chain | 15,907 | 3.4% | 556 | authored-bounded-decisions-v1 |
| workflow_customer-service / invoice | 23,014 | 4.9% | ~7,900 | workflow directories |
| dataset_workbench, knowledge_evidence | ~28k | 6% | ~1,800 | Luna workbench / evidence |
| public-dataset curricula | ~45k | <10% | | see below |
| long tail | | | | ~140 other families |

The public-dataset curricula break down as:
- HotpotQA: 20,981 + 2,038;
- bgkit (repo/web): 9,425;
- CommaQA: 4,512;
- QuALITY: 3,534;
- ProntoQA: 2,853;
- TextWorld: 2,063;
- ProofWriter: 1,127;
- KQA Pro: small.

Concentration: in `joint-sft-v3` the top 5 families hold 56% of rows and the top 10 hold 64%. `fresh_service_recovery`
alone is 35% there, about 563 rows per world.

### By task shape

Decoded sample: 3,969 rows, uniform over all leaves, `/tmp/cov-audit/sample2.json`.

| row type | sample rows | distinct call signatures | dominant signature |
|---|---|---|---|
| non-communication | 2,746 | 640 | `select_record(record: Row, policy: Policy): boolean` (696, 25%; the service-recovery/workbench template) |
| communication | 1,080 consumer rows | 1 | `nl@eval:N(item: Item): string` (all of them) |

The rest of the non-communication rows are `summarize(policy)`, `reduce_decisions()`, `category_total()`, ticket
classifiers and small `nl@eval` predicates. The communication rows are exactly the narrowness the owner noticed.

### Communication graphs (owner addition)

Source: helper B, a full scan of all 533 leaves.

**Rows by source:**
- 338,474 rows: big2-stream tranches 000-002;
- 10,245 rows: big2-offload-1000;
- 119 rows: genuinely varied (teacher-decision-label worlds, 135 rest27-reviewed minus withdrawals).

**Shape:**
- Every big2 producer is `answer_collection(): Record<string,string>` and every consumer is
  `nl@eval:8(item: Item): string`.
- Opening and instructions are identical; only item data differs.
- About 10% of rows reverse the roles.

**Constant payload:**
- 89% of big2 edges carry the identical inline-site payload "Do what item.request asks, using only
  item.evidence. Follow item.answer_format exactly and return only the answer."
- That is 302,168 of 338,474 stream rows, plus 7,716 of 10,245 in offload-1000.

**Targets:**
- 89.9% of producer targets are one byte-identical 327-token `iterateOn(...)` eval program. That is about 33M of
  about 42M target tokens per tranche.
- Consumer targets are mostly paging: `read_page` 53.5%, `item.evidence` 17.4%, final answer 29%.
- Only 2.6% of the 1.6B processed tokens per tranche are targets (7k-token producer contexts).

**Diversity that exists:** it is only in the item contents, which come from 14 port-record families:
- code_python 93k;
- ALFWorld 43k;
- agent_continuation 33k;
- file_overview 33k;
- HotpotQA 29k;
- function_call/xLAM 26k;
- tool_policy 24k;
- science 36k;
- qa_mcq 10k;
- WebShop 4.6k;
- ChatQA2 4.2k;
- memory_qa 2.4k.

**Derivation:**

| step | what happens | evidence |
|---|---|---|
| 1 | "Big2" is cohort `neuralese-e2e-cloud-big2-protected-20261015-v1`: one family `curriculum_port_record_calls`, 70,656 chains (49,156 `port-record-s3-v1`, 21,500 `port-record-view-ask-v3`) | `/mnt/external/natlang-lfm3-data-20261028/big2-stream/meta/receipt.json` |
| 2 | Chains are scripted replays, not teacher runs: one root directory reducer applying one inline `nl` template to every record, child = reference replay of the gold target. The `teacher-program:` ids do not mean teacher behaviour. | `build-port-record-calls.mjs:7-17,142` |
| 3 | `e2e_mine.py` mined the edges | |
| 4 | `build_native_semantic_pairs_e2e.py` emits one 2-node graph per edge, so each chain's producer turn is repeated about 23 times | |
| 5 | The "cap readers per distinct body at 64" rule was not applied | `E2E_DATA.md` |

## 2. Inventory and pipeline trace by source family

### Pipeline stages

| stage | meaning |
|---|---|
| reg | registered in `training/neuralese_corpora.json` with `training_admission` |
| conv | converted to natlang tasks or native turns |
| LFM3 | rendered with the LFM3 tokenizer (`runs/lfm3-renders`, `runs/lfm3-teacher-graphs`, `/mnt/external/natlang-lfm3-data-20261028`) |
| mix | in a mixture or append that reached on817200 |

### Size reference: the LFM2.5 line

The registry entry `natlang-target-union-r11-program-header-ready-20261015-v1` trained the LFM2.5 line on:
- 899,851 train program-header documents;
- 362.7M supervised tokens;
- 124 member sources.

The LFM3 `joint-sft-v3` mixture had 205,948 rows and 32.6M supervised tokens.

### Trace

| source family (kind; license per registry) | available | reg | conv | LFM3 | mix | drop point and reason |
|---|---|---|---|---|---|---|
| Qwen/Qwen3.6/Flint/Luna verified teacher trajectories (teacher; self-hosted) | ~1,070 registry packages | admitted | yes | 900 program-header twins, all verified (`runs/lfm3-teacher-graphs`) | yes: v3 used 865, additional-v3 35, plus 255k full variants | none. This is most of the ordinary data. Its narrowness is the curricula the teacher was given (section 1). |
| Luna held-shapes/repair, folder-criteria, greenhouse, workbench-salvaged, qad, mellum-proxy, sol-repair, clinc v5/6/10/19/21/24, refine, pop-v43, folder-edit, clef-new (teacher, Luna) | 300-800 docs each | admitted | yes | registered | yes (v3 + additional-v3) | refine (1,048) contributes 0: judge rows fail "root outcome not verified" before the choice-call exemption (`idiom_scores.py:155-171`; G3 effectively open) |
| s72 semantic curriculum (Qwen3.6 teacher) | 68,318 PH docs | admitted | yes | verify-failed, then repaired subset | yes (via `repaired-fourteen-mixture-v1`) | whole-render verification fail (D2), recovered by the exact-subset repair |
| bgkit-knowledge-rationalized (static + rationales) | 69,760 | admitted | yes | same | yes (repaired) | same as s72 |
| static-lambda v2 + expansion | 52,394 + 8,825 | admitted | yes | same | yes (repaired) | same as s72 |
| campaign-uncovered, closed-backlog-v5, clef-pools x5, misc-admitted, s62000 | 28k, 23.6k, 14k, 2k, 0.8k | admitted | yes | same | yes (repaired) | same as s72 |
| **native-sft-v13-r2** (Luna teacher, exact oracle) | **111k decisions, 102k PH docs** | admitted | yes | **render failed** (`done marker does not match current render job policy: 0000/lfm3`, `render_native_turn_corpus.py:1510`) | no | stale done-marker from an older render policy; never re-run |
| **static-s7k-curated** (static) | **65k PH docs / 87k decisions** | admitted | yes | **render failed** (part 2 failed 5x, `render_parts.py:322`) | no | never re-run |
| **external-expansion first-wave D01/D08** (static typed handoffs; acquired) | **44,219 PH docs** | admitted | yes | rendered, verify-failed (whole-trajectory: 34 docs changed split/group) | no | D2 + D3 |
| **external-expansion agentic-coding D40** | **160,403 PH docs** | admitted | yes | verify-failed (3,383 WT docs) | no | D2 + D3; coding-heavy, belongs in the coding share |
| **external-expansion rightfit-long, gatc, deepdive-long** | 33,045 / 12,125 / 7,374 | admitted | yes | verify-failed | no | D2 + D3 |
| **external-expansion codeagent D05** | **149,840 PH docs** (LFM2.5 line: 236M supervised tokens) | admitted | yes | verified + graphed 10-30 05:18 | no | D1. Policy v3 even says codeagent and first-wave "go through the normal tiers" (`lfm3-joint-sft-mixture-policy-v3.json` `coding_mix.reason`). |
| external-expansion rightfit, deepdive, deepdive-ref-qa, agentsynth, deepnlp | 17,330 / 4,826 / 8,024 / 2,186 / 30 | admitted | yes | verified + graphed 10-29/30 | no | D1 |
| external-expansion evidence-terminal D10/D29/D33 | 339,288 PH docs (4.5B reference tokens) | admitted | yes | rendered, merged 2026-10-31 00:52 (still in the queue) | no | in progress |
| **Search/OWL D02/D21** (external native teacher turns; MIT/Apache) | **40,818 cases / 658k decisions; 251k PH + 82k WT graphs** | admitted (49-52 entries) | yes | **no**: views lfm25/mellum21 only | no | D5 |
| **acquired agent sessions D12-D32, acquired-agent-full D17 / native-recovered** (third-party agent SFT) | **~300k+ graphs, ~13B lfm25 tokens; D17 16k graphs** | admitted (71/72) | yes | **no** | no; used only as protection (2,090 rows withdrawn) | D5 |
| **gemini-clinc-static** (free-provider Gemini, CLINC CC-BY-3.0) | **20,140 + 7,132 + 640 decisions** (LFM2.5 line: 19,981 docs) | admitted (24 entries) | yes | rendered, **0 train rows** | no | D4 |
| teacher-campaign-evidence-20261005 | 27,667 docs (LFM2.5 line) | admitted | yes | only inside campaign-uncovered (repaired) | partly | |
| SWE-rebench OpenHands bulk (coding) | 31,526 trajectories, 24,406 eligible, 1,354 repos | admitted | yes | sample of 1,892 rendered | 1,892 | `sample_coding_mix.py --per-repo 2` cap; LFM2.5 line trained ~190k swe-rebench-bulk docs |
| harness-bench-v4 wholetraj | 489 | admitted | yes | registered | no | D1 |
| neuralese-e2e-cohort v2-v4 (communication; LFM2.5 era) | 95k-106k edges | admitted | records | via big2 (port records) and rest27 (the rest) | partly | below |
| **rest27** (mined e2e, 101 source families) | **39,134 LFM3 train pairs, 16,585 distinct payloads** | admitted | yes | pairs built, **no verification, no graphs** | only the 135-row reviewed subset | D6 |
| **big2 tranches 003-008** (port-record communication) | **~677k rows, ~251M target tokens** | admitted | yes | verified LFM3 graphs | no | D6. More of the single shape: fix D7 first. |
| big2 tranche 009 + 010-012, offload-5000 (incl. view-ask v3) | 8.7k + 149.7k + 50.6k pairs | admitted | yes | 009 OOM-killed 2026-10-30 09:05 (`lfm3-data-c-big2-eight-20261029`), 010-012 staged only, offload-5000 rendered without verification | no | D6 |
| S1 port records (50 public datasets, 1.87M records, 45 families) | 1.36M train records | admitted (`s1-full-final-20261003`) | S3 subset only: 25k/family cap, 569k train rows | via big2 only | big2 000-002 | D8 |
| port-record-calls S3 v2 (HotpotQA, NarrativeQA, TriviaQA, needles, people) | 10.3 GB manifest | admitted | yes | no | no | data missing: `data/neuralese/corpora/port-record-calls-s3-20261011-v2` points to a missing `/home/werg/data/natlang-corpora/...` |
| view/ask v4 (CNN/DM, CodeSearchNet, WTQ, xLAM...; 172,657 records) | 160k-record view stage | v4 admitted | to port-record-view-ask v3 | as big2 chains | 10,245 pairs + 855 whole-chain rows (offload-1000 only) | the projected `view(x): Neuralese` / `ask` design was retired with the projected trainer (c1d3b5edc; `SHARED_STATE_MIGRATION_20261015.md`), so its view-operator gate can never pass. Static replay keeps only `answer_collection` + inline child. 50,693 reconstruct records were dropped ("identity as a call"). |
| MuSiQue hop chains | 67k train records / ~17k questions | held | | | no | family hold `musique_hops` (`training/source-reviews/family-holds.json`): the hop adapter selects support by answer mention, without checking the relation |
| free-provider Luna/Google pools, choice-confidence, annotation folders (~350 entries) | | `training_admission: false` | | | no | held pending review; outside this pipeline |
| held/dev packets, protected bgkit benchmarks (SQuAD/bAbI/needles) | | protected | | | no | correct |

Teacher curricula over public datasets: ProntoQA, KQA Pro, CommaQA, ProofWriter, EntailmentBank and TextWorld have
96-127 programs each. ScienceWorld/ALFWorld tasks, FOLIO and ANLI have 0 rows in v3.

### Row ledger, v3 to on817200

Every step was id-diffed (helper C). No duplicate ids at any step.

| step | rows | change |
|---|---|---|
| v1 + v2 + v3 | 206,674 | |
| v3-split-repair | | -109 held-connected |
| coding | | +1,170 +1 |
| mixed-held-correction | | -4,682 |
| big2-offload + deferred coding | | +10,245 +722 |
| chains | | +127 +727 |
| additional-render-and-twin-mixture-v3 | | +82,416 |
| repaired-fourteen withdrawal | | -2,545 |
| clef-exact | | +2,248 |
| repaired-fourteen source successor | | +183,759 |
| acquired-protection withdrawal | | -2,090 |
| big2 tranche 000 | | +112,455 |
| small additions | | +62 |
| big2 001-002 | | +226,019 |
| full variants | 1,072,735 | +255,535 |

The repaired-fourteen rows reach on817200 partly under `acquired-protection-withdrawal-v1/cohorts` names. Only
3,639 of its 185,308 rows are absent.

## 3. Defects that caused systematic exclusion

Ranked by rows lost.

| # | defect | evidence | rows lost | fix |
|---|---|---|---|---|
| D1 | No path from a newly graphed render to the live recipe. The mixture is a one-off `build_lfm3_joint_sft.py build`, and every later source was a hand-built append. | Since 2026-10-28 16:31 `student_rerender.py register` (line 647) crashes on every queue-loop pass (105 tracebacks in `.../lfm3data/logs/queue-loop.log`): `swe-rebench-coding-mix-render-20261028-v1` is marked `graphed` by `runs/lfm3-renders/coding-mix.sh` without a `student-verification.json`. All later graphed renders stay unregistered. | ~182k PH docs (codeagent 149,840, rightfit 17,330, deepdive-ref-qa 8,024, deepdive 4,826, agentsynth 2,186, harness 489) | fix the queue-loop crash; make "graphed and verified" automatically produce an append candidate |
| D2 | Student-rerender verification is all-or-nothing against the LFM2.5 reference. One document whose split/source groups changed (usually because protection widened since the LFM2.5 render, i.e. the LFM3 render is more conservative) fails the whole render. | `student_rerender.py:524,529,565` | 21 renders, ~580k PH docs blocked; 14 later recovered (183,759 rows) by a one-off exact-subset repair | accept a render minus the changed ids (the repair's logic) as the normal path |
| D3 | The exact-subset repair skips parts-kind renders. | `continue-exact-render-subsets-v1.py:5` requires a dict `target_parse`; parts renders store the string "not checked (parts render ...)" | first-wave 44k, agentic-coding 160k, rightfit-long 33k, gatc 12k, deepdive-long 7k (whole-trajectory failures only; full/program-header views identical to reference) | include parts renders |
| D4 | Registry admission not applied at render time. | gemini-clinc LFM3 renders ran without `--registry-admits-reason`; `render_native_turn_corpus.py:785-797` dropped every row as `not_approved: scripted source reference requires independent semantic and training review` (20,140/20,140 in `selection-omissions.jsonl`; receipt `registry_admits_reason: None`). The LFM2.5 render applied it. | ~27.9k decisions | re-render with the registry reason; take the flag from the registry automatically |
| D5 | LFM3 rerender queue built only from registered text renders. Graph-direct corpora (no text render) were never queued. No LFM3 plan mentions them. | | Search/OWL ~333k graphs; acquired sessions ~300k graphs | graph-direct LFM3 conversion, capacity qualification for long sessions |
| D6 | Prepared communication builds abandoned: verification/graph step never run, or a build killed and not restarted. | big2-stream tranche 009 OOM-killed 2026-10-30 09:05; 010-012 staged only; offload-5000 never verified; rest27 39k pairs never verified | rest27 39k diverse; big2 003-012 ~835k single-shape | |
| D7 | Big2 converter design multiplies one template. | one 2-node graph per edge repeats the producer 23x; no 64-reader cap per payload body (`E2E_DATA.md`); consumer cut to one paging decision | ~90% of communication mass is one program | one graph per chain; cap readers per payload; keep the consumer's answer decision |
| D8 | Port-record conversion covers 18 of 45 S3 families at a 25k cap. | 22 families never run through `build-port-record-calls.mjs`: BIRD/Spider text_to_sql, SynLogic, reasoning-gym, Knights, repo_*, tool_digest, memory_recall, recall_*, table_qa, trajectory_continuation_{swe,terminal,search,research} | ~1.28M records above the cap | |
| D9 | Held carve overflow (v3 code `d36139f02`): every component touching a carved group was taken, the first 1,500 rows held, and the rest dropped as "carved held group beyond the held row cap (not trained)". Fixed in `90d702bb7` for new builds, never re-appended. | 0 of 8,648 present in on817200 | 8,648 (folder_index/triage/mixed 5.2k, clinc labels 1.7k, labeled_judgments 478) | re-admit after a current protection-union check |
| D10 | Held carve selects whole folder families (current code). `carved()` marks a component if any of its groups hashes under 1.5%. Folder/reducer rows carry ~52 groups, so rows with >50 groups are 100% carved; in v3 folder_mixed lost 100% and folder_index/triage 54-67%. | `build_idiom_mixture.py:251-257,320` | thousands of folder rows per build | one draw per component; cap a family's held share |
| D11 | False "un-awaited nl" exclusions. A named template `const f = nl<T>\`...\`; await f(...)` is counted as un-awaited, so whole chains are excluded. ~87% false in samples (folder-criteria 122/142, closed-backlog 814/935). | `idiom_scores.py:92-96,176-177` | ~6.8k of 7,852 decisions; hits `named_versus_inline` and per-item idioms specifically | count a bound-then-awaited template as awaited |
| D12 | Oracle regrade hold with no regrade. `regrade_passed: []`. | policy v3; `build_idiom_mixture.py:182-214` | 6,713 rows (contract_diagnosis, edit_stream) | run the regrade |
| D13 | Answer-only drop rule removes verified typed text answers. | `idiom_scores.py:235-259` | 277 (folder_find, chat_reply) | count a verified invocation's typed final text as a natlang action |
| D14 | No caps after policy v2, and equal mass per cohort file. | `fresh_service_recovery` 35% of v3; t2-scope 42 of 88 files ≈ 47.7% of loss. The later 4-family CE does not rebalance within `ordinary-source`. v1/v2 had the opposite failure: caps cut 2,138 → 338 and 3,049 → 388. | concentration, not lost rows | bounded mean-one down-weight per (template, world) instead of drops or caps |
| D15 | Coding mix capped at 2 trajectories per repository. | `sample_coding_mix.py --per-repo 2` | 1,892 of 24,406 eligible trajectories (the LFM2.5 line trained ~190k swe-rebench docs) | |

**Not material:** "decision already taken by another sample" mostly removes program-header copies of decisions that
a whole-trajectory chain already covers, plus duplicate teacher package ids (v19 mellum-prefix copies).

**Correct exclusions, kept:** held-connected withdrawals (9,426 rows). The builder only checked direct overlap, so
they were caught late; the closure fix is in `ee821d48f`.

**Repeat history:** `DATA_INVENTORY_20261011.md` section A found the same pattern on 2026-10-11. native-sft-v13-r2,
s72, bgkit, static-lambda and harness-bench "fell by the wayside" in the LFM2.5 line. In LFM3 v13-r2 and harness are
lost again, and s72/bgkit/static-lambda only came back through the repair.

## 4. Biggest losses: usable but absent

Ranked by diversity value times size. Sizes are rows or documents of the stated view.

| # | source | size | what it takes |
|---|---|---|---|
| 1 | **external-expansion codeagent + rightfit + deepdive(+ref-qa) + agentsynth** (graphed LFM3) | ~182k PH docs | Fix D1 (queue-loop crash), score, mixture append through `diloco_resident publish-source-update`. Codeagent is a natlang port of agentic coding/research (port_record_calls directory reducers, not plain SWE); weight it per policy. CPU hours. |
| 2 | **Search/OWL D02/D21 + acquired agent sessions D12-D32/D17** | ~333k + ~300k graphs, ~13B lfm25 tokens | Graph-direct LFM3 re-encode (D5); capacity qualification for long sessions; protection-union check. Largest broad, real-agent source. CPU days. |
| 3 | **external-expansion first-wave, agentic-coding, rightfit-long, gatc, deepdive-long** (rendered LFM3) | ~257k PH docs | Exact-subset repair for parts renders (D3), drop the ~5k changed ids, append. CPU hours. Agentic-coding counts toward the coding share. |
| 4 | **native-sft-v13-r2** | 102k PH docs / 111k decisions | Re-run the render with a fresh output dir (stale done marker), verify, append. |
| 5 | **static-s7k-curated** | 65k PH docs / 87k decisions | Re-run the failed part 2, verify, append. |
| 6 | **Neuralese over broad sources** (owner addition; keep the communication share, broaden it) | | see list below |
| 7 | **evidence-terminal** | 339k PH docs (rendered 2026-10-31) | Verify (D2 applies), register, append. |
| 8 | **gemini-clinc static** | ~27.9k decisions | Re-render with the registry reason (D4). |
| 9 | **SWE-rebench coding** beyond the per-repo-2 sample | ~22.5k trajectories | Larger sample within the ~10-15% coding share. |
| 10 | **D9 carve-overflow rows; D11 false un-awaited exclusions; D12 regrade** | 8.6k + ~6.8k + 6.7k | Re-append after protection check; fix `idiom_scores.py` and rescore; run the oracle regrade. |
| 11 | **Unconverted S3 families** (BIRD, Spider, SynLogic, reasoning-gym, table_qa, tool_digest, memory, trajectory continuations) and the missing S3 v2 data (TriviaQA, NarrativeQA) | ~570k train records | Run `build-port-record-calls.mjs` on these families with multiple program shapes (not one `answer_collection` template); recover S3 v2 from Pop/backup or rebuild. |

Neuralese over broad sources (item 6), in order of readiness:
- **(a) rest27:** 39,134 LFM3 train pairs, 101 families, 16,585 distinct payloads (s61 edit_stream, bgkit, HotpotQA,
  s72, translation, banking77 folders, CommaQA, CUAD, TextWorld, workflow). Run `runs/lfm3-communication/verify.py`,
  emit graphs, regrade or exclude `edit_stream` (old oracle), check protection, append. CPU hours. Then review its
  67,364 `source_quality_quarantine_chain` omissions, up to 5x more.
- **(b) Teacher trajectories:** run `e2e_mine` over the verified Qwen3.6/Qwen/Flint teacher trajectories. Real
  model-written child results read by later calls; ~5-20k edges from existing batches; whole-graph chains, not 2-node
  splits.
- **(c) HotpotQA hop chains:** re-mine all 17k (rest27 holds 9.3k): ~+60k result edges. MuSiQue needs per-hop
  relation checks before admission.
- **(d) Port-record builders with intermediate values:** new builders whose children return intermediate gold values
  that later calls read. Examples: per-hop sub-answers, per-document summaries to a combiner, extract-then-answer,
  lead/sub-agent splits over agentic sessions. Also the real view/ask shape: a `view(x): Neuralese<string>` child
  plus an `ask` child that sees only the view, ~120k consumer records. Yield is hundreds of thousands of edges with
  distinct payloads. Needs DECOMPOSITION and live prompt measurement.
- **(e) Before admitting more big2:** apply D7 (one graph per chain, 64-reader cap, answer decision kept). Tranches
  003-012 add ~835k rows of the same single shape. As is, they would make the communication share even narrower.

## 5. Recommendations for the next mixture

- **Communication share stays at ~40%, filled differently:** rest27 and teacher-mined chains first, then
  D7-corrected big2. Not 003-008 as built.
- **One automated path, `rendered → verified-minus-changed-ids → scored → appended`,** owned by the queue loop.
  Fixes D1-D3. Also a coverage check that reconciles every `training_admission: true` registry entry against the
  current train source and fails loudly on unexplained absence.
- **Diversity weighting:** bounded down-weight per (template, world) and per program shape (D14), measured by
  distinct call signatures and payloads per packed group, not row counts.
- **Variant copies:** keep the 255k `:prompt-full` copies as prompt robustness, but count them as duplicates in
  diversity and family shares.

## 6. Recovery progress (2026-10-31, supervisor order: fix delivery and recover the losses)

Custody and held checks are unchanged throughout. Every recovered row still passes render verification, the
current protection union and held closure before it is sealed.

| defect | state | commit / unit |
|---|---|---|
| D1 register crash | fixed: graphed items with no or failed verification are skipped. The queue loop registers every 30 min; 92 renders are registered, including codeagent, rightfit, deepdive (+ref-qa), agentsynth and harness | a08e9eae6, 931242166 |
| D2/D3 all-or-nothing verification, parts renders | per-document exact subset is the normal path for direct and parts renders. It reproduces the one-off repair's counts exactly. The queue launches `subset` for every eligible verify-failed item: 17 queued, post-step slots raised 3 → 6 | 892fb2986 |
| reference-free renders (coding mix) | verify with `path: None` (decode, call markup, train/test overlap, target parse) | 3e4263c2d |
| D4 gemini registry reason | `--registry-admits-reason` from the receipt or row hold reason; all 10 gemini renders re-queued as `-r2` | dd0eac29f |
| v13-r2, static-s7k | stale state moved to `/mnt/external/natlang-lfm3-data-20261028/stale-render-state-20261031/`; re-rendering with `--serving-entry mix` | queue |
| D5 Search/OWL | LFM3 parts render running (164 turn files, protected bgkit index, serving mix) | queue `external-expansion-search-owl-native-20261015-v1-render` |
| D5 acquired sessions | `convert_acquired_sessions_student.py`: the same converter, protection v3 manifest and decision holds over all 14,767 admitted exports (D13 12,223, D14 2,493, D32 49; 15 D24 `/work` exports are missing). Two shards, output `/mnt/external/natlang-lfm3-data-20261028/acquired-sessions-lfm3-20261031/` | d66f2ce5a; `lfm3-cov-acq-{0,1}` |
| D8 port records | S3 re-extract at 100k per family (superset of the 25k selection) is waiting for ledger memory. Conversion of all 45 families follows | `lfm3-cov-s4-subset` |
| D9/D10 carve | component carve (one draw per component, max 3,000 held rows); overflow rows go back to train | policy v4 |
| D11 un-awaited | named and bound-then-awaited templates are handled: folder-criteria 142 → 0 flagged | 78e232aed |
| refine-judge (1,048 rows scored 0) | crisp-checker-label rows count as verified | a1dc24a4b |
| D14 caps / mass | `balance()`: 24 rows per near-duplicate cluster, family share 4% with a 3,000-row floor, continuation families exempt. Logical-family declared/1 objective from an exact own-output census (communication 40 / natlang 40 / coding 12 / external-agent 8), with max_scale 8 as a true cap. This replaces source-cohort | d66f2ce5a, a1dc24a4b |
| 255k `:prompt-full` copies | each decision appears once, in either its program-header or its full view (20% header share, policy `prompt_views`) | d66f2ce5a |
| D15 coding | v2 sample at per-repo 12 (v1 had 2: 1,892 of 24,406 trajectories). v2 is a superset of v1, so v1 is retired from the queue when v2 graphs | `lfm3-cov-coding-v2` |
| rescoring | scores now live in `runs/lfm3-mixture/scores/idiom-<sha12 of idiom_scores.py>`, so every scorer fix rescores. `score-renders` and `score-teachers` are running at low priority | `lfm3-score-*`, `lfm3-cov-score-teachers` |
| D12 regrade, D13 answer-only | open (D12 is a hold; it is not relaxed) | |

Next: `build_lfm3_joint_sft.py build --policy v4` once scoring completes. Held packets are dev v1/v2 and held v3/v4.
The build then runs the protection-union check, CE, logical-family objective, geometry, learner inputs and
registration, followed by a composition report by source and task shape. Target: ≥ ~900k docs and ~360M supervised
tokens.
