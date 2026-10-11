# Training data inventory and recovery (2026-10-11)

Audit by DGX. Machine-readable twin: `DATA_INVENTORY_20261011.json`, built by
`scripts/neuralese_data/build_data_inventory.py` from the registry, the manifests and the curated table
`DATA_INVENTORY_20261011.curated.json`. Owner rules applied:
- Maximize data. Admit unless there is a concrete defect; "pending review" is recorded as a review status.
- All licences are fine.
- Breadth comes from natlang domains.
- Training data covers whole trajectories.
- The generic system prompt is dropped after a warm-up; program-specific content always stays.
- Pop repairs the defect sets in section D.

## Reading inventory totals

Registry admission status and observed data shape are separate fields. The generated
JSON now samples declared local JSON/JSONL members and reports `data_shape` plus
`registry_counts_by_data_shape`, so native targets and conversion inputs can be
found inside administratively held snapshots. Missing remote files, compressed
members and unsupported shapes remain unclassified; a sample does not prove
correctness, split eligibility or admission. File and line totals are gross
manifest totals, including repeated snapshots and tokenization twins. They must
not be reported as unique recoverable examples or unique token counts. Registered
piece sidecars must reach the shared renderer; copying the files alone does not
make soft source references visible in rendered training inputs.

## A. Answer to the owner

**"I thought we had sizable corpora of statically generated and teacher-generated data."**

Yes. The current Neuralese direct stage trains on one set: the 4962-document union, 3258 train documents from only
**409 train trajectories**. Its supervised decision targets total **250k tokens** (Mellum).

Already on disk, natlang-native, and in the same teacher-turn format the union was built from:

| Corpus | Decisions | Target tokens (est.) | Origin |
| --- | --- | --- | --- |
| `native-sft-v13-r2` | 111,301 | ~12.5M | gpt-6-luna teacher, exact-oracle checked |
| `semantic-inline-curriculum-s72-teacher-turns` | 71,364 approved (80,056 total) | ~14M | DGX Qwen3.6 teacher; found unregistered and registered in this audit |
| `bgkit-knowledge-rationalized` | 77,767 | ~4.2M | static reference demonstrations + rationales |
| `static-lambda-rationalized-v2` | 54,753 | ~5.8M | static inline-lambda generator + rationales |
| `clef-new-rationalized-v2` | 2,619 | ~0.24M | Clef labels |
| `static-lambda-expansion` | 9,137 | ~0.65M | static generator, exact oracles |

That is about **325k decisions and ~37M target tokens, roughly 150× what the direct stage sees**. None of these
corpora shares a single source group with the union, so they are clean additions.

Beyond them:
- The admitted harness bench (3,843 SWE trajectories; 629k target tokens; already rendered for both tokenizers).
- The S1 port-record corpus (1.87M records, 45 families).
- The view/ask corpus (160k records).
- 63k more public SWE-rebench trajectories.
- Pop's free-provider and Luna campaign long tail, about 1,000 registry entries.

**"Did any available training data fall by the wayside?"**

Yes. Nothing *inside* the native lineage was lost: every gold-text document of every earlier warm-up is in the
4962 union (checked by id for 19 corpora). What fell by the wayside are whole cohorts:

1. **native-sft-v13-r2 (111k decisions).**
   - The crisp LFM student was trained on it for 1 epoch (2026-10-04), and Maple used a 200-step bootstrap.
   - It was dropped when the foundation moved to reviewed cohorts (2026-10-05), and is absent even indirectly since
     the direct restart, which starts from base models.
   - Its registry admission said "existing-native-sft-approved-recipe" but never had `training_admission`.
2. **The harness-bench cohort.**
   - Owner-admitted 2026-10-10 for both students, and used only by Pop's LFM `luna-ar-native-harness-*` runs.
   - Mellum never got it: run-v9 ran with `text_data: null`.
   - The direct restart silently dropped it from both lines.
3. **Five free-provider ordinary-text packets** (1,191 train docs, 104k target tokens), admitted 2026-10-10/11, were
   never added to any line's inputs.
4. **80k Qwen3.6 teacher turns of the s72 campaign** (`runs/semantic-teacher-20261005/turns.jsonl`) were never
   registered: only the campaign's results file was. They are now registered and admitted.
5. **The rationalized static and Clef corpora** (bgkit-knowledge, static-lambda v2, clef-new; ~135k decisions) were
   built 2026-10-06/07 and never trained. They are registered "verified replay" with no `training_admission`.
6. **S1 (1.87M records) and S2.**
   - S1 was used only in S3 pilots (~2k records per family); the S3 full run never launched.
   - Its remaining hold was licensing and provenance; licensing is void since 2026-10-10.
   - Correction (2026-10-11): the 799 JeFF split conflicts are not in S1. They are leaks of reserved test texts into the labeled-source inline curricula (D3).
   - S2 generated families never reached the Neuralese line.
7. **63k of the 67k public SWE-rebench trajectories** were downloaded as the harness bench source, but only 3,843 were
   normalized.
8. **Teacher campaign outputs** (DGX Qwen3.6 pools; `teacher-campaign-evidence-20261005`, 704k jsonl lines incl.
   repeats) were never imported beyond what fed v13.

Also dropped, on purpose (recorded, not lost):
- the Mellum self-distill / preserve stream (owner: target SFT, not self-distillation);
- `maple-broad-mixed` (owner: no broad mixing);
- the Maple renders (retired tokenizer);
- the soft-Neuralese v13 conversion (superseded format).

## B. Ranked inventory (top datasets by value for capable Neuralese natlang models)

Value means natlang-native trajectories first, verified targets, new domains and size. Size is decisions or
records; "target tok" is the supervised decision text (estimates marked ~).

| # | Dataset | Origin | Format | Size | Target tok | Admission now | Last use | Consumable by direct now |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `native-sft-v13-r2` | teacher gpt-6-luna, natlang curriculum, exact oracle | teacher turns | 111,301 (105,869 / 5,432) | ~12.5M | **admitted (this audit)** | LFM crisp student SFT 10-04 | no: render + language migration |
| 2 | harness-bench text v2 (LFM, Mellum) | public SWE-rebench → natlang pi harness | rendered text | 3,843 (2,969 / 874) | 629,543 | admitted 10-10 | Pop LFM AR runs; dropped at the direct restart | **yes** |
| 2b | `semantic-inline-curriculum-s72-teacher-turns-20261007-v1` (was unregistered) | teacher Qwen3.6 on semantic-s72 curriculum | teacher turns | 80,056 (71,364 approved) | ~14M | **registered + admitted (this audit)** | never | no: render |
| 3 | `bgkit-knowledge-rationalized-20261007-v1` | static references (bgkit/Schnitzeljagd/HotpotQA) + rationales | teacher turns | 77,767 | ~4.2M | **admitted** | never | no: render + protected check |
| 4 | `static-lambda-rationalized-20261006-v2` | static inline-lambda generator + rationales | teacher turns | 54,753 | ~5.8M | **admitted** | never | no: render |
| 5 | `s1-full-final-20261003` | bgkit/Schnitzeljagd/public → port records, 45 families | port records | 1,870,591 | n/a | **admitted** (defects in D) | S3 pilots | no: render as natlang calls |
| 6 | `nebius-swe-rebench-openhands-trajectories` | public, Qwen3-Coder-480B | OpenHands trajectories | 67,074 (32,161 resolved) | n/a | **admitted (resolved)** | 3,843 normalized | no: pi normalization |
| 7 | `view-ask-20261010-v4` | public docs/logs/html/code + computed targets | port records | 160,118 | n/a | **admitted** | never | no: render |
| 8 | `teacher-campaign-evidence-20261005` | DGX Qwen3.6 campaigns | campaign results | 703,918 lines (repeats) | n/a | **admitted (accepted, deduped)** | fed v13 | no: dedup + import |
| 9 | `s2-generated-20261004-v2` | S2 self-improvement, self-play, decision data | mixed | 211,957 lines | n/a | **admitted (accepted only)** | never | no: select + convert |
| 10 | free-provider packets v28/v29/v73/sb16/v64 | free-provider teachers | rendered text (LFM) | 1,191 | 103,720 | admitted 10-10/11 | never | LFM yes; Mellum needs twin |
| 11 | `static-lambda-expansion-20261005-v1` | static, exact oracles | native decisions | 9,137 | ~0.65M | **admitted** | never | no |
| 12 | `clef-new-rationalized-20261006-v2` | Clef labels, verified replay | teacher turns | 2,619 | ~0.24M | **admitted** | never | no |
| 13 | `semantic-inline-curriculum-s72-teacher` | Qwen3.6 teacher | results/traces | 2,184 accepted | n/a | **admitted** minus leak | never | no |
| 14 | `student-posttraining-turns-20261005` | student post-training phases | native turns | new turns only | n/a | **admitted** | student phases | no |
| 15 | Pop campaign long tail (free-provider 343, luna-semantic 145, luna-v 89, luna-choice 78, google-choice 53 entries) | Luna/Gemini/Groq/… teachers | results, reviews, held natives | ~1,000 entries, ~1.4M jsonl lines | n/a | held: Pop to sweep (E) | partly via union | no |
| 16 | 4962 union + Mellum twin | Pop authored/Luna cohorts | rendered text | 4,962 (3,258 / 1,704) | 250,020 (M) | admitted | **current direct v2** | yes |
| 17 | 4962 program-header twins (new) | derived | rendered text | 4,962 | 250,020 (M) | **admitted (this audit)** | — | yes, after the rendering mode lands |
| 18 | `translation-sources`, `source-episode-pools`, `bgkit-exports` | NLLB 241k IRs, S2 pools, bgkit stores | generation sources | — | 0 | source only | generators | feed generation (F) |

Excluded, with reasons recorded in the registry:
- `maple-sft-v13-r2`: duplicate render.
- `teacher-neuralese-v13-v8`: superseded format.
- `s3-subset`: subset of S1.
- `maple-broad-mixed`: owner rule.
- `mellum21-self-distill-v3`: owner rule.
- `protected-bgkit`: never train.

Registry overview: 1,395 corpora. The JSON's `registry_counts_by_class` sizes each class: training-admitted,
held-or-evidence, generation source, model/training state, protected evaluation. More than half of the registered
bytes are checkpoints and run state, not data.

## C. Admission decisions made (registry updated, 2026-10-11)

- `training_admission: true`, with `review_status` and `required_conversion` recorded on each:
  - `native-sft-v13-r2` (Pop-owned)
  - `bgkit-knowledge-rationalized-20261007-v1`
  - `static-lambda-rationalized-20261006-v2`
  - `clef-new-rationalized-20261006-v2`
  - `s1-full-final-20261003`
  - `view-ask-20261010-v4`
  - `nebius-swe-rebench-openhands-trajectories-20261009-v1` (resolved trajectories)
  - `teacher-campaign-evidence-20261005` (accepted, deduped)
  - `s2-generated-20261004-v2` (accepted only)
  - `static-lambda-expansion-20261005-v1` (approved 9,137)
  - `semantic-inline-curriculum-s72-teacher-20261007-v1` (accepted, minus the leak)
  - `student-posttraining-turns-20261005` (new turns only)
  - `semantic-inline-curriculum-s72-teacher-turns-20261007-v1`: newly registered; 71,364 approved rows outside the
    54 held cases
- These are recorded with `consumable_by_direct_stage: false`. Admission is not consumption: each needs the
  conversion named in the registry, which Pop owns (section D).
- `training_admission: false`, with the reason: `maple-sft-v13-r2`, `teacher-neuralese-v13-v8`,
  `maple-broad-mixed-v1-20261005`, `s3-subset-20261005`.
- **New corpora:** `neuralese-foundation-text-input-4962-program-header-{lfm25,mellum21}-20261011-v1`, admitted for
  the post-warm-up phase.
- **Splits:** held sets stay held in every decision. Held documents of the union, and the 5,432 v13 test rows,
  are never moved. Protected sets are never admitted.

## D. Defect sets for Pop to repair (exact defect, rows, suggested repair)

Pop owns these repairs (coordinator, 2026-10-11). Each is also in the JSON under `defects`.

| # | Dataset | Defect | Rows affected | Suggested repair |
| --- | --- | --- | --- | --- |
| D1 | `semantic-inline-curriculum-s72-teacher-20261007-v1` | Leak: train cases on reserved-test sources | 38 cases (dataset_workbench 30, labeled_judgments 6, cross_source_folders 2), listed in `runs/semantic-teacher-20261005/s72-holds.jsonl` | Move to test or omit; admit the other accepted rows |
| D2 | same | Train cases on hash-test sources | 16 anli_batch cases (`s72-holds.jsonl`) | Source review, then move to test or admit |
| D3 | **Corrected 2026-10-11: not S1.** Labeled-source inline-curriculum corpora | Leak: JeFF reserved test texts (ag_news, sst2, emotion, sms_spam; `training/source-split-reservations.json`) inside train rows | Exact-text audit (`JEFF_RESERVATION_OVERLAP_20261011.json`, row ids per corpus): static-lambda-rationalized-v2 lambda-v3-s61 70 texts / 132 train rows; extras-s61 2 / 2; clef-new-v2 2 / 3; semantic-s72 IR 39 / 59 (the s72 turns inherit these); native-sft-v13-r2 ≥37 / 215 (lower bound); union 0; bgkit-knowledge 0 | Move the affected programs (whole trajectories and source groups) to test, or omit them |
| D4 | `s1-full-final-20261003` | Ambiguous provenance aliases. **Overstated:** no alias list exists. The only record is a backlog sentence (`coverage_backlog`, `neuralese_dataset_coverage.json`) | unknown | DGX derives it: group S1 lineage by (store, upstream, upstream_id) and flag ids that map to more than one revision or store. Until then this is not a known defect |
| D5 | `static-lambda-expansion-20261005-v1` | Noisy dataset-label answers (label quality) | 1,088 held decisions | Relabel with constructed-world oracle or a teacher vote; else keep held |
| D6 | `static-lambda-rationalized-20261006-v2` | Source conflict (current rule) | 2 cases (lambda-v3-s61) | Source review; keep held until resolved |
| D7 | `free-provider-v73-admitted-ordinary-text-846-20261011-v1` | Failed assistant turns not masked | 82 histories, 145 spans, 25,517 tokens | **Resolved by Pop** (coord pop-22b5). Every twin (Mellum, program-header) must carry `context_only_spans` shifted to its own token coordinates |
| D8 | free-provider packets v28/v29/v73/sb16/v64 | Format: LFM tokenizer only | 1,191 docs | Mellum twin with `scripts/neuralese_data/text_tokenizer_twin.py` |
| D9 | `native-sft-v13-r2` | Format: teacher turns at runtime contract 19 (old language surface, old system prompt) | 111,301 | Render with `native_gold_packet` in both modes and both tokenizers. Migrate the language surface with `ts-host/scripts/migrate-neuralese-language.mjs` where the eval code uses removed API; rows that fail migration go to omissions |
| D10 | `bgkit-knowledge-rationalized-20261007-v1` | Possible leak into protected bgkit benchmarks (HotpotQA, SQuAD, TriviaQA, NQ questions) — unverified | 77,767 (check all) | `scripts/neuralese_data/cross_corpus.py` closure against `protected-bgkit-20261003`; hits to omissions. `*.prompts.jsonl` predate the full-context view: render from `*.turns.jsonl` |
| D11 | `s2-generated-20261004-v2` | Quarantined and failed attempts mixed with candidates | per attempt status | Keep accepted and grader-passed only; convert to teacher turns |
| D12 | `teacher-campaign-evidence-20261005` | Duplicates: alternate pools repeat the same programs, and some fed v13 | ~704k lines | Dedup by trajectory digest against v13; import accepted terminal results |
| D13 | `nebius-swe-rebench-openhands-trajectories-20261009-v1` | Format: OpenHands dialect | 63,231 not yet normalized (32,161 resolved in total) | Harness-bench normalize and replay (DGX can run this; see F) |
| D14 | `maple-broad-mixed-v1-20261005` | Missing source pins and split audit | 6,000 | Not needed (excluded by owner rule); repair only if the rule changes |
| D15 | `view-ask-20261010-v4` | Format: port records, not natlang call documents | 160,118 | Render as natlang view/ask calls (both modes, both tokenizers) |
| D16 | `s1-full-final-20261003` | Format: port records | 1,870,591 | Render as natlang call documents; start from the s3-subset slice (569k train) |

These are not defects: missing licences (owner rule); "pending review" status; the 673 union rows that cannot be
re-tokenized from text (escaped specials, handled by the token splice); superseded harness v1/v3 splits (fixed in
v4).

## E. Direct-stage training budget (bug found; run held)

**The bug.** `neuralese-direct-v2` sets steps × target_tokens × batch = 768,000 "supervised positions ≈ 2 epochs of
~384k". That is wrong in three ways:
1. A window supervises its span after the context prefix, not `target_tokens` positions. The mean is **508**
   (Mellum, target 1024) or **203** (LFM, target 256).
2. The 384k counts train+held suffix tokens (250,020 + 133,994).
3. The trainer samples one window per update uniformly with replacement over **6,801** train windows (Mellum;
   16,238 for LFM), of which only 3,258 contain a decision target. The rest are history-only windows that repeat
   earlier decisions of the same trajectory.

Measured (exact replay of `prepare_text_windows` geometry over the twins):

| | Mellum twin, target 1024 | LFM, target 256 |
| --- | --- | --- |
| Train docs / trajectories | 3,258 / 409 | 3,258 / 409 |
| Train tokens (system prompt share) | 17.6M (80%) | 17.4M (81%) |
| Train windows | 6,801 | 16,238 |
| Supervised positions per window (mean) | 508 | 203 |
| Forward tokens per window (mean) | 5,564 | 5,569 |
| Span positions per epoch (non-system) | 3.45M | 3.30M |
| Decision-target (suffix) tokens per epoch | 250,020 (77/doc) | 223,339 |
| Planned run | 750 updates | 3,000 updates |
| What it covers | **0.11 epoch**; ~27.6k suffix tokens; ~10% of train decisions seen once | **0.18 epoch** |
| Unique tokens in a prefix trie (train) | 1.95M (9× less than rendered) | — |

Cost model for Mellum with MuonSR: about **17.4 s fixed per update** (Newton-Schulz over the latents) plus about
**0.175 s per 1k forward tokens**. Fitted from 18.8 s at 8k and ~22 s at 32k; LionSR is 4.8 s + 0.175 s/1k.
Updates, not tokens, dominate. One window per update is therefore the expensive design.

**Corrected budget.** The unit is epochs over train decision targets, sampled by document. Cohorts are sampled by
document; the shared `CohortDocumentSampler` already exists in `text_supervision.py`, and `direct_sft` should use it
instead of uniform windows.
- **Phase W (warm-up, full system prompt):** 0.5 document-epoch of the union (1,629 decisions).
- **Phase P (program-header):** 2 document-epochs of the program-header twin (6,516 decision visits).
- **Then:** the new cohorts as they are rendered (harness first), each at 1–2 epochs of its targets.

Mellum cost options for W + P on the current data:

| Option | Updates | Wall at the cost model | Needs |
| --- | --- | --- | --- |
| (a) today's trainer, one window per update | 1,629 + 6,516 | ~8 h + ~32 h | nothing (too slow) |
| (b) length-bucketed padded batches up to 32k tokens per update | ~330 + ~270 | **~4 h** | right-padded batch with an attention mask (lift the "same prefix and length" restriction in `direct_sft`); latent lr retune for the ~5–24× larger batch |
| (c) whole-trajectory documents, program-header: one document per trajectory, every assistant turn supervised | 409 trajectories = 0.94M tokens per epoch → ~30 packed updates/epoch | **< 30 min for 2 epochs** | a trajectory renderer that keeps each assistant turn's reasoning. This is the owner's whole-trajectory rule; decision docs are near-prefixes of one another |

LFM (~1.3 s/update) can run option (a) with the document sampler: 6,516 updates ≈ 2.4 h.

Other efficiency notes:
- **Prefix-cache reuse of the shared system prompt** becomes moot after warm-up, since the prompt is gone. During
  warm-up, (c) or a shared-prefix pack (349 distinct system prompts over 4,962 docs) gives the same saving without
  detaching gradients.
- **Supervising the system prompt at reduced weight** is not recommended now. The owner's drop-after-warm-up
  decision distils it instead. If wanted during warm-up: each distinct prompt once per epoch (1.13M distinct train
  tokens over 234 prompts, Mellum) at weight 0.25, deduplicated, not once per decision.
- **The program-header twins cut train tokens 17.6M → 4.3M** (Mellum) and mean forward length 5,564 → 1,443. Window
  count falls from 6,801 to 5,573 at target 1024.

The queued Mellum v3 launcher stays stopped until the recipe takes (b) or (c) and the document sampler. This is a
coordinator and owner call; no run was changed here.

## F. Next data-generation steps (cheap, natlang-native, by yield)

1. **Render the admitted teacher-turn corpora** (v13, s72 turns, bgkit-knowledge, static-lambda v2, clef-new,
   static-lambda-expansion): ~325k decisions, ~37M target tokens, CPU only.
   - Use the same packet renderer, in both modes and for both tokenizers, plus the whole-trajectory render.
   - Biggest and cheapest win; Pop owns the conversion (D9–D12).
2. **Normalize 32k resolved SWE-rebench trajectories** through the harness-bench pipeline: ~8× the current harness
   cohort, no teacher cost. DGX.
3. **Static generators at scale** (zero model calls, exact oracles): inline-curriculum generator families
   (edit_stream, dataset_workbench, desks, inline_multi_capture, relational_policy), static-lambda-expansion,
   static depth2.
   - New seeds and new domains from the source pools: NLLB translation (241k IRs), the S2 episode pools
     (contractnli, scifact, CSP, timeseries, code golf, crossword, optimization) and the bgkit task stores
     (repo QA, tool digest, overview).
   - Rationales via Qwen3.6 on the DGX teacher (:8082) in the teacher window.
4. **S1 → natlang calls:** render the port records as natlang functions over their sources (45 families, 1.87M).
   This adds QA, memory, SQL, reasoning-gym, terminal and research domains as natlang programs.
5. **Teacher campaigns on the local teacher:** v13-style whole-program campaigns on the inline curriculum with the
   exact-oracle check, on DGX Qwen3.6 and Pop's free-provider lanes. Clef has ~1.22M tokens/day shared (DGX ≤800k),
   enough for about 300–500 short programs a day; spend it on labels for new source pools (Clef-flash labelling, as
   for clef-new) rather than on whole trajectories.
6. **Every new set gets three renders:** full-prompt (warm-up share), program-header, whole-trajectory.

## G. Unregistered data on disk

Swept `runs/` and `/home/werg/data` (NVMe) for JSONL files over 1 MB outside the registered paths: 49 GB in 40
directories, mostly copies or intermediates. Full list in the JSON key `unregistered`.

| Path | Size | What | Action |
| --- | --- | --- | --- |
| `runs/semantic-teacher-20261005/turns.jsonl` (+ `admitted.jsonl`) | 6.1 GB, 80,056 turns | Qwen3.6 teacher turns, s72 | **registered and admitted** (`semantic-inline-curriculum-s72-teacher-turns-20261007-v1`) |
| `runs/neuralese-integration-20261003/v13-full-candidate-v2/teacher.candidate.jsonl` (HDD) | 8.3 GB | pre-filter v13 candidate | omission recorded: superseded by the filtered `native-sft-v13-r2` |
| `/home/werg/data/harness-bench/build-*/` | 1.5 GB | harness pipeline intermediates | not additional |
| `/home/werg/data/maple-sft/maple-v13.sft.jsonl` | 3.1 GB | Maple render of v13 | duplicate (excluded) |
| `/home/werg/data/neuralese-s3-full/records/` (files beyond the include list) | 15 GB | S3 subset of S1 | subset of S1, not additional |
| HF cache: Toucan-1.5M, SWE-smith 66k, Multi-SWE-bench trajs, swe-rebench-v2 GLM5.1/Qwen, claude-code traces, kodcode, smoltalk, Ling-Coder-SFT, fineweb-edu, pg19, … | — | public raw datasets | not natlang-native; candidate sources for harness-style normalization (agent/SWE trajectories); register a snapshot when a converter consumes one |

`/mnt/external` beyond the registered S1, v13 and S2 paths was not exhaustively re-swept (slow, contended HDD).
`training/neuralese_data_ledger.json` remains the full source ledger for bgkit and Schnitzeljagd: 241 excluded,
55 migrated into S1, 22 queued (e.g. `web_sds_v3` adapter missing, `natlang:v13-corpus` migration,
`bgkit:teacher` runs, retired Python sources). Its 22 queued items are the next conversion backlog after section F.

## H. Held-data recovery on DGX (2026-10-11, owner: "held data seems like a cesspit")

Method: every file of all 1,377 registry manifests (165,526 files) compared by path and size between DGX and Pop. Sixty entries have
files that exist on DGX and not on Pop; 91.2 GB of them sit in entries without `training_admission: true`.

| Class | Size | Action |
| --- | --- | --- |
| Model and training state that Pop offloaded to DGX (`pop-inactive-model-*`, bonsai teacher, failed bootstrap, action45 exports, Maple GGUF) | 60.8 GB | Only copy. Not delete candidates; listed in a separate block for Pop to decide |
| Superseded or duplicate training-shaped data | 20.8 GB | Delete candidates in `runs/DELETE-CANDIDATES-HELD-20261011.txt`: harness records v1-v3 and Mellum text v1 (3.0), view-ask v2/v3/slice (6.5), Maple gold twins (1.0, subset of the union), v28 intermediates (0.45), luna-v17 proposals/previews (1.6), soft v13 conversion (8.1), S5 smoke replay (0.03) |
| Generation sources and indexes | 7.9 GB | Kept: view-sources v1-v3, cross-corpus index, SWE-rebench tasks |
| Live training-shaped, admitted here | 1.8 GB | below |

Recovered (all set to `training_admission: true` with `review_status`; JeFF reserved-text scan found 0 hits in each):
- `luna-v17-v9-portable-context-view-20261010-v1`: label was "root-admitted" with no flag. 3,207 train native records (~582k target+reasoning tokens, chars/4). 2,792 are decisions already in the 4962 union. **415 train decisions are not in the union text** (413 `curriculum_decision_extract_chain` from 58 trajectories whose other decisions are in the union; ~55k tokens). The 2,035 test records stay out.
- `refine-judge-exact-20261009-v1`: 1,020 train rows of exact predicate labels; 307 held-out predicate/family rows stay out.
- `refine-author-static-20261009-v1`: 28 authoring pairs (~197k tokens).
- `neuralese-operator-cases-20261009-v1`: 1,166 train spans (834 test stay out); S5 objective cases.
- `natlang-specialization-cases-declines-20261009-v1`: 7 compiled cases and 8 declines of recorded calls.
- `harness-bench-...-records-20261010-v4` (2,969 train records) and `neuralese-foundation-text-input-4962-mellum21-twin-20261011-v1`: owner-admitted or inherited, only the flag was missing.

Excluded with reasons: harness records v1 (9 records with the OpenHands timeout-notice defect; v2-v4 carry the rest); every other superseded set has no unique rows (checked by id for the Maple twins against the union; by supersession chain and closure notes for harness v1-v3, view-ask v2/v3).
Entries missing files on both machines (checkpoint state of Pop's closed lineages, e.g. `neuralese-map-v5-ar-state-evidence`, 25 files) are listed by the comparison; they are model state, not data.
