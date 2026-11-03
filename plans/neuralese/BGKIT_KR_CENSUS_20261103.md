# bgkit knowledge-reasoning stores: census and RL reader supply (2026-11-03)

Owner question: does the bgkit knowledge-reasoning (KR) data contribute to training? Answer, with the evidence below:
a little to the corrected SFT mixture (about 4% of its rows, all with zero earned exposure so far), not at all to the RL
pool, heavily to the compaction-operator cases, and not at all through the registered `bgkit-knowledge` render's large views.
Five stores that had exact or checkable answers and no use now feed RL reader families (section 3).

## 1. Method and caveats

- **SFT mix.** Every row of the corrected diverse mixture (`ops/corrected-mix-diverse-1103/train-sources.json`, 117 source
  files) was streamed and its `source_groups` read: 100,840 of the 100,996 listed rows (the other 156 sit in one small
  JSON-described acquired delta that carries no bgkit groups). A row counts as bgkit-derived when a group namespace is
  `bgkit-*` or `memory-episode`, or its id is `chain:bgkit-knowledge:*`; KR reader namespaces (`knowledge`, `hotpotqa`,
  `hotpotqa-q`, `quality`, `narrativeqa-doc`, `chatqa2-doc`) are counted separately. Identification is by group namespace and
  id, never by id alone. Earned exposure is not determinable: this mixture has not trained (the full67 parent is a zero-SGD
  source adoption), so every row has exposure 0 and a scheduled quota of 1 to 7 visits.
- **RL pool** (`selection-pool-20261103T171122Z-authoring-round2`, 11,275 rows): 0 rows mention bgkit; its reader families
  (HotpotQA, SQuAD, 2Wiki, MuSiQue, TriviaQA, QuALITY, ...) come from the upstream datasets directly.
- **Operator cases**: row counts of `/mnt/external/natlang-compaction-operators-20261103/cases/*/*.port-records.jsonl`.

## 2. Census

SFT mixture counts are rows of the corrected 100,840-row scan (families: rest27 = `native-communication-rest27`).

| Store / corpus | Size | SFT mix (rows, 0 exposure) | RL pool | Operator cases | Why unused / status |
|---|---|---|---|---|---|
| bgkit-knowledge-rationalized (+ lfm3 render, registered) | 77,767 source rows, 70,040 selected, views 67,325 program-header / 36,913 whole-trajectory | 2,615 rest27 `chain:bgkit-knowledge:*` graphs; 4,181 rows with chain id or `knowledge:` group (2,729 rest27, 1,363 natlang, 49 chains, 40 comm) | 0 | - | The registered renders' large graph views are not in the mix: only the rest27 chain graphs were admitted. Quality correction withdrew 488 of 1,940 predecessor `knowledge:` rows. |
| tool digests (`tool_digest_v2`, `_ext_v3`) | 7,429 + 14,263 | 904 (rest27) `bgkit-digest` | 0 | compact-value tool-digest 11,160 | used |
| repo stores (`repo_qa_file_v3`, `repo_localisation_v3`, overviews) | 15,652 + 9,895 + 150,000 + 8,106 | 1,856 `bgkit-repo` (1,314 rest27, 506 natlang, 17 comm, 19 chains) | 0 | repo-qa 6,498 + agent-tool-slots-repo-qa 4,800 | used; predecessor withdrew 96 of 638 |
| web stores (search_r1, sds, openresearcher, openseeker, miroverse, webshaper) | tens of thousands | 352 `bgkit-web` | 0 | compact-value web 24.6k (search-r1 15,509) | used; no oracle outcome for RL (needs a search service) |
| `qa_narrativeqa` (+ provenance `narrativeqa`) | 14,604 rows / 1,217 stories; provenance 32,367 Q | 97 (`narrativeqa-doc`: 72 natlang, 15 rest27, 7 chains, 3 comm) | 0 | s1-qa-narrative 4,105 | **now RL** (`bgkit_narrative`) |
| `memory_qa_v2_pairs` (+ provenance `memory`, `memory_recall_v2`) | 6,657 (provenance 4,000, recall 41,811) | 109 `bgkit-memory` + 172 `memory-episode` | 0 | none | predecessor withdrew 238 of 261 `memory-episode` rows; **now RL** for the verbatim-answer subset (`bgkit_memory`) |
| `babi_filler_v3` | 46,565 | 0 (no `babi-story:` group in predecessor or corrected) | 0 | 0 | **now RL** (`bgkit_haystack`) |
| provenance `searchqa` | 117,384 Q (key, clue, answer) | 0 | 0 | 0 | question-only store; snippets are upstream `lucadiliello/searchqa` (same 117,384 keys, local HF cache). **now RL** (`bgkit_searchqa`) |
| provenance `pubmedqa` | 211,088 artificial yes/no + PMID | 0 bgkit rows (PubMedQA train items are teacher decision cases, a different path) | 0 | 0 | abstracts absent locally for the 211k; the 1,000 expert-labelled items (`decision-sources/qiaojin__PubMedQA`) are **now RL** (`bgkit_pubmed`) |
| provenance `newsqa` | 37,136 | 0 | 0 | 0 | no article text locally; the "question" is the highlight text and equals the gold. Unusable. |
| provenance `msmarco_passage` | 502,939 | 0 | 0 | 0 | gold answers are free-form human text and passages are absent (browse tree carries ids only). No exact oracle. |
| provenance `kilt_wikipedia` | 184,094 | 0 | 0 | 0 | prompts are "Tell me about <id>"; gold is the page intro. Not a QA task. |
| provenance `git_history` | 3,599 Q over 1,576 commits | 0 | 0 | 0 | LLM-written free-form answers; checkable only by a git service over `bgkit-data/repos` (next action, section 5). |
| `qa_pairs` / `qa_pairs_filtered` | 114,598 / 86,113 QA over 1,961 repos | 0 | 0 | 0 | registered "queued, bgkit.repo_metadata" but that converter was never written. Answers are small-model free-form text that is often wrong (a spot check: a `.gitignore` question answered with `NSApplication` delegate methods), so there is no exact oracle. |
| `qa_squad`, `qa_squad_article_v2`, `qa_hotpotqa`, `qa_hotpotqa_distractor`, `qa_triviaqa` | 98,169 / 8,480 / 97,852 / 47,405 / 58,489 | 0 direct (`hotpotqa` groups 449 + `hotpotqa-q` 557 come from other sources) | upstream families | compact-params multihop 3,300, trivia 3,300 | benchmark questions are in the protected bgkit index (HotpotQA/SQuAD/TriviaQA/NQ); RL already reads the upstream datasets |
| `qa_quality_mcq` | 3,810 | `quality` 457 (other sources) | upstream QuALITY | s1-qa-mcq 811 | covered |
| `qa_multineedle`, `_long_fam` | 74,240 / 74,220 | 0 | 0 | 0 | SQuAD train passages in filler; same passages as the upstream SQuAD family, whose titles are protected. Not added. |
| `chatqa2_long_sft` / `chatqa2_narrativeqa` | 30,000 / 14,760 | 2 (`chatqa2-doc`; predecessor withdrew 192 of 194) | 0 | s1-qa-long-context 4,300 | mixed upstream, unverified licence terms and protected-benchmark overlap unchecked; candidate for a verbatim-span family later. |
| `babilong` (+ browse trees) | 11,854 | 0 | 0 | 0 | protected evaluation (showcase benchmarks). Not usable. |

Predecessor figures below (the 7b99d86d source before correction, matched to the correction ledger by id) come from a scan of the
first 86 of its 118 source files that the memory ledger preempted; they are lower bounds on withdrawals, not totals.

Totals: 3,995 of 100,840 corrected-mix rows (4.0%) are bgkit-store derived, and 6,782 (6.7%) are bgkit or KR-reader
derived; all of them are scheduled, none earned. Predecessor-to-corrected withdrawals in the KR namespaces were
heavy (in the scanned part: hotpotqa 1,006 of 1,798, memory-episode 238 of 261, chatqa2 192 of 194, quality 110 of 566), from the template-reader,
near-duplicate and wrapper-collapse caps of the 2026-11-03 recipe review.

## 3. RL reader supply (new)

Source module `scripts/rl_tasks/sources/bgkit_reading.py`, registered in `scripts/rl_tasks/SOURCES` as `bgkit-searchqa`,
`bgkit-haystack`, `bgkit-pubmed`, `bgkit-narrative`, `bgkit-memory`. Every case is a callable folder (root plus helpers), not a
single wrapper call; `Program` and `_case` are the composed-structure builders (`composed.py`).

| Family | Programs (distinct call graphs) | Constructs | Oracle |
|---|---|---|---|
| `bgkit_searchqa_reading` (Jeopardy clues, ~12 web snippets each) | snippet-notes-vote, snippet-candidates-judge, snippet-groups-tally (`iterateOn`, typed vote counts), clue-note-then-snippets (a Neuralese note relayed into every snippet reader), propose-then-verify (typed proposals relayed to a verifier), score-then-read-top (typed `number` scores, sort and filter in code, read only the top four) | per-snippet Neuralese fan-in, typed `Neuralese<{...}[]>`, tally and sort in code, `iterateOn` with a measure | span F1 0.8 on SearchQA's answer; aliases without parentheses/article |
| `bgkit_haystack_reading` (bAbI stories hidden in wikitext/PG-19/fineweb text) | haystack-state-reducer (`iterateOn` world state), haystack-fact-finders (typed fact lists to a settler), haystack-plan-then-spot (plan relayed to chunk readers), haystack-triage-then-spot (boolean flags filter chunks in code) | `iterateOn` over raw chunks, typed fan-in, relay, filter | normalized exact; the story facts a pattern finds must equal the generator's fact count |
| `bgkit_narrative_reading` (NarrativeQA plot summaries) | story-passage-finders, story-notes-reducer (`iterateOn`), story-outline-then-passages (outline relayed), story-triage-then-passages (boolean triage) | typed fan-in, reducer, relay, filter | span F1 0.8 with the other reference answers as alternates; the answer must occur verbatim in the summary |
| `bgkit_memory_reading` (dated chat sessions, answers by a bgkit teacher) | session-recall-judge, session-notes-reducer (`iterateOn`, newest wins), session-triage-then-recall | typed per-session fan-in, reducer, filter | span F1 0.8; kept only when the answer occurs verbatim in a session and is not an abstention |
| `bgkit_pubmed_reading` (1,000 expert-labelled abstracts) | abstract-stance-readers (typed per-section stances to a verdict), abstract-evidence-reducer (`iterateOn` over typed evidence for and against) | typed fan-in, reducer | exact yes / no / maybe |

Reference children (privileged-context notes for the Neuralese producers) are `checked` (the sentence that holds the gold
answer, the bAbI facts the pattern found) except PubMed, whose per-section stances are `mechanical:decisive-section-overlap`
(the expert label is a verdict on the whole abstract). Producer sites without a note (the plan/clue/outline relays, whose
content is not derivable from the data) are reported as uncovered, never invented.

## 4. Qualification, custody and repetition

Builds: `/mnt/external/natlang-rl-bgkit-kr-20261103/build/bgkit-*/raw-cases.jsonl`. Verification ran through `verify.mjs`
(main ts-host, `NATLANG_CALL_STORE=off`): reference replay, four meaningful wrong answers per case (other items' gold,
perturbations) that must all be rejected, and alias probes (aliases are measured, not a keep condition).

| Source | Built / kept | Wrong accepted / tried | Alias rejected / tried | Structure smoke (runs / shapes) | Assembled train |
|---|---|---|---|---|---|
| `bgkit-searchqa` | 1,580 / 1,579 | 1 / 6,320 | 0 / 1,736 | 228 / 96 | 1,499 |
| `bgkit-haystack` | 1,072 / 1,072 | 0 / 3,216 | 0 / 1,072 | 471 / 177 | 996 |
| `bgkit-narrative` | 980 / 980 | 0 / 3,920 | 0 / 1,594 | 184 / 65 | 900 |
| `bgkit-memory` | 419 / 419 | 0 / 1,676 | 0 / 419 | 84 / 32 | 388 |
| `bgkit-pubmed` | 999 / 999 | 0 / 2,997 | no probes (3-valued label) | 107 / 40 | 951 |

The smoke runs the intended orchestration (Neuralese arguments into children, typed Neuralese arrays, `iterateOn` over
Neuralese state) through `executeProgram`; it qualifies structure only, not that a model uses it.

Custody. Splits come from the upstream split where one exists (SearchQA validation, bgkit `eval` rows of babi-filler and
NarrativeQA become dev/confirmation by hashed unit) and a 3% + 3% unit carve-out for the memory and PubMedQA items. Units
are the clue, the story (every filler variant of a bAbI story shares `babi-story:` groups with the SFT converters), the
summary document, the chat episode and the abstract. SearchQA clues are screened against the protected bgkit question
hashes. The LFM3 held-set group closure (`union_groups_fast.py`, 5,049 rows) found 4 conflicts, all `babi-story` groups
of haystack rows, which were excluded; `ProtectedWorlds` plus this source's own held rows dropped none further.
Assembled source: `/mnt/external/natlang-rl-bgkit-kr-20261103/natlang-rl-bgkit-kr-20261103/` (`cases.jsonl` 4,734 train
rows, levels 0/1/2/3 = 1,479 / 1,997 / 1,106 / 152; `held/dev.jsonl` 146, `held/confirmation.jsonl` 165; `qualification.json`,
`receipt.json`, `repetition*.json`). Not registered; the held packets need `evaluation-only-never-training` registration
before use, and the pool is not promoted from here.

Reference notes for Neuralese producer sites (`child_notes.coverage`): SearchQA 1,749 of 1,998 sites covered, haystack 1,244 of
1,494, narrative 1,125 of 1,350, memory 515 of 515 (all `checked`); PubMed 474 of 1,428 (`mechanical`; the evidence-reducer
program has no notes). Uncovered are the plan, clue and outline relays and the PubMed reducer, whose content is not derivable
from the data and is never invented.

Repetition (`measure_repetition.py`, declared text of train rows; the new `--against` option reports overlap with a pool):

| | rows | distinct / effective instruction texts | distinct / effective structures | multi-function | Neuralese fan-in | `iterateOn` | typed Neuralese result |
|---|---|---|---|---|---|---|---|
| new source | 4,734 | 52 / 48.4 | 10 / 7.8 | 100% | 43.3% | 28.0% | 71.5% |
| latest pool | 11,275 | 3,730 / 1,278 | 126 / 32.4 | 65.5% | 11.9% | 7.2% | 4.9% |
| pool + new | 16,009 | 3,782 / 891 | 129 / 35.0 | 75.7% | 21.2% | 13.4% | 24.6% |

Overlap with the pool: 0 of 52 instruction texts and 0 of 52 program texts occur in the pool, 0 rows share a functional
program with it, and all 5 families are new to it; 7 of 10 declared structures already exist in the pool (3,403 rows).
**The honest finding is that this supply is world-diverse but program-poor**: 4,360 distinct split units, but only 52 program
texts (about 90 rows each), so adding all of it to the pool cuts the pool's effective instruction and program counts by
about 30% (instruction texts 1,278 to 891, programs 1,459 to 978; first version with 37 programs: to 784 and 860), while
raising Neuralese fan-in, `iterateOn` and typed-result shares two to five times and structure diversity (32.4 to 35.0
effective). Recommendation to the pool owner: admit it through a per-program row cap (about 40 rows per program, roughly 2,000
rows) or a per-source quota, rotating worlds across rounds, rather than as a flat 4,734-row block. More distinct programs per
store are the real remedy (next action K1).

## 5. What remains unused and why

- NewsQA, MS MARCO, KILT, PubMedQA-artificial: no article/passage text locally (browse trees carry ids only) or no exact answer.
  Next action if wanted: download the upstream passages, then reuse `SearchQA`-style snippet programs; owner of the decision:
  KR supply strand. Not scheduled.
- `git_history` (3,599) and `qa_pairs` (114,598): free-form LLM answers. Real value is as question seeds for git-service
  tasks (commit-introducing-change, file-at-commit, blame line) whose oracle is `git` on `bgkit-data/repos` (12,083 bare
  repositories). Next action: a `git` service source under `scripts/rl_tasks/sources/codebase_services.py`.
- Operator data is the heaviest bgkit consumer; nothing to change there.
- `bgkit-knowledge-rationalized` views (67,325 / 36,913 graphs) beyond the 2,615 rest27 chain graphs: not in the mix. Their
  hotpot-derived content is already RL-covered through the upstream readers and enters operator `extract` cases (fallow
  register C4).

## 6. Reproducing

```
python3 -I scripts/rl_tasks/build.py bgkit-searchqa OUT      # likewise bgkit-haystack|pubmed|narrative|memory
scripts/rl_tasks/verify_source.sh-style verify.mjs ...        # reference, negative, alias replay (frozen ts-host)
node scripts/rl_tasks/structure_smoke.mjs TS_HOST OUT/bgkit-X/raw-cases.jsonl smoke.json 3
python3 /home/werg/data/broad-rl-tasks-20261031/union_groups_fast.py all-verified.jsonl held-set-union.json
python3 scripts/rl_tasks/assemble.py --build OUT --out <new dir> --legacy-per-family 0 --train-cap 5000 --exclude-ids held-set-union-exclusions.json
python3 scripts/rl_tasks/measure_repetition.py --cases <cases.jsonl> --against <pool cases.jsonl> -o repetition.json
```
