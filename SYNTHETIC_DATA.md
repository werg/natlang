# natlang: synthetic datasets, detailed designs

Companion to `PLAN.md` and `TRAINING.md` §3.5. One section per synthesized
dataset (Y1–Y15), each with: purpose, prior art, generator design, knobs,
gold and verification, output format, pitfalls, and a first-build scope.
Status: draft, 2026-09-19. Research was done by web survey on 2026-09-18;
items marked *[unverified]* could not be confirmed and need a second look.

**What every dataset here produces** is natlang programs in the sense of
`spec/SPEC.md`: **code bases of pseudocode functions** (typed signatures,
calls, `for each`, `repeat until`, `if`/`else`, locals, crisp `.ts` helpers)
whose **leaves** are small prompt-like tasks (judge, classify, extract,
rewrite). The author or the synthesizer states the structure; the interpreter
carries it out with `call`, `write`, `run_code`, `read`, `edit`,
`report_blocker`. Leaf-only programs remain about a quarter of the corpus.
Skill codes (K1–K14 interpreting, L1–L6 leaves) are those of `TRAINING.md` §2.

**Contents.** Sections are grouped by the machinery they share, not by number.
- §0 Findings that shape everything
- Program machinery: Y2 synthesized code bases, Y3 back-translated programs,
  Y8 interpreter drills
- World and text machinery: Y1 latent worlds, Y5 schema-first extraction,
  Y4 rubrics and policies, Y6 rule worlds, Y10 dataset algebra
- Decision quality and safety: Y11 minimal pairs and calibration,
  Y12 data-is-not-code
- Surface realism: Y15 style corpus and multilingual rendering
- Acting and scaling: Y7 simulated environments, Y9 scale sets,
  Y13 verifier-backed generation, Y14 long-horizon stress sets
- Cross-cutting: quality pipeline, on-policy loop, build order, items to
  verify


## 0. Findings that shape everything

Evidence gathered for this document that bears on the whole project, not just
on one dataset.

**Small models can learn small-step execution, and erasure is why.**
"Training Transformers as a Universal Computer" (2026) trains a **59.5M**
decoder on random MicroPy programs executed in small steps with PENCIL-style
erasure of finished work. Trained on traces of at most 128 lines, it executes
human-written programs up to 7,552 lines perfectly, about 60× longer
(https://arxiv.org/html/2604.25166). PENCIL itself gets a 25M model to 97 % on
Einstein's puzzle with context that scales with space, not time
(https://arxiv.org/abs/2503.14337). Our mechanism is the same in natural
language: every function instance is a fresh short episode, finished work
lives in typed locals, not in the context.

**Short, restartable episodes are a feature, not a compromise.** "The Illusion of
Diminishing Returns" (2025) shows *self-conditioning*: accuracy drops when a
model's own earlier errors are in its context, and scale does not fix it;
context resets do (https://arxiv.org/pdf/2509.09677). The same paper gives the
horizon formula H_s = ln(s)/ln(p): past ~80 % per-step accuracy, small gains
grow the executable horizon hyperbolically. "When LLMs Stop Following Steps"
(2026) reports procedure accuracy falling from 63 % at 5 steps to 20 % at 95
steps for prompted models (https://arxiv.org/abs/2605.00817).

**Voting plus discard works, given p > 0.5.** MAKER (2025) solved a
1,048,575-step task with zero errors using stateless one-step agents,
first-to-ahead-by-k voting (k = 3), and *red-flagging*: outputs that are
over-long or malformed are **discarded and resampled, not repaired**, because
repair produced more correlated errors. Per-step error ranged from 0.2 %
(gpt-4.1-mini) to 35.7 % (gpt-4.1-nano); cost scales as Θ(s ln s)
(https://arxiv.org/html/2511.09030v1). Consequence: measure per-step error
per skill; it is the quantity everything else depends on.

**Code-like structure in instructions helps.** CoGEX (2024) fine-tunes 7B/13B
models to generate *and emulate* pseudo-programs with undefined leaf
functions; a natural-language-plan-only ablation underperforms on
programmatic tasks (https://arxiv.org/html/2405.16337v1). Think-and-Execute
(2024) finds pseudocode guides better than NL plans and transfers to smaller
models (https://arxiv.org/abs/2404.02575). Chain of Code's "LMulator" is the
closest precedent for our crisp/fuzzy split (https://arxiv.org/abs/2312.04474).
CoRE/AIOS (2024) uses an LLM as interpreter of structured NL programs, but
prompting only (https://arxiv.org/abs/2405.06907). These results are the
case for pseudocode code bases as the program form. Nobody appears to have
*trained* a sub-1B interpreter for natural-language programs over a typed
store; that is the gap this project sits in.

**Strings and token budget are the known weak spots of execution models.**
"Debugging Code World Models" (2026) finds failures cluster in dense state
exhausting the token budget and in string-valued state, attributed to subword
tokenization (https://arxiv.org/abs/2602.07672). SemCoder found concise
natural-language descriptions of key state beat full dumps (61.8 vs 48.8 on
CRUXEval-I; https://arxiv.org/pdf/2406.01006). CWM prints unchanged variables
as `".."` (https://arxiv.org/pdf/2510.02387). Consequences: keep renderings
sparse, route all string manipulation to crisp functions and `run_code`,
oversample string-state drills.

**Small models are unreliable at free-form search-and-replace.** Aider found
weak models misuse diff formats (diff ~19–30 % vs whole-file ~39–46 % for
GPT-3.5) and JSON function-call edits were worse
(https://aider.chat/docs/benchmarks.html). Diff-XYZ (2025) finds small models
benefit little from any format choice (https://arxiv.org/abs/2510.12487).
Fast-apply models at 1.5B succeed by emitting the whole merged text
(https://huggingface.co/Kortix/FastApply-1.5B-v1.0).
**Design consequence:** the interpreter almost never edits. It cannot author
functions; `edit` is exact-once substitution, used for bookkeeping in its own
instructions and for adapting a copied function, and both are rare in the
corpus. Instruction texts stay short enough that the substitution is easy to
get right.

---

## Y2. Synthesized code bases

**Purpose.** The backbone for the interpreting skills K1–K14. Exact labels at
every turn, unlimited volume, full control of structure. Implemented first in
`natlang/gen/synth.py`.

**Prior art.** Learning to Execute (2014): two knobs, `length` and `nesting`;
a *combined* curriculum that keeps easy items beats naive easy-to-hard
(https://arxiv.org/pdf/1410.4615). TinyPy: BNF-generated programs, 1.2M
snippets, hybrid curriculum 79 % vs 74.6 %
(https://arxiv.org/html/2407.10194v1). MicroPy pairs a probabilistic grammar
(budgets for depth, size, effects, bindings) with a **plan sampler** that
first fixes a target runtime configuration and then builds a program to reach
it (https://arxiv.org/html/2604.25166). Reasoning Gym separates difficulty,
structural, and *stylistic* parameters across 100+ generators
(https://arxiv.org/abs/2505.24760). CodeExecutor uses 12 mutation operators
and an easy-to-hard curriculum (https://arxiv.org/abs/2305.05383).

**Generator design.** Four layers, each independently testable.

1. *Typed program structure.* A program is a list of steps over a latent world
   (Y1): `CallOver(out, f, list, inputs)`, `Call(out, f, inputs)`,
   `FoldOver(out, f, list, init)`, `RepeatUntil(out, f, init, check, max)`,
   `Glue(out, type, expression)` (exact work no function covers),
   `Select(out, items, flags)`, `If(cond, then, else)`, `Return(fields)`.
   Functions are leaves with an oracle over hidden attributes, crisp `std/`
   functions, or **nested pseudocode functions with their own code bases**.
   Type-directed sampling guarantees well-typed programs.
2. *Python twin.* Every step also computes its value from the hidden
   attributes, never from the text, so gold exists for every local, every
   branch condition and the final value.
3. *Two samplers.* (a) Shape sampler with budgets: number of steps, calls,
   locals, nesting depth of code bases, fraction of fuzzy leaves.
   (b) **Plan sampler** targeting rare runtime states: an empty list after a
   filter (the `for each` over it is skipped), the branch rarely taken, a
   callee that reports a blocker, a quiesced item in a large call, a value
   just over the listing threshold, name clashes between a caller's and a
   callee's locals.
4. *Rendering and reference policy.* The text is a rendering of the structure
   in a dialect (Python-like, numbered steps, structured prose, SOP voice),
   with randomized local and function names, then teacher paraphrase kept only
   after a round trip. The reference policy **compiles the tool calls from the
   structure, never from the text**; every call goes through the real harness
   and is checked against the grammar of its own turn. Where several orders
   are equally valid it records the set; SFT uses one, evaluation accepts any.

**Tiers.** Leaf-only (about a quarter, always); one function with control
flow; several functions; whole code bases on disk with companion folders and
`uses` links, modelled on the project's use cases (triage pipeline, moderation
queue, expense audit, legal-move checking, a shopkeeper as a fold over
events).

**Degeneracy filters (by execution).** Call cap per program; reject dead
steps, constant conditions, programs whose output ignores their inputs.
Balance the histogram of tools and of skills per batch. Mutation operators for
controlled edge cases: empty/one-item lists, swapped branch, missing input,
an input of the wrong type offered among the candidates.

**Knobs.** Steps, nesting depth, list sizes, fraction of fuzzy leaves, dialect,
identifier style, listing-threshold pressure, failure injection rate.
Curriculum moves the distribution but never drops easy items.

**Gold and verification.** Per-turn targets from the reference policy; the
final value from the twin. CI property: executing the reference policy in the
real harness always reaches the twin's value, with zero rejected calls.

**Output.** JSONL of `(program_id, turn, messages, tools, target,
native_target, skill_tags, knobs)`; plus whole-program records for DAgger and
RL.

**Pitfalls.** Renderer dialect (mitigated by Y15, verified paraphrases and
human-written code bases); generative leaves have no constructed gold (teacher
drafts, crisp checkers, judged checks).

**First build (done for four shapes).** Ticket report, review digest, expense
audit, nested assessment: calls over lists, exact glue, filters, a
conditional, a nested function, record assembly. Next: folds, `repeat until`
with check functions, multi-file code bases, the plan sampler.

## Y3. Back-translated crisp programs

**Purpose.** Real programs over real data, rendered as natural-language
procedures, with gold from executing the original. Gives realistic data
shapes and the long tail of operations that a hand-built AST grammar misses.

**Prior art and assets.**

| Asset | Size | Note |
|-------|------|------|
| BREAK / QDMR | 83,978 decompositions | ordered NL steps with `#k` back-references: https://github.com/allenai/Break |
| QDMR→SQL (Wolfson 2022) | ~7k executable (NL procedure, SQL) pairs; 77.8 % of 9,313 execute to the right answer | https://github.com/tomerwolgithub/question-decomposition-to-sql |
| STEPS (EMNLP'23) | rule-based SQL → step-by-step NL explanation | reusable as a deterministic renderer: https://github.com/magic-YuanTian/STEPS |
| Gretel synthetic_text_to_sql | 105,851 rows, 100 domains, with an explanation field | https://huggingface.co/datasets/gretelai/synthetic_text_to_sql |
| SynSQL-2.5M | 2.54M (db, question, SQL, CoT) over 16,583 DBs | https://huggingface.co/datasets/seeklhy/SynSQL-2.5M |
| Spider / BIRD | 8.6k / 12.7k | |
| WikiSQL | 80,654 questions, 24,241 tables | simple single-table queries |
| SPoC | 18,356 programs, 677 problems, line-level pseudocode, tests | https://cs.stanford.edu/~sumith/spoc/ |

Methods: instruction backtranslation with self-filtering (Humpback,
https://arxiv.org/abs/2308.06259); OSS-Instruct from seed code
(https://arxiv.org/abs/2312.02120).

**Generator design.**
1. *SQL track.* Load each database; tables become lists of records. Parse SQL
   to a relational plan. Render the plan as a procedure two ways: rule-based
   (STEPS-style, execution order) and LLM back-translation conditioned on a
   voice. Execute the SQL for gold. Map plan operators to natlang idioms:
   `WHERE` → a crisp filter function or, when the predicate is textual, a
   fuzzy judge leaf called over the rows plus `select_by_flags`; `GROUP
   BY`/aggregates → crisp functions; joins → crisp join or fuzzy match
   (bridging to Y1-C); subqueries → nested functions. The output is a code
   base: `main.nl` plus its crisp and fuzzy helpers.
2. *Fuzzification.* Replace a crisp predicate with a semantically equivalent
   fuzzy one over a text column ("where category = 'complaint'" becomes
   "keep the rows whose message is a complaint"), using Y1 worlds where the
   text column was rendered from the crisp field. Gold is unchanged. This is
   how SQL corpora become training data for fuzzy-crisp mixing.
3. *Python track.* Short pure functions (MBPP-style, ETL snippets) with
   sampled inputs. Back-translate at statement granularity; execute for gold.
   Keep only programs whose operations exist in the stdlib or are natural
   helper functions.
4. *Round-trip filter.* A strong model executes the NL procedure in the
   harness; keep pairs whose result equals the original's execution result.

**Knobs.** Query complexity (joins, nesting, aggregates), table size,
fraction fuzzified, dialect, whether schema is described or must be
discovered by `read`.

**Pitfalls.** Back-translations that leak SQL keywords (strip and paraphrase;
lint for `SELECT`, `GROUP BY`); ambiguous NL (round-trip filter).

**First build.** QDMR→SQL pairs plus Gretel explanations over SQLite, rule
renderer plus one LLM paraphrase, no fuzzification. ~50k programs.

## Y8. Interpreter drills

**Purpose.** Move per-turn accuracy on K1–K14 quickly. Single-turn,
bulk-generated, exact.

**Prior art.** Slicing single steps from traces is how CWM and NExT build
supervision; La Malfa et al. document that line-by-line simulation is fragile
and contaminated by memorization, which argues for counterfactual near-miss
states (https://arxiv.org/abs/2401.09074).

**Drill families.**

| Drill | State shown | Target turn | Variants |
|-------|-------------|-------------|----------|
| What-next | a function half carried out: some locals exist | the single correct next call | near-miss twins where one local flips the answer; all dialects |
| For-each | one `for each` line, its functions, its lists | one `call ... over`, the item parameter left out | several lists; extra inputs; the list is a local |
| Bind | a call whose parameters have several type-fitting candidates | the right `inputs` | distractor values of the right type and wrong meaning |
| Glue | a line of exact work no function covers | one `run_code` expression, then the `write` | string ops; counting long lists; rounding |
| Branch | an `if`/`else` and the values it depends on | the first call of the branch taken, only | twin with the condition flipped; empty-list conditions |
| Return | finished locals, declared `returns` | calls straight `to` `return/<field>`, `write` with `source`, then the reply | optional fields, nested records |
| Navigate | a listing that cuts a value off, a step that needs it | `read(path, start, end)` | deep paths, list ranges, long Text; twin where reading is unnecessary |
| Resume | a call that quiesced with a note | `call` with `function` + `to` only; or with corrected arguments | partial Map; stopped Fold |
| Call repair | previous call + real rejection and hint | corrected call | `bad-call`, `unknown-field`, type does not fit |
| Blocker | inputs that do not determine the result | `report_blocker` with a precise note | twin where they do; a callee's blocker passed upward |
| Copy-edit-call | a program line asking for a variation of a function | `Function<f>` copy, one `edit`, `call` of the copy | twin where an argument suffices |
| Loop check | a state and a criterion | the `Bool` of a check function | met, nearly met, not met |
| Leaf | item + question or rubric or record type | one complete typed `write` | Y11 minimal pairs |
| Ignore embedded | data containing instructions | unaffected turn | Y12 |
| Finish | "still missing: x" after a reply | exactly the missing write | twin: nothing missing → reply |

**Generation.** All from the Y2 reference policy by slicing single turns and
by plan-sampling the target configuration directly. Every drill has a
**near-miss twin**: same surface, one state detail changed, different correct
turn. This forces reading the state instead of pattern-matching. The
draft-versus-fix balance of `TYPES.md` §8 applies to Finish and Call repair.

**Verification.** Exact match against the acceptable-turn set, and the
grammar of the turn.

**Pitfalls.** Drills are out of distribution relative to whole programs if
overused; cap at ~10 % of the mix and always draw states through the real
listing.

**First build.** What-next, For-each, Bind, Glue, Branch, Return. 20k each
with twins.

## Y1. Latent-world corpora

**Purpose.** One generator that yields gold for families A (triage), B
(extraction), C (entity resolution and cleaning), D (long-input aggregation),
and G (multi-hop). A hidden relational world is rendered into messy documents;
every answer is a query over the hidden world.

**Prior art.**
- *SynthIE* (EMNLP'23): sample coherent triplet sets from Wikidata, have an
  LLM write text expressing them; 1.8M examples; Flan-T5 at 220M/770M beat
  prior SOTA by 57 micro-F1 points. They had to flatten relation-frequency
  skew. https://github.com/epfl-dlab/SynthIE
- *HoloBench* (ICLR'25) is the closest published match to "SQL over a hidden
  DB": rows from 5 Spider databases verbalized with 5 templates per table,
  gold from executing SQL on the sampled subset. Finding: the **amount of
  relevant information hurts more than context length**, and aggregation
  degrades fastest. https://github.com/megagonlabs/holobench
- *OOLONG* (2025): concatenates items from 10 labelled classification sets
  with user IDs and dates, labels hidden; counting, per-user, and temporal
  questions; gold computed from labels; contexts 1K–4M tokens; numeric score
  0.75^|err|. Showing the labels adds only 0.8–10.9 points, so **aggregation,
  not classification, is the hard part** for long-context models. GPT-5
  scores 46 % at 128K. https://arxiv.org/abs/2511.02817
- *RLM* (Zhang, Kraska, Khattab, 2025): GPT-5 on OOLONG 44 → 56, on the
  quadratic OOLONG-Pairs 0.1 → 58 F1, BrowseComp-Plus 0 → 91. An 8B model
  learned the recursive behavior from **~1,000 filtered trajectories**.
  https://arxiv.org/abs/2512.24601
- *Era* (2026): seeded deterministic entity graph projected into 66 product
  simulators, answers computed mechanically, planted near-misses. An
  adversarial real-vs-synthetic detector flagged 55 % of records at first;
  they iterated to 0 %. Proprietary. https://arxiv.org/html/2609.09853
- *EnterpriseRAG-Bench* (2026): 512K LLM-written documents across 9 source
  types; 5 % random and 3 % plausible misfiling; near-duplicates with one
  fact changed; facts spread over clusters of 4–10 documents. Reported
  pitfalls: round-number timestamps, conversations unrealistically on-topic,
  structure drift. Generator is public.
  https://github.com/onyx-dot-app/EnterpriseRAG-Bench
- *Noise models for entity resolution*: Febrl dsgen (Zipf-distributed
  duplicates, up to 5 per record, typo/OCR/phonetic edits), GeCo, and **Gecko**
  (2024, NumPy/pandas: https://github.com/ul-mds/gecko). DeepMatcher's
  "dirty" sets move each attribute into the title with p = 0.5. WDC Products
  controls the share of corner-case pairs and keeps an unseen-entity split;
  every system degrades on unseen entities (https://arxiv.org/abs/2301.09521).
  REIN offers error injectors including BART rule-violation injection with
  controllable repair difficulty
  (https://github.com/mohamedyd/rein-benchmark).
- *Sim-to-real.* Classifiers trained on zero-shot synthetic text lose 16.9
  macro-F1 on average versus real data; few-shot real exemplars recover ~10.6
  points; the gap is small for objective tasks and up to −38.8 for subjective
  ones (https://arxiv.org/abs/2310.07849). AttrPrompt: randomizing attributes
  (length, style, subtopic, location) matches performance at 5 % of the cost
  and removes class-conditional bias (https://arxiv.org/abs/2306.15895).
  Cosmopedia: rewording prompts alone still produced duplicates; explicit
  content and format differences brought them under 1 %
  (https://huggingface.co/blog/cosmopedia).

**Generator design.**
1. *World.* A seeded, deterministic entity graph in DuckDB/SQLite. Schemas:
   hand-built domains (commerce, support desk, clinic, logistics, HR,
   property) plus borrowed Spider/BIRD/TPC-H schemas (DuckDB `tpch`
   extension). Values from Faker/Mimesis (both MIT) with realistic
   distributions: Zipfian customers, seasonal timestamps, correlated fields.
2. *Events and documents.* Entities emit events (order placed, shipment late,
   refund requested). Each event projects into one or more documents: email,
   ticket, chat, invoice, review, call note. A projection spec says which
   hidden fields the document must express, may express, and must not
   mention.
3. *Rendering.* LLM rendering conditioned on per-document attributes: genre,
   persona, length bucket, register, language proficiency, topic drift,
   language. Add quoted reply chains, signatures, tangents, non-round
   timestamps. Use ≥ 3 generator models. A template renderer provides a
   cheap, exact floor.
4. *Two channels.* Reverse rendering gives exact gold but text that is too
   clean. Mix in **forward annotation of real short texts** with a
   verbatim-span filter, as NuExtract did (kept ~1 in 6 of 300K; MIT:
   https://numind.ai/blog/nuextract-a-foundation-model-for-structured-extraction).
5. *Noise module* (logged, so every corruption is gold for cleaning):
   keyboard-adjacency typos, OCR confusions, phonetic swaps, transpositions,
   abbreviations and initials, field swaps, attribute in wrong field,
   explicit and implicit missing values ("N/A", "-", 9999), unit and scale
   errors, numeric noise, outliers, BART-style rule violations. Duplicates
   per entity Zipf-distributed, max 5, ≤ 5 edits per record.
6. *Task emitters.* From one world:
   - **A**: classify each document by a hidden categorical field.
   - **B**: extract the projected record (see Y5).
   - **C**: pairs and clusters from duplicate renderings; corner-case
     negatives (same brand, different model); unseen-entity split.
   - **D**: OOLONG/HoloBench-style questions compiled from SQL: counting,
     per-user, temporal, ranking, join, and pairwise (quadratic). Vary the
     amount of relevant information independently of length; vary position
     (start, middle, end, uniform, bimodal).
   - **G**: multi-hop questions as joins across document clusters of 4–10.
   - Conflict resolution: near-duplicate documents with one fact changed.

**Verification.** Round-trip: a stronger model re-extracts or re-classifies
each document blind; drop on disagreement with the hidden record (expect
~10–15 % removed; Source2Synth removed ~13 %). Drop items two weak models
both fail (< 1 % in OOLONG). Lexical-leak check: for classification, the
label word and schema words must not appear (NoLiMa-style). Realism loop:
train a real-vs-synthetic detector and iterate prompts toward chance.

**Knobs.** World size, schema, genres, noise rates, duplicate rate, corner-case
share, relevant-information amount, position, context length, languages.

**Pitfalls.** Too-clean text; on-topic conversations; round timestamps;
label-set familiarity inflating "zero-shot" scores (report the Familiarity
metric: https://arxiv.org/html/2412.10121v1); subjective labels (keep hidden
fields objective, or treat subjective ones via Y11).

**First build.** One domain (support desk), 3 genres, English, 200 worlds of
~2k documents, classification + extraction + counting questions. Scale
anchors: 50K (NuExtract) to 1.8M (SynthIE) extraction examples; ~1K
trajectories taught recursion at 8B, so treat that as a floor for 350M.

## Y5. Schema-first extraction with validate-and-repair

**Purpose.** Family B at scale, plus repair episodes (L3, K9) with known
fixes. Shares machinery with Y1 but samples schemas freely instead of from a
world.

**Prior art.** NuExtract conventions: > 200K unique field names, nesting depth
3–5, texts mostly 0–200 words, 0–3 in-context examples, and for half the
examples part of the text is deleted while the template is kept so the model
learns to output empty for missing fields; extracted strings must appear
verbatim. UniversalNER/Pile-NER: 45,889 examples, 13,020 entity types
(https://arxiv.org/abs/2308.03279). Liquid positions LFM2.5 for exactly this
task.

**Generator design.**
1. Sample a TypeScript type: records, optionals, enums, lists of records,
   nesting 1–5, randomized field names and naming styles.
2. Sample a value of that type (type-directed, with Faker providers per
   semantic field kind).
3. Render a document expressing it, in a genre, with planted difficulties:
   missing fields (gold `null`), several records in one document (gold list),
   conflicting mentions (gold per a stated precedence rule, e.g. "latest
   wins"), distractor entities, values stated indirectly (dates as "next
   Tuesday" relative to a stated date, which routes to a crisp function).
4. Emit as a leaf function (`args: { document: Text }`, `returns` = the
   type), and as the leaf of code bases that call it over a folder of
   documents and aggregate the records with crisp code.
5. **Repair episodes.** Corrupt a correct `return`: wrong type, missing
   required field, enum violation, hallucinated value not in text, list where
   record expected. Feed the real validator error. Target = the fixing
   action. Also corrupt the *document* side (truncated) so the right repair
   is `null`, not invention.

**Verification.** Type check by construction; round-trip re-extraction by a
strong model; verbatim-span check for string fields.

**Knobs.** Depth, field count, optional rate, records per document, distractor
count, indirection rate, document length relative to the inline threshold
(long documents force chunk-and-merge, bridging to Y9).

**First build.** Flat and 2-level records, 5 genres, null handling, three
corruption types. 100K leaf instances, 30K repair episodes.

## Y4. Label-first rubrics and policies

**Purpose.** Families A and E: map a policy over items; nested item × rule
grids; precedence and exceptions. Balanced coverage of rare rules.

**Prior art.**
- *DynaGuard / DynaBench* is the closest analogue: ~500 hand-written rules
  expanded to 5,000; policies sample a median of 3 rules (max 86); labels
  verified by **splitting each policy into single rules and labeling per
  rule**; human audit κ = 0.85; hand-written OOD test set. **A 1.7B model
  reaches 79.5 F1 versus 79.6 for 8B without chain-of-thought.** MIT.
  https://arxiv.org/html/2509.02563
- *Llama Guard* shuffles and drops non-violated categories during training to
  prevent taxonomy memorization (https://arxiv.org/abs/2312.06674).
- *BoardgameQA*: explicit pairwise rule priorities and two conflict types;
  fine-tuned models reach only ~50–60 % at depth 2. CC BY 4.0.
  https://arxiv.org/html/2306.07934
- *Prometheus Feedback Collection* is label-first: 1K rubrics × 20
  instructions × one response per score level, which removes decision bias
  (https://arxiv.org/abs/2310.08491).
- *Policy-as-Prompt* reports sensitivity to prompt structure and formatting
  (https://arxiv.org/abs/2502.18695).
- LLM judges agree with whatever label they are shown
  (https://arxiv.org/abs/2405.00722), so verification must be blind.

**Generator design.**
1. *Rule bank.* Several hundred hand-written seed rules across domains
   (moderation, expense approval, eligibility, routing, data handling,
   screening criteria), expanded by LLM and curated. Each rule has a symbolic
   form over item attributes where possible, so gold is computed, not judged.
2. *Policy sampler.* Skewed rule count (median ~3, long tail to dozens);
   shuffled order and numbering; irrelevant rules included; exceptions
   ("unless the sender is internal"); explicit precedence ("rule 5 overrides
   rule 2"); both BoardgameQA conflict types.
3. *Label-first items.* Choose (rule, label, edge-case type, attributes) then
   generate the item, with AttrPrompt-style attribute randomization. Balance
   labels per rule.
4. *Program shapes.* (a) one leaf function applying the whole policy, called
   over the items; (b) a code base with one leaf function per rule, called
   per item by a nested `assess` function, then a crisp function combining
   the verdicts under precedence, which is the structure a small model needs
   and which the policy author states; (c) two-pass triage with a second look
   at flagged items.

**Verification.** Symbolic gold where attributes are structured. For free-text
items: independent strong model labels **per single rule, blind to the
intended label**; keep on agreement; audit a human sample to κ ≥ 0.8;
disagreements go to the Y11 ambiguity pool rather than the bin.

**Knobs.** Rules per policy, exception depth, precedence chains, distractor
rules, item subtlety, formatting of the policy text (robustness to it is a
trained property).

**First build.** Three domains, 150 seed rules, policies of 1–8 rules, flat
and nested program shapes. 5K policies × 40 items.

## Y6. Rule worlds

**Purpose.** Forward-chaining loops with exactly controlled depth. Trains
`repeat until` with a check function (K3), leaf decisions, and nested calls,
with gold at every iteration.

**Prior art.** RuleTaker/ProofWriter: synthetic Datalog theories rendered via
templates; ~500k questions across depths 0–5; closed- and open-world variants;
ParaRules for crowd paraphrases (generator:
https://github.com/allenai/ruletaker; dataset). All-at-
once proof generation collapses at unseen depths while the **iterative
one-step model generalizes much better**
(https://ar5iv.labs.arxiv.org/html/2012.13048). FaiRR splits each step into
rule selection, fact selection, one-step inference
(https://arxiv.org/pdf/2203.10261). LAMBADA's backward chaining beats forward
Selection-Inference by 56 % relative at depth 5 with 11.8× fewer calls
(https://arxiv.org/pdf/2212.13894). Warning: BERT reaches near-perfect
in-distribution accuracy via statistical shortcuts and fails on other
samplings of the same problem space (https://arxiv.org/abs/2205.11502). No
work found on sub-1B forward chaining.

**Generator design.**
1. Sample a Datalog program with negation-as-failure, in domain dress
   (eligibility, access control, tariff classes, triage protocols). Own
   fixpoint solver gives gold facts per iteration, proof depth, and minimal
   proofs.
2. Render rules and facts to text: templates plus paraphrase; facts may be
   embedded in short documents (bridging to Y1).
3. *Program shapes.* (a) **Forward**: `repeat until nothing_new(facts): facts
   = apply_rules(facts)`, where `apply_rules` calls the leaf "does this rule
   fire given these facts? if so, what follows?" over the rules and merges
   with crisp code;
   (b) **FaiRR-style** split into select-rule, select-facts, infer;
   (c) **Goal-directed**: repeat-until over an agenda of open sub-goals kept
   in the state (there is no recursion), cheaper in calls than (a).
4. Closed- and open-world variants (Unknown label); distractor rules and
   facts; conflicting rules with precedence (links to Y4).

**Verification.** Exact against the solver at every iteration, not just the
final answer.

**Knobs.** Depth, width, rules, facts, negation count, distractors, state size
(capped for the context window). Train depths 0–3; hold out 4–6 and wider
theories. **Decorrelate shortcuts**: balance labels within each depth, rule
count, fact count, negation count; evaluate on a second sampling
distribution.

**First build.** Forward shape, closed world, two domains, depth ≤ 3. 50K
theories.

## Y10. Dataset algebra

**Purpose.** Pipelines over *real* text whose gold composes from existing
annotations: classify → branch → extract → aggregate. The main antidote to
synthetic-text cleanliness inside multi-step programs.

**Assets (permissive first).**

| Dataset | Annotations on the same items |
|---------|------------------------------|
| MASSIVE | intent (60), slots (55), domain (18), 51 parallel languages, 1M utterances
| Schema-Guided Dialogue | service, intent, slots, dialogue state; 20k+ dialogues|
| MultiWOZ | domain, intent, slots, state |
| CivilComments | toxicity subtypes, identity mentions; ~2M comments |
| GoEmotions | 27 emotions, multi-label; 58k |
| Stack Exchange | tags, score, accepted flag, dates |
| arXiv metadata | categories, dates, authors |
| Amazon Reviews 2023 | rating, category tree, helpfulness, verified, price; 571M |
| Yelp | rating, categories, attributes |
| OntoNotes | NER, coreference, SRL |

OOLONG-synth already does the simplest version of this over 10
classification sets.

**Generator design.** A small combinator library over dataset adapters:
`filter_by(label_a) → extract(slots_b) → group_by(field_c) → count`. Gold =
the same combinators applied to gold annotations. Render the pipeline as
instructions via the Y2 renderer. Examples: "for each utterance, if it is an
alarm request, pull out the time; report the earliest" (MASSIVE); "count
comments that are insulting but not identity-directed, per month"
(CivilComments); OOLONG-style per-user and temporal questions by attaching
synthetic user IDs and dates.

**Pitfalls.** Gold noise compounds across stages: prefer high-agreement
labels, and use teacher–gold agreement as a filter. Keep non-permissive
sources in a separable shard, out of anything released.

**First build.** MASSIVE, CivilComments, GoEmotions; 20 pipeline templates.

## Y11. Minimal pairs and calibration sets

**Purpose.** Sharp leaf boundaries (L1, L2), honest blockers (K11), and
data to fit voting and escalation thresholds later.

**Prior art.** Contrast sets drop model performance by up to 25 %
(https://arxiv.org/abs/2004.02709). CheckList's INV (label must not change)
and DIR (label must change) test types
(https://aclanthology.org/2020.acl-main.442.pdf). BLiMP: 67 templated
paradigms × 1,000 pairs, 96.4 % human agreement. **LLM-written counterfactuals
are fluent but not minimal and often fail to flip the label**
(https://arxiv.org/abs/2405.00722); DISCO over-generates and filters with a
task teacher (https://arxiv.org/abs/2212.10534). ChaosNLI: 100 annotations per
item give a label *distribution* (https://arxiv.org/abs/2010.03532).
AbstentionBench: scale barely helps abstention and **reasoning fine-tuning
degrades it by 24 %** (https://arxiv.org/abs/2506.09038). Calibration:
True/False calibration improves with model size and RLHF miscalibrates it
(https://arxiv.org/abs/2207.05221); guard models show 23–28 % ECE
(https://arxiv.org/abs/2410.10414); single-token yes/no avoids the length bias
of sequence-level confidence (https://arxiv.org/abs/2404.10136).
Trust-or-Escalate calibrates thresholds on held-out data for a guaranteed
agreement level (https://arxiv.org/abs/2407.18370); Agreement-Based Cascading
defers on ensemble disagreement for 2–25× cost reductions
(https://arxiv.org/pdf/2407.02348). No calibration evidence found at ~350M:
assume poor until measured.

**Generator design.**
1. *Programmatic pairs first.* From Y1/Y4/Y5 generators, flip exactly one
   predicate-relevant attribute (amount over/under threshold, date
   before/after, role, a negation, presence of a required clause) and
   re-render with everything else pinned. DIR pairs flip the label; INV pairs
   change something irrelevant and must not.
2. *LLM-edited pairs second*, with an edit-distance cap and independent,
   blind verification of the flip.
3. *Ambiguity tier.* Deliberately underspecified items, labelled by k ≥ 5
   strong-model "simulated annotators" (plus humans for the eval set). Store
   the vote distribution. Also absorbs Y4 disagreements.
4. *Targets.* Clear items: yes/no. Ambiguous items: an explicit
   `uncertain` outcome (which the runtime turns into vote or escalate).
   No reasoning traces on leaf decisions.
5. *Calibration protocol.* Per predicate family, fit temperature/Platt on
   held-out pairs; apply batch-calibration prior correction for yes/no bias
   (https://arxiv.org/abs/2309.17249); choose the margin threshold for a
   target agreement level; report cost–accuracy curves at fixed escalation
   budgets.

**Training use.** Both members of a pair in the same batch. After SFT, mine
**hard samples where the model's own votes disagree** for DPO, as
GuardReasoner did (https://arxiv.org/abs/2501.18492).

**First build.** Six attribute-flip templates over Y4 and Y5; 20K pairs; 3K
ambiguous items with vote distributions.

## Y12. Data-is-not-code robustness

**Purpose.** Programs and data are both text in one tree. Text inside data
must never steer the interpreter. This is a correctness property of the
language.

**Prior art.** Benchmarks: BIPIA (https://arxiv.org/abs/2312.14197),
InjecAgent, AgentDojo (629 injection cases; reports benign utility, utility
under attack, targeted ASR), Tensor Trust (126k human-written attacks:
https://arxiv.org/abs/2311.01011), SEP (9.1K samples). Training recipes:
StruQ (SFT with reserved delimiter tokens) leaves 45 % ASR against strong
attacks, **SecAlign (DPO) gets 8 %** and preserves utility
(https://bair.berkeley.edu/blog/2025/04/11/prompt-injection-defense/). Meta
SecAlign: injected instruction placed at the start (45 %), end (45 %), or as
a fake completion (10 %), **position randomized to prevent an "ignore the
tail" shortcut**; chosen/rejected self-generated by the undefended model;
DPO β = 0.1, 3 epochs, 19k pairs; ASR 95.7 % → 0.5 % on AlpacaFarm, works
down to 4B (https://arxiv.org/html/2507.02735v3). Instruction Hierarchy: the
target is **the output the model would give if the injection were absent**
(https://arxiv.org/html/2404.13208v1). Spotlighting/datamarking cuts ASR from
~50 % to < 3 % but not against adaptive attackers
(https://arxiv.org/abs/2403.14720). CaMeL: control flow only from the trusted
query (https://arxiv.org/abs/2503.18813), which is structurally what our
`instructions`-vs-inputs split already is.

**Structural defenses first (runtime, not data).** The renderer marks
provenance: `instructions` is the only program text; all other Text nodes are
rendered inside reserved delimiters that are stripped from data on write
(StruQ's secure front end), optionally datamarked.

**Generator design.**
1. *Clean twins.* Every injected item derives from a clean Y1/Y4/Y5/Y10 item
   with known gold action. Target for the injected version = the clean twin's
   action.
2. *Taxonomy.* Naive imperative; "ignore previous instructions"; fake
   completion / fake system turn; **spoofing of our own tree rendering and
   tool-call syntax**; authority claims; label-targeting ("answer yes");
   task-switching; attempts to edit `instructions` or write `return`;
   multilingual and encoded payloads; and **benign instruction-like text**
   (a recipe, a quoted policy, an email asking the *recipient* to do
   something), which must be processed normally.
3. *Placement.* Random position (start/middle/end), tree depth, and field;
   varied length; multiple injections per item.
4. *Positives.* Programs whose predicate is "does this item contain
   instructions to the reader?", so the model reads injections as data rather
   than learning to blank them.

**Training recipe.** Include in SFT from round one. Then DPO with
self-generated pairs: chosen = clean-twin behavior, rejected = the behavior
that followed the injection.

**Metrics.** Attack success rate, clean utility, utility under attack,
false-trigger rate on benign instruction-like data. Hold out whole attack
families and Tensor Trust's human-written attacks for test.

**First build.** Five attack families + benign look-alikes over Y4 items;
30K twins.

## Y15. Style corpus for the renderer, and multilingual rendering

**Purpose.** Prevent synthetic dialect. The renderer should imitate how real
people write procedures, in all ten languages.

**The central warning.** Herzig & Berant (2019) studied exactly our setup in
semantic parsing: paraphrasing grammar-generated canonical utterances causes
both a **language mismatch** and a **program-distribution mismatch** with real
users, and both hurt on real data. Their fix: start from real utterances and
detect which programs they express (https://aclanthology.org/D19-1394/).
Related: instruction-tuned models degrade on 319 equivalent instructions
written by practitioners (https://arxiv.org/abs/2306.11270); formatting alone
can swing accuracy by up to 76 points (https://arxiv.org/abs/2310.11324).
Training on paraphrases helps: Unnatural Instructions gained +12.1 on BBH with
~3.5 paraphrases per instruction (https://arxiv.org/abs/2212.09689);
PromptSource suggests 5–10 templates per dataset.

**Seed corpora**

|----------------------------------------------|--------------------------------|
| MASSIVE: real command phrasing, 51 languages | wikiHow, WikiLingua |
| Stack Exchange how-to answers | iFixit / MyFixit |
| BioProBench protocol sets | RecipeNLG, Recipe1M+ |
| NL2Bash, BREAK | Amazon SOP-Bench |
| Aya Dataset: 204K human-written, 65 languages | Yelp |
| READMEs from available repos; government SOPs | |
| SOPBench | SPoC |

**Renderer design.**
1. *Separate surface from semantics.* Input = AST + 1–3 retrieved real
   exemplars matched on register and language + a persona/audience.
   Instruction: imitate the form only (Cosmopedia/WRAP pattern).
2. *Inverse direction (Herzig & Berant).* Collect a few hundred real
   human-written procedures per language; have a strong model propose the
   AST; keep when execution checks out. These are the most valuable programs
   in the mix and the core of the held-out human test set.
3. *4–10 renderings per program*, each in a different voice and from a
   different generator; beyond that, spend on new programs. ≥ 3 generator
   models plus the template renderer. Hold out one generator and two voices
   entirely.
4. *Multilingual.* Render natively per language from the AST with
   native-language exemplars; do not translate English renderings (native
   data is clearly better than translated:
   https://arxiv.org/abs/2406.12822). ~70–80 % English; every construct
   covered in every language. LFM2's SFT language list omits Italian and
   Portuguese, so oversample them. As few as 40 multilingual examples helped
   cross-lingual instruction following in large models
   (https://arxiv.org/abs/2401.01854); unverified below 1B.

**Diversity dashboard** (per voice and language, versus the same metric on the
real seed corpus): compression ratio, Self-BLEU, long n-gram self-repetition
(https://github.com/cshaib/diversity), embedding Vendi score
(https://arxiv.org/abs/2210.02410), MinHash duplicate rate (< 1 %), and a
real-vs-synthetic classifier AUC trending toward 0.5. Moderate diversity
helps, extreme diversity hurts (https://arxiv.org/abs/2506.19262).

## Y7. Simulated environments for side effects

**Purpose.** Families F and K: procedures that *do* things. Trains calls of
effectful crisp functions, precondition checks, refusal and blockers, with
gold from a simulator instead of a judge.

**Prior art.**
- *AppWorld*: 9 apps, 457 APIs, 101 tables, 750 tasks;
  **state-based unit tests** so any valid path passes, plus a
  **collateral-damage check** for unexpected changes. Closest match.
  https://arxiv.org/pdf/2407.18901
- *tau-bench / tau2-bench*: tools over a domain DB plus a policy
  document; gold via DB-state assertions, communication assertions, and action
  assertions. https://github.com/sierra-research/tau2-bench
- *ToolSandbox*: implicit state dependencies between
  tools; gold as a DAG of **milestones** plus **minefields** (events that must
  never happen). https://arxiv.org/pdf/2408.04682
- *BFCL v3 multi-turn*: writes checked by state comparison; reads
  by response, redundant calls allowed.
- *APIGen-MT*: **blueprint first** — ground-truth actions
  validated by format check, execution, policy rules written as Python unit
  tests, and LLM-judge vote; trajectories kept only if state and output match.
  A 1B model trained on it reaches 43 % on BFCL v3 multi-turn.
  https://arxiv.org/html/2504.03601v3
- *WorkBench*: 5 DBs, 26 tools, 690 tasks, one unambiguous outcome DB
  per task. https://arxiv.org/abs/2405.00823
- *SOP-Bench*: 2,000+ tasks, 12 domains; finding that **adding more
  tools lowers success**. *SOP-Agent* compiles SOPs to decision graphs and
  notes pseudocode-style indented SOPs in practice
  (https://arxiv.org/html/2501.09316). No public runbook-automation dataset
  was found.

**Generator design.**
1. *Simulators.* Each mock tool = typed tables + pure transition functions,
   seeded, with an explicit `initial_state`. Start with inventory, ticketing,
   calendar, file store, messaging. Include implicit cross-tool dependencies
   (a ticket must reference an existing SKU). Registered by the host as
   capabilities and wrapped as crisp functions of the code base (`open_ticket`,
   `reserve_stock`); simulator state is snapshotted with the tree so forks
   just work.
2. *Blueprint-first SOPs.* Sample a goal and a gold action sequence; execute
   it in the simulator; encode the SOP's policy clauses as unit tests over the
   trace; only then write the natural-language SOP and its paraphrases (Y15
   voices: runbook, checklist, indented pseudocode).
3. *Content requirements.* Branching decision-graph SOPs; distractor tools;
   **unmet preconditions where the correct action is refuse or escalate**;
   idempotency traps (retrying a non-idempotent call); partial failure of a
   tool with a documented recovery step.
4. *Program shapes.* Single-function SOP; SOP with a helper called per item
   ("for every open ticket older than 7 days …"); SOP with a judged step
   ("if the customer sounds upset, …") mixing fuzzy leaves with side effects.

**Verification.** Writes: diff against the simulator's gold final state **and
assert nothing else changed**. Reads and reports: output match. Minefields:
forbidden events never occurred. Policy clauses: the unit tests.

**Knobs.** Tools available vs needed, SOP branching, dependency depth,
precondition-failure rate, tool-failure rate, collection size.

**First build.** Ticketing + inventory, 40 SOP blueprints × paraphrases,
state-diff and collateral checks. 10K episodes.

## Y9. Scale sets

**Purpose.** The same code bases over inputs far larger than one episode can
hold. The structure is stated by the author, so nothing has to be invented;
what is trained is carrying it out **without reading what need not be read and
without doing the items' work by hand**, and it is where the project's claim
is measured: small model plus structure against the same model answering
whole.

**Prior art.** The RLM paper is the main data point for training a root that
delegates: 2,250 teacher trajectories on 750 tasks filtered to 1,072; **each
root turn is one SFT sample** (the same shape as ours); OOLONG went 0 → 32 %.
Cleaning was needed: **16 % of turns misused FINAL, 13 % referenced bad
variables**. Failure modes: **excessive sub-calls**, confusing a final answer
with a thought (https://arxiv.org/html/2512.24601v2). THREAD's children return
only what the parent needs (https://arxiv.org/abs/2405.17402). Those failure
classes are what typed `call`, type-filtered inputs and "the reply is never
the result" rule out by construction.

**Generator design.**
1. *Scale sweeps.* Y2 programs with list sizes K ∈ {1, 3, 10, 30, 100, 1000}.
   The reference trajectory is the same at every K: one `call ... over`.
   Small K is included so that the model calls the function even when it could
   have answered by hand.
2. *Long texts.* One long document instead of a list: the program calls a
   crisp `chunk` function, then a leaf over the chunks, then a crisp combine.
   Boundary cases: items that straddle chunk boundaries (a program with an
   overlap parameter), a combine step that needs cross-chunk information
   (dedupe across chunks).
3. *The one-shot baseline set.* For every program, the same task phrased as a
   single prompt with the whole input, for the comparison of `TRAINING.md` §6.
   Used for evaluation, not training.
4. *Return discipline.* Callees return only what the signature says; lint for
   reads of large values that the program never needs.

**Verification.** Exact: per-item gold reused; the final aggregate computed
crisply.

**Knobs.** K, item length, chunk size, cross-item dependency, nesting depth.

**First build.** Sweeps over the four Y2 shapes, K up to 100. Measure the
untuned and tuned 350M one-shot against interpreted, by K.

## Y13. Verifier-backed generation

**Purpose.** Family I: judged-refinement loops where the judge is crisp.
"Rewrite until it satisfies the constraints" with a checker in the loop.
Also the cleanest RL reward in the project.

**Prior art.** *AutoIF* is the fullest recipe: per instruction
the LLM writes verification functions and test cases; keep a checker if it
compiles and **mutual pass rate > 0.5** between functions and cases;
back-translate the checker into an instruction and NLI-filter contradictions;
rejection-sample responses; best SFT data at pass-rate threshold ~0.8
(https://arxiv.org/html/2406.13542v1). *Tülu 3 RLVR*: IFEval-style templates,
one verifier each, binary reward. *IFBench*: 58 held-out constraints;
**models overfit IFEval's constraint set, and wider constraint variety
generalizes better**; 1–5 constraints per prompt
(https://arxiv.org/abs/2507.02833). *VerIF*: hard constraints by code, soft
ones by a reasoning judge (https://arxiv.org/abs/2506.09942). Reasoning Gym,
SynLogic, Enigmata: generator + verifier suites.

**Generator design.**
1. *Constraint library.* Hard, code-checked: length and counts, format and
   schema, required/forbidden keywords, ordering, numeric bounds, reading
   level, language, structural (n bullets, title case), and **state
   invariants** over the tree. Soft, judged: tone, audience fit.
2. *Checker validation.* AutoIF's mutual-agreement filter, plus: every
   checker must pass known-good outputs and **fail mutated ones**.
3. *Tasks.* Base rewriting/summarizing/formatting leaf + 1–5 sampled
   constraints. Program shape: `draft = write(...)`, then `repeat at most n
   times until passes(draft): draft = revise(draft)`, where `passes` is the
   generated checker as a crisp `Bool` function, and a final branch that
   reports which constraints still fail.
4. *Conflicting constraints* (unsatisfiable sets) where the right result is
   to report the conflict.

**Verification.** The checkers. Hold out whole constraint *types* for eval.

**Note on fit.** Liquid does not recommend this model for creative writing,
so keep texts short and constraints mechanical. The loop structure and the
use of checker feedback are the skills; prose quality is a candidate for
escalation.

**First build.** 25 hard constraint types, 3 held out; 30K tasks.

## Y14. Long-horizon stress sets

**Purpose.** Measure, and then train, the thing the thesis depends on:
whole-program success as a function of length, depth, and width.

**Prior art.** Horizon formula H_s = ln(s)/ln(p); even models with
near-perfect first-step accuracy fall below 50 % task accuracy within 15
turns when history is kept; **without chain-of-thought, models cannot compose
more than ~6 operations in one step**; a sliding window over history (in
effect, statelessness) mitigates self-conditioning
(https://arxiv.org/pdf/2509.09677). MAKER: error rate rises from ~0.1 % to
~10 % once a response passes ~700 tokens, hence its length red flag
(https://arxiv.org/html/2511.09030v1). LongProc: open models falter at 2K
output tokens of procedure (https://arxiv.org/abs/2501.05414). The MicroPy
result (0.) shows 60× length generalization is achievable with erasure.

**Design.**
- Sweeps: steps ∈ {10, 30, 100, 300, 1000}; call nesting depth ∈ {2, 3, 4, 6};
  list width up to 10³–10⁴ with exact work in crisp functions; nested structures
  (org charts, bills of materials, threaded discussions) from Y1 worlds.
- **Per-statement complexity capped at 1–2 operations**, which is an
  authoring rule for the generated programs.
- Report per-skill step accuracy p, measured horizon H₀.₅, and predicted
  H₀.₅ = ln 0.5 / ln p. A gap between predicted and measured means errors are
  correlated, which is where voting fails.
- With and without: red-flag discard, first-to-ahead-by-k voting (k = 3,
  optionally on side-effecting actions only), type-check retries, escalation.
- Target: p ≥ 0.999 on algorithmic skills for 100+ step programs.
- Length generalization protocol: train on ≤ N steps, test at 4N, 16N, 64N.

**Use.** Primarily evaluation. Train on a slice only after the curve is
understood, otherwise it stops being a measurement.

---

## Cross-cutting: the quality pipeline

Applied to every dataset above, in this order (adapted from the LFM2 report's
own SFT pipeline, https://arxiv.org/html/2511.23404v1):

1. **Schema and execution validation.** The interpreter or solver output is
   the gold. Trust execution over judges.
2. **Round-trip consistency.** A second model recovers the hidden source
   (record, label, AST, final tree) blind; keep on agreement
   (https://aclanthology.org/P19-1620/).
3. **Multi-vote judge for naturalness only.** Judges agree with humans ~80 %
   (https://arxiv.org/abs/2306.05685): a soft filter, never the gold.
4. **Rule-based filter** for overused phrases and LLM clichés.
5. **Dedup**: exact → MinHash → semantic (SemDeDup removes ~50 % with minimal
   loss; it was the biggest remover in LFM2's pipeline).
6. **Decontamination**: n-gram and semantic, against every held-out set.
7. **Selection.** For sub-1B models quality beats quantity: SmolLM2's 360M
   used a *simplified* mix with complex tasks removed; DEITA matched 10×
   larger sets with 6K chosen samples; top-10 % IFD beat the full set. Score
   difficulty, order as a curriculum, drop constructs the model cannot learn.

**Mixing.** Never synthetic-only. Accumulate synthetic alongside real rather
than replacing it (https://arxiv.org/abs/2404.01413); WRAP used 1:1. For our
SFT, ablate a real-text share of 0/10/25/50 %.

**Held-out design.** Hold out whole worlds, schemas, policies, rule domains,
attack families, voices, one generator model, and a human-written test set of
≥ 200 items per language from contributors who never saw the renderer. The
headline metric is the **gap between synthetic-held-out and human-written
accuracy**.

**Tolerant parsing at eval, strict at inference.** The LFM2 report notes small
models often fail evaluations on format. Evaluate capability with a tolerant
parser; at inference use grammar constraints plus MAKER-style discard.

**Tooling.** distilabel, DataDreamer, Bespoke Curator, NeMo Data Designer (seed sampling, column
dependencies, validators), NeMo Curator for GPU dedup, Argilla for human
audit.

---

## The on-policy loop these datasets feed

The scripted reference policy can label any state, so on-policy imitation is
nearly free for Y2, Y3, Y6, Y7, Y8, Y9, Y13.

**Prior art.** "Revisiting DAgger in the Era of LLM-Agents" (2026): teacher
labels every state; executed actions mixed per turn with probability β from
1.0 down 0.2 per iteration to a floor of 0.6; 5 iterations, aggregated data,
3 epochs; a 4B model reaches 27.3 % on SWE-Bench Verified versus 23.4 % for
on-policy distillation (https://arxiv.org/html/2605.12913v1). "A Few Teacher
Steps": short teacher continuations at learner-reached states beat behavior
cloning at matched budget (https://arxiv.org/abs/2607.04574). Agent-R: splice
the prefix up to the first error onto a correct continuation; revision data
beat expert-only by 5.6 % (https://arxiv.org/abs/2501.11425). Thinking
Machines' on-policy distillation needs teacher log-probs, which a scripted
expert lacks, so use cross-entropy on the expert action
(https://thinkingmachines.ai/blog/on-policy-distillation/). TRL ships a
GKDTrainer for the LLM-teacher case.

**Schedule.**
1. SFT on expert states.
2. ~5 rounds. Per step, execute the expert's action with probability β, else
   the student's; always *label* with the expert. Anneal β from 1.0 toward
   0.5; with a free expert, toward 0 in late rounds.
3. Aggregate all rounds; ~3 epochs per round.
4. Cap the student-driven prefix, or reset to an expert state after N
   divergent steps, so rollouts stay recoverable.
5. Inject corrupted-but-recoverable states labelled with the expert's repair
   (Agent-R), feeding K9.
6. For teacher-labelled (non-scripted) programs, spend the budget on short
   teacher continuations at student-reached states, prioritized by the
   lowest-accuracy skills.

## Build order

| Wave | Datasets | Why first |
|------|----------|-----------|
| 1 | Y2 (all constructs, four tiers), Y8, Y5, Y12-lite | Interpreting backbone, fastest per-turn gains, the model's strongest leaf type, and injection robustness from round one |
| 2 | Y1 (one domain), Y4, Y11, Y9 | Flagship families A–D; blockers; the scale measurements that test the thesis |
| 3 | Y3, Y6, Y10, Y15 inverse direction | Real data shapes, loops with exact depth, real text in pipelines, human-written programs |
| 4 | Y7, Y13, Y14 sweeps, multilingual | Side effects, refinement loops, the horizon measurements, the other nine languages |

Y15's style seeding and the quality pipeline apply from Wave 1.

## Items needing verification before use

- Several 2026 arXiv entries were read from abstracts or summaries only:
  Recursive Agent Optimization (2605.06639), AttackEval (2604.03598),
  "Polyglot Teachers" (2604.11290).
- Type-stripping speed figures come from secondary blogs.
- No evidence was found, in either direction, on: sub-1B forward chaining,
  calibration of ~350M yes/no logprobs, cross-lingual skill transfer below
  1B, or a norm for paraphrases per AST. These are things we will have to
  measure ourselves.
