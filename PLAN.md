# natlang: a small-model interpreter for natural-language programs

Design and execution plan. Status: draft, 2026-09-19.
The normative language definition is `spec/SPEC.md` (v0.2-draft).
Training, use cases, and datasets are covered in depth in `TRAINING.md`.
Detailed designs for the synthesized datasets are in `SYNTHETIC_DATA.md`.
The type system, validation, and validation feedback are specified in `TYPES.md`.

## 1. Thesis

Small language models (LFM2.5 230M / 350M) are not capable enough to act as
general agents, but they are fast: a 200-token step costs well under 100 ms on
a consumer GPU, and thousands of independent steps can be batched into one
forward pass. The bet is that *algorithmic structure* can keep every individual
model decision inside the competence of a small model, while composition,
explicit state, and cheap verification supply everything else. The result is a
system that runs programs no conventional language can express (fuzzy leaves:
judge, classify, extract, rewrite, route) at a speed and cost no frontier model
can match.

**Programs are pseudocode algorithms, and the structure is the author's.** A
natlang program is a code base of functions with typed signatures, subroutine
calls, control flow (`for each`, a value carried along a list, `repeat until`,
`if`/`else`), local variables and pseudocode types. The **model is the
interpreter**: it reads the pseudocode, decides what the next statement asks
for, and carries it out. It does not invent decompositions and cannot author
functions. Prompt-like single tasks remain essential, as the **leaves** of such
programs and as a retained share of the corpus.

The harness provides memory, typing, a sandbox, I/O, and a scheduler. It
parses no instructions, holds no cursor, and owns no control flow. Every
constraint it imposes is type-level and applies at write time. All
interpretive discipline (taking statements in order, one call per loop, taking
one branch, routing exact work to code, reporting a blocker instead of
guessing) is instilled through training data, not enforced by code.

Stated as arithmetic: with per-step success `p` and `n` steps, naive success is
`p^n`. A 97 % model fails half of all 25-step programs. Structure alone does
not rescue a small model; structure plus **error containment** does. Every
design decision below either shrinks a step or contains an error.

What this favors: shallow semantics × heavy structure × lots of data. Inbox
triage, normalizing messy collections into schemas, fuzzy dedup, extraction
over thousands of short texts, rule application, judged refinement loops,
long-lived reactive programs. What it does not favor: leaves that need
knowledge or synthesis beyond the model (structure cannot shrink a leaf below
the model's knowledge floor).

## 2. The language

### 2.1 The object tree and its types

The program's state, inputs, and outputs are one typed object tree. Every node
has a path (slash-separated, numeric indices for lists) and a type, fixed at
creation. Types are written in TypeScript type syntax: `Text`, `Num`, `Bool`,
`Null`, `Blob`, `T[]`, `Dict<T>`, records with optional fields, unions, enums
of string or numeric literals, named and recursive types. No `Ref` type:
values are copied, never aliased. Values travel as JSON in native tool calls
and are parsed **against the type of the slot** (type-directed), which removes
the classic ambiguities of untyped formats. Validation happens on every write;
type errors come back as ordinary tool errors with a one-line hint and are a
trained recovery case.

### 2.2 Functions, instances, and code bases

A **function** is a file: `name.nl` holds YAML frontmatter (`args`, `returns`,
`types`, `description`, `uses`, `effects`) and a body
of pseudocode, or for a leaf a plain task description; `name.ts` is a **crisp
function** whose body is TypeScript. A function's private functions live in a
companion folder of the same name (`somefun.nl` + `somefun/`). Scope is
**lexical**: a function can call the functions of its companion folder and its
`uses` links, nothing else. Code bases are **immutable** and shared by
reference. **There is no recursion**: no function can reach itself, directly
or through others; the loader refuses a code base that contains a cycle. All
repetition goes through `call` (over a list, carried along a list, or repeated
until a check holds), whose bounds the harness controls.

A **lambda** is an instance of a function, and it holds every zone of state an
episode can touch:

| Part | Written by | |
|------|-----------|--|
| `type`, `instructions` \| `code`, `types`, `effects`, `codebase` | the author | the interpreter may edit its own `instructions` for bookkeeping |
| `args` | the caller | frozen at start, read-only to the instance |
| `let` | the interpreter | typed locals `let/<name>`, created by the first write, private |
| `return` | the interpreter | checked as `Draft<T>` while drafting, as `T` at completion |
| status, note, provenance | the harness | |

Semantics:

- **There are no anonymous lambdas.** The only way to start a sub-task is
  `call`, which instantiates a function of the acting lambda's code base.
- **Completion** requires a value: `return` complete, fitting `T`, with no
  call still pending inside it. There are no tail calls. On completion the
  node is replaced by its value and the instance moves to provenance.
- **Quiescence without completion** is a normal outcome: a blocker, a reply
  without a valid result, an exhausted budget, a crash, a crisp function that
  threw. The node stays exactly as it was, with a note. The caller sees the
  note and may call again to resume. There is no separate failure
  representation.
- **Scope is the subtree.** A callee sees its own `args`, locals and code
  base, nothing of its caller. That is what makes an instance self-contained,
  shippable, and resumable.
- **To change a function**, the interpreter copies it into a local
  (`write` with type `Function<f>`), edits `let/<copy>/instructions`, and calls
  `let/<copy>`. The original is untouched; specialization normally belongs in
  arguments.

### 2.3 Type safety (summary; full design in `TYPES.md`)

- **Every function is typed by its author**; every local is typed at
  creation; a `call` derives every type it needs from the callee's signature
  and is refused if an input does not fit or a required parameter is unbound.
- **Type preservation is enforced by the harness**: every accepted action
  takes a draft-well-typed tree to a draft-well-typed tree, and completion
  takes `Lambda<P, T>` to `T`. A slot of type `T` accepts a `T` or a call in
  progress whose result fits `T`.
- **Holes are fine, lies are not.** While drafting, `return` is checked
  against `Draft<T>`: incompleteness is accepted and reported; contradictions
  are rejected at write.
- **Commit points** demand full conformance: completion, a `call`, an
  effectful call.
- **The validator is a pure function of the tree**, run after every action.
  Its report is always logged and drives error containment, trace filtering,
  metrics, and RL reward, whether or not the model sees it.
- **Feedback is state, not interruption**: what is still missing is listed in
  the workspace; blocking diagnostics are listed only after a refused commit.
  The balance between drafting and fixing is set by training data
  (`TYPES.md` §8), not by the harness.

### 2.4 Crisp functions, combinators, and control flow

The harness reads no instructions and owns no program-level control flow. It
provides a small set of **typed node kinds with fixed semantics**, all reached
through `call`.

**Crisp functions.** A function whose body is TypeScript. Calling it runs the
code in the sandbox with `args` as its typed input; no model episode is
involved. Results get provenance like everything else, and large values land
in their slot without passing through the model's tokens. A failing crisp
function quiesces like any other, with the error as its note. Crisp functions
may have side effects, subject to declared `effects`. A small linkable
library, `std/`, covers the common exact operations (`count_true`,
`select_by_flags`, `group_count`, `word_count`, ...).

**`run_code`** runs one TypeScript expression and **returns its result to the
interpreter**. It sees `args` and `locals`, may cause declared effects, and
never writes to the tree: the interpreter writes what it learned. It is for
exact glue the author did not name a function for.

**Combinators.** Node kinds, not tools. They get no episode of their own; only
their body instances do.

| `call` with | Builds | Reduces to |
|------|--------|-----------|
| `over` | `Map<A, B>`: one instance per item; the item goes to the one parameter left unbound, under the author's name for it | `B[]`, length and order preserved by construction |
| `over` + `init` | `Fold<A, S>`: a function `(acc, item) -> acc`, the harness threads `acc` | `S` |
| `init` + `until` + `max` | `Iterate<S>`: a step `S -> S`, a `Bool` check function run as a fresh episode (or crisp) after every round, a mandatory bound, crisp cycle detection by content hash | `S` |

- *Partial results*: if some body instances quiesce, the combinator quiesces
  with values in the finished slots. Calling again with only `function` and
  `to` **resumes**: only what did not finish runs.
- **Open lists and long-lived programs.** A list may be *open*: an external
  source keeps appending to it. **A reactive system is an ordinary fold over
  an open list of events**, with a step function `(acc, event) -> acc` and
  outputs emitted as effects inside the step. State size, lifetime and
  shutdown are host concerns.
- **Legal actions by construction.** A crisp function computes the legal
  options from the state; the decision is a leaf with a finite return type;
  the crisp function that applies the choice refuses anything illegal. Large
  numeric ranges are discretized into named options by crisp code.

What the interpreter is trained to do (SPEC §7): take statements in order;
`x = f(a)` is one `call`; `for each` is one `call ... over`, never one call per
item and never the items' work by hand; `repeat until` is one call with a
check; for `if`/`else`, evaluate the condition and carry out only the branch
taken; exact work goes to functions or `run_code`, never to estimation;
intermediate results go into the locals the pseudocode names; a value that
already exists is written with `source`, never re-emitted; when the inputs do
not determine the result, `report_blocker`.

## 3. The agent interface

### 3.1 Tools (six)

| Tool | Arguments | Notes |
|------|-----------|-------|
| `read` | `path`, `start?`, `end?` | a value, a range of it, or `codebase/<f>` |
| `write` | `path`, `type`, `value` \| `source` | a complete plain value into `return` or a local; `source` copies an existing value; type `Function<f>` copies a function into a local |
| `edit` | `path`, `old`, `new` | `old` must occur exactly once; own `instructions` and function copies |
| `run_code` | `code` | exact glue; sees `args` and `locals`; the result comes back |
| `call` | `function`, `to`, `inputs?`, `values?`, `over?`, `init?`, `until?`, `max?` | place and run an instance in one action; again with only `function` + `to` to resume. Present only when the lambda has functions |
| `report_blocker` | `missing` | end without a result, saying exactly what is missing |

The tool list is fixed; only argument schemas change, regenerated each turn
from the tree, the types, and the code base. **Instructions and data travel in
different channels**: the user message is the instructions, the expected
return type, and the function listing (signature plus one-line description);
the harness performs the first step, `read(path="args")`, and the workspace
arrives as a tool result. **The reply ends the episode and is never the
result**: the result is what was written to `return`; a reply with `return`
incomplete gets one line saying what is missing, twice at most, then the
lambda quiesces.

**Constrained decoding** (`natlang/native.py`). Each turn is decoded under a
grammar built from the turn's tool schemas, over the model's *native* call
text. The model chooses between calling and replying and which call to make;
it cannot name a path or a function that does not exist, put a wrongly typed
value into a slot, bind a parameter to a value whose type does not fit, invent
or omit a field, or read a range that is not there. Tools carry
`x-natlang-alternatives`, argument sets that belong together (a path, its
type as a constant, and the value grammar; a function and its parameters). A
turn's grammar is built from the state before the turn, so a call that depends
on what an earlier call created goes in the next turn. For servers that parse
tool calls themselves, decoders take per-model tool aliases.

### 3.2 Context model and rendering

**The lambda is the unit of context.** Interpreting one function instance is
an ordinary short multi-turn agent episode; each callee gets a fresh context.
What is required is **restartability: the context is a cache, never the
state.** Everything durable is in the tree (locals, `return`, the effect
journal), so a context can be dropped at any moment and a fresh episode can
continue from the tree alone. Cold restart is a core trained skill. Episodes
stay short because functions stay small, which limits the known degradation
from long contexts and from a model seeing its own earlier errors; rejected
calls that were silently resampled are not added to the context.

The rendering policy (SPEC §8, `render/0.2`) is part of the language: fix it,
version it, do not tune it per experiment. Its rules came from measurements:
short texts are shown **whole** (a cut-off rubric reads like a complete one),
lists preview three items, no value-like placeholder is ever shown, filled
parts of `return` are listed apart from what is missing, nudges carry no data.

### 3.3 Decisions and write-time typing

A yes/no or categorical decision is a write to a node whose type is finite.
Decisions that matter are their own leaf functions with a finite `returns`.
The primary reliability mechanism is the type system applied at write time
(`TYPES.md` §6); what still fails validation is discarded and resampled before
the model sees it. The distribution over members for finite-typed writes is
logged to provenance at no cost. **Deferred**, pending evidence from those
logs: voting, confidence thresholds, calibration, escalation to a larger
model, speculative execution.

## 4. The runtime

### 4.1 Components

- **Tree** with typed slot references, path resolution with scope and
  writability, `Draft<T>` validation, commit checks (in memory today; SQLite
  with content-addressed copy-on-write is the target).
- **Code-base loader** (`natlang/codebase.py`): `.nl`/`.ts` files,
  frontmatter, companion folders, `uses`, inherited types, cycle check,
  inline `codebase:` in YAML programs.
- **Runtime**: instances, crisp functions, Map/Fold/Iterate with resumption,
  swap-out, quiescence, effect journal, open lists, recursion guards (no
  identical child, at most 6 nested pending nodes, run budgets of 256 episodes
  and depth 8).
- **Tool surface** (`natlang/surface.py`) and **native constrained decoding**
  (`natlang/native.py`, portable GBNF; `natlang/gbnf.py` is a recognizer used
  to test grammars without an engine).
- **Sandbox**: QuickJS with a crisp standard library; memory and time limits.
- **Model servers**: llama.cpp `llama-server` for local work (per-request
  grammars, `/apply-template`, top-token probabilities, prompt cache), served
  with each model's **official chat template**; SGLang or vLLM for bulk
  generation and RL rollouts.
- **Scheduler** (target): batch all live instances' next turns across runs;
  the items of a `call ... over` are the natural batch.
- **Provenance**: per-instance trace, locals, effect journal, logged
  distributions; the raw material for training.
- **Grading** (`natlang/checks.py`): expected value, expected status, crisp
  checks, judge checks; structure lint beside the outcome.

### 4.2 What the harness does *not* do

Parse instructions. Impose a cursor. Own control flow. Decide what to call or
when. Choose among well-typed actions. Everything interpretive is the model's
job.

### 4.3 Speed levers

Batching across the items of a call and across concurrent runs. Memoization by
content hash of (function, inputs). Prompts of a few hundred tokens per turn.
Crisp functions for everything exact.

## 5. Training data generation

Because the tree is the state and episodes are short, **one agent turn = one
training example**: `(messages so far, the turn's tools) → target turn`, in
exactly the message structure the ToolAgent uses, with cold-restart twins.

### 5.1 The program synthesizer and the reference policy

`natlang/gen/synth.py` composes programs from constructs: calls over lists,
plain calls, exact glue through `run_code`, filters, `if`/`else`, nested
pseudocode functions with their own code bases, aggregation into records
(next: folds, `repeat until` with check functions, multi-file code bases on
disk). Each program is built over a **latent world** whose hidden attributes
are the gold.

- **Gold comes from a Python twin** of the program, never from the text.
- **The reference policy compiles the tool calls from the program structure,
  never from the text.** It performs every call through the real harness.
  Three checks run on everything it emits: the harness accepts every call,
  the grammar of the call's own turn accepts the target, and the final value
  equals the twin's.
- **The text is a rendering of the structure** in several pseudocode dialects
  (Python-like, numbered steps, structured prose), with varied local and
  function names.
- **The teacher paraphrases with a round trip**: a paraphrase is kept only if
  the teacher itself, executing the paraphrased program through the harness,
  reaches the known answer on fresh instances (`scripts/paraphrase.py`).

**Tiers.** Leaf-only programs (kept at roughly a quarter); one function with
control flow; several functions; whole code bases modelled on the use cases
(triage pipeline, moderation, expense audit, legal-move checking, a shopkeeper
as a fold over events).

### 5.2 Leaves

Leaf functions with constructed gold (hidden attributes, label-first items,
schema-first records), leaves from existing datasets (a task definition is a
leaf body, its instances are the collection), and undetermined instances whose
gold is a blocker. Generative leaves get teacher drafts checked by crisp
constraints or a judge.

### 5.3 Recovery data

Perturb states with realistic errors and label the fix: a rejected call and
its hint, an input of the wrong type, a quiesced item in a large call (resume),
a callee's blocker (pass it upward), a reply before `return` is complete.

### 5.4 The teacher

Ternary Bonsai 2 27B, served locally. It is oracle, paraphraser, judge, and
content source, and it is **tested as an interpreter of stated structure**; it
is not a source of invented decompositions (a 27B answers whole and has no
reason to split work). Human-written code bases run by the teacher through the
identical harness give trajectories on messy real pseudocode, kept only when
the outcome is right and the structure lint passes. Details: `TRAINING.md`
§3.4.

### 5.5 Verification signals (for filtering and later RL)

Synthesized programs: agreement with the reference policy at every turn (a
free process reward) and the twin's final value. Human-written programs:
crisp checks, type-checked returns, judged checks.

### 5.6 Dataset registry

Families refer to the use cases in `TRAINING.md` §1 (A triage, B extraction,
C entity resolution and cleaning, D long-input aggregation, E rules and
policies, F human-written procedures, G decomposed QA, H summarization,
I judged refinement, J segment-wise transformation). Names are recorded from
memory: **verify license, availability, and version before use**, and track
each in a data manifest with its license.

**Existing datasets: natural programs with verifiable outcomes**

| Dataset | Family | Role | Verification |
|---------|--------|------|--------------|
| SPoC | F | human pseudocode programs | test cases |
| Django, CoNaLa | F | NL ↔ code lines, crisp-routing phrasing | code execution |
| NAPS, NL2Bash, tldr-pages | F | algorithm and shell descriptions | execution / exact match |
| BREAK (QDMR) | G | NL step programs with back-references | source QA gold answers |
| MuSiQue, StrategyQA, HotpotQA, 2WikiMultiHop | G | multi-hop with decompositions | gold answers |
| DROP | D | extract + count/sort/add | exact answers |
| FinQA, TAT-QA, TabFact, WikiTableQuestions | D | table + text, gold arithmetic programs | exact answers |
| GSM8K | – | arithmetic routing (small share) | exact answers |
| ProofWriter / RuleTaker, CLUTRR | E | forward chaining over NL rules | gold proofs by depth |
| bAbI, ProPara | F | state tracking | gold states |
| BIG-bench / BBH algorithmic | F | small exact procedures | exact answers |
| OOLONG, S-NIAH, RULER, BABILong | D | long-input aggregation; **held-out eval first** | exact answers |

**Existing datasets: leaves and collections with gold labels**

| Dataset | Family | Role |
|---------|--------|------|
| Super-NaturalInstructions | all | task definition = a leaf function's body; instances = the collection it is called over |
| FLAN collection, P3, Tülu 3 SFT mix | all | broad leaf coverage, replay |
| Tülu 3 RLVR / IFEval-style constraints | I | verifiable constraints with checkers |
| Banking77, CLINC150, HWU64, MASSIVE | A | intent routing; out-of-scope for abstention |
| Bitext support, Twitter support, Enron, GitHub issues, StackExchange | A | triage and routing |
| AG News, DBpedia-14, 20 Newsgroups, Yahoo Answers | A | topic |
| GoEmotions, TweetEval, SST-2, Yelp/Amazon reviews, SemEval ABSA | A | sentiment and aspects |
| Civil Comments / Jigsaw | A, E | policy-as-rubric moderation |
| CLEF eHealth TAR, SYNERGY | A | abstract screening against criteria |
| PubMed-RCT, LEDGAR, LexGLUE | A | domain classification |
| CoNLL-2003, OntoNotes, Few-NERD, WNUT-17, Pile-NER, DocRED, REBEL | B | entity and relation extraction |
| Schema-Guided Dialogue, MultiWOZ, ATIS, SNIPS | B | slot filling to typed records |
| SROIE, CORD, FUNSD, Kleister | B | document text → record |
| CaseReportBench, JSONSchemaBench | B | eval only |
| SQuAD 2.0, NewsQA, Qasper | B | span extraction with abstention |
| Magellan/DeepMatcher suite, WDC Products | C | entity matching |
| Hospital, Adult, Restaurant, Buy (data-wrangling suite) | C | error detection, imputation |
| SOTAB, Sherlock/VizNet, TURL | C | column-type annotation |
| Quora Question Pairs, PAWS, MRPC | C | text dedup |
| CUAD, ContractNLI, LegalBench subsets | E | clause × rule grids |
| XSum, CNN/DM, SAMSum, QMSum, MultiNews, GovReport, BookSum | H | summarization, hierarchical |
| ASSET, JFLEG, W&I, GYAFC, FLORES, WMT | J | segment-wise transformation |
| Prometheus Feedback Collection, HelpSteer2, UltraFeedback, SummEval | I | rubric judging |
| xLAM-60k / APIGen, ToolACE, Hermes FC, Glaive | – | tool-format retention |
| BFCL, τ²-bench, IFEval | – | external eval only |
| Spider, WikiSQL, BIRD | D | source programs for back-translation (Y3) |

**Datasets we synthesize** (summary in `TRAINING.md` §3.5; full designs, prior art, and build order in `SYNTHETIC_DATA.md`)

| ID | Dataset | Gold comes from | Serves |
|----|---------|-----------------|--------|
| Y1 | Latent-world corpora: hidden database rendered as messy text | the hidden database | A, B, C, D, G |
| Y2 | Synthesized code bases (`natlang/gen/synth.py`) | Python twin + reference policy | all interpreting skills |
| Y3 | Back-translated SQL / Python → NL procedures | executing the original | D, F |
| Y4 | Label-first rubrics and policies | construction | A, E |
| Y5 | Schema-first extraction, with corruption for repair | sampled value | B, repair |
| Y6 | Rule worlds in domain dress | exact solver | E, loops |
| Y7 | Simulated environments with mock side-effect tools | simulator state | F, K |
| Y8 | Interpreter drills, one per interpreting skill | reference policy | K1–K14 |
| Y9 | Scale sets: the same code bases over inputs far larger than an episode | per-item gold | the thesis comparison |
| Y10 | Dataset algebra: chained real datasets | composed gold | pipelines |
| Y11 | Minimal pairs and calibration sets | construction, vote splits | L1, L2, K11 |
| Y12 | Data-is-not-code robustness (embedded instructions) | unaffected behavior | safety, correctness |
| Y13 | Verifier-backed generation constraints | generated checkers | I |
| Y14 | Long-horizon stress sets | Y1/Y2 machinery | eval, some training |
| Y15 | Style corpus for the renderer (unlabeled) | – | surface diversity |

### 5.7 Hand-written code bases

Real programs keep the synthesizer honest, feed the teacher-as-interpreter
runs (§5.4), and are the demonstrations of the thesis. The first six:

| code base | shape it forces |
|---|---|
| **shopkeeper** | a long-lived fold over an open list of events; state in the accumulator; legal actions passed as inputs and checked by a crisp function |
| **legal-move checking** | exact rules in crisp functions next to fuzzy reading of a position described in prose |
| **moderation with escalation** | conditionals, blockers that propagate, a second opinion by copy-edit-call |
| **semantic highlighter for natlang** | input is a natlang code base itself: a call over files, then over lines/spans; every span gets a semantic role (signature, call, local, control flow, exact step, prose step, type); crisp functions assemble a format that tools can use (HTML with classes, or LSP semantic-token arrays). Data-is-not-code under maximum pressure: the input *is* instructions |
| **web server** | every HTTP request is an event handled by natlang: routing, reading state, deciding, and generating the page; effects (`http`, `store`) behind capabilities; a host adapter turns the open-list fold into a real listening server. Complete means: routing, static and generated pages, forms, sessions, errors, logging |
| **natural-language Prolog** | a knowledge base of facts and rules in prose; forward chaining to a fixed point with repeat-until: one round applies every rule to what is known (a leaf per rule), a crisp merge says whether anything was new, the loop stops when a round adds nothing; then the goal is compared with what is known. Open world: "unknown", never a guess. Implemented: `codebases/nlprolog/` |

Each lives on disk as `name.nl` + `name/`, has inputs with known answers where
the domain allows it, and becomes a conformance program.

## 6. Training

1. **Base model**: LFM2.5-350M for development. Shrink to 230M only after the
   pipeline is stable; the size difference will show mostly in leaf quality,
   not in interpreting.
2. **Format**: the model's native tool-calling format, history rendered by the
   official chat template, outputs grammar-constrained from day one so SFT and
   inference see the same distribution.
3. **Stage A, per-turn SFT**: reference-policy turns and leaf examples mixed.
   Target: interpreting skills in the high nineties per turn. This is the
   number the whole thesis rides on.
4. **Stage B, on-policy (DAgger)**: run the student, label the states it
   reaches with the reference policy (free), retrain.
5. **Stage C, RL**: GRPO or similar on programs with verifiable outcomes.
   Reward: final-state correctness, turn-level agreement with the reference
   policy where available, penalties for turns and oversized reads.
6. **A memory-constrained training mode is a goal**, so that people can
   fine-tune on their own machines: LoRA/QLoRA, 8-bit optimizer states,
   gradient checkpointing, short sequences (episodes are short by design), a
   documented configuration that fits in 8 GB. Real training runs on a larger
   GPU elsewhere; this machine proves out the whole system.

## 7. Evaluation

- **Per-skill accuracy** (`TRAINING.md` §2): next statement, call, call over a
  list, binding, glue, branch, resume, blocker, leaf, finish.
- **Whole-program success vs. number of calls** and vs. nesting depth of the
  code base; outcome score and structure lint reported separately.
- **The thesis comparison**: the same model answering a task as one prompt
  versus interpreting the code base, over input size; and against a frontier
  model on quality, latency and cost.
- Held-out program shapes, dialects, worlds; the conformance suite.
- Speed: turns per second batched, items per second for a call over a list.

## 8. Phases

**Phase 0, language spec.** Done: `spec/SPEC.md` v0.2-draft, conformance
suite, static checker.

**Phase 1, harness.** Done on the development machine: tree, types, code
bases, `call`, locals, tool surface, native constrained decoding, GPU serving
of the student and the teacher, grading, the conformance suite as code-base programs, complete swap-out.
Remaining: batching; SQLite store.

**Phase 2, baselines.** Untuned 350M and the teacher on the code-base
conformance programs; the one-shot-versus-interpreted sweep over input size.

**Phase 3, data pipeline.** Synthesizer across all constructs and tiers,
dialect renderers, paraphrase round trips at scale, leaf adapters for existing
datasets, recovery generator, drills. Target a few million turns with a
curriculum by structure.

**Phase 4, SFT** (on the larger GPU), including the memory-constrained recipe.
Iterate on the mix until interpreting skills are in the high nineties.

**Phase 5, on-policy and RL.** DAgger rounds, then RL. Review logged
finite-type write distributions and decide whether any deferred confidence
feature is worth building. Shrink to 230M if warranted.

**Phase 6, applications and speed.** Three real code bases measured against a
frontier model on quality, latency, and cost.

## 9. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Per-turn interpreting accuracy plateaus below ~98 % | Thesis fails. Measure early; keep statements to one or two operations; more recovery data; resume instead of redo. |
| The model does the items' work itself instead of calling | Structure lint in grading and in trace filtering; drills with small lists where answering by hand is tempting. |
| Leaves exceed the model's knowledge floor | Choose task families with small-judgment leaves; authors split leaves; escalation per leaf later. |
| Grammar constraints distort the model's outputs | Train under the same constraints; monitor P(call) and the mask-bind rate (`TYPES.md` §6.5). |
| Tool schemas grow with the state | Alternatives only for named slots and never inside a call in progress; path enums capped; measured in tests. |
| Rendering drifts between data generation and inference | One versioned policy; the generator and inference share `surface.py`; equality tested in CI. |
| Overfitting to synthetic pseudocode | Several dialects, verified paraphrases, human-written code bases, held-out shapes and dialects. |
| The base model is weak at code | Exact work is the author's crisp functions; `run_code` is one expression; a Python-shaped fallback to measure. |

## 10. Decisions, status, findings

### 10.1 Decided

- **Programs are pseudocode code bases; the author states structure; no
  anonymous lambdas; `call` is the only way to start a sub-task; copy, edit,
  call the copy to vary a function; `let` locals; the lambda holds every zone
  of state** (2026-09-19).
- Six fixed tools; no selective filtering of the tool list; the reply never
  carries the result; data only through the tool channel.
- Template-agnostic by default: the prompt is rendered by the loaded model's
  own official chat template; per-model adaptations (wrappers, tool aliases)
  are opt-in.
- `Map`, `Fold`, `Iterate` as node kinds; no tail calls; `Iterate` has a
  mandatory bound and crisp cycle detection; reactive systems are a fold over
  an open list; legal actions by construction.
- No recursion: a code base with a cycle is refused at load; run-time guards
  stand behind that (an edited copy could otherwise reintroduce one).
- Types in TypeScript syntax; structural typing only in v0.2 (postconditions
  and refinements deferred); write-time typing is the primary reliability
  mechanism.
- TypeScript for crisp functions and `run_code`; `run_code` returns to the
  interpreter and does not write; at-least-once effects with a journal.
- Research artifact; English first; teachers K2 Horizon 7B or Ternary Bonsai 2
  27B; authoring-time help from a larger model is out of scope; state size and
  shutdown are host concerns.
- This machine proves out the whole system, including data generation; real
  training runs elsewhere; a memory-constrained training mode is a goal.
- **Teacher paraphrases vary wording, dialect and names, never structure.**
  Rewriting structure is not forbidden in principle, but there is little to
  gain from instilling it through training data; structural variety comes
  from the synthesizer, where the twin still knows the answer.
- **Generative leaves get teacher-written reference outputs**, accepted by
  checks (crisp constraints plus a judge question) and stored with the latent
  world, so that programs that produce text are part of the corpus.
- **The first hand-written code bases** (`codebases/`, §5.7): six, chosen to
  differ in shape.

### 10.2 Status (2026-09-19)

Package `natlang/`: `types.py`, `values.py`, `nodes.py`, `refs.py`,
`paths.py` (tree and types); `codebase.py` (loader; refuses cycles);
`runtime.py` (sessions, `call`, combinators, guards); `surface.py`,
`tool_agent.py`, `native.py`, `decoder.py` (model-facing side); `js.py` +
`prelude.js` (sandbox); `checks.py` (grading); `host.py` (`load`, `load_fold`
for long-lived programs over an open list), `__main__.py`.

**Code bases** (`codebases/`, each with a scripted end-to-end test in
`tests/`): `nlprolog` (forward chaining by repeat-until), `highlighter`
(natlang reading natlang; HTML out), `webserver` (a fold over HTTP requests;
`scripts/serve_web.py` is the listening host adapter), `moderation` (early
returns; second opinion by copy-edit-call), `shopkeeper` (fold; legal actions
by construction), `legal_move` (exact rules, fuzzy reading, a blocker that
travels up); plus `examples/triage`.

**Corpus generation** (`natlang/gen/`, `scripts/generate.py`), all verified
turn by turn (harness accepts, the turn's grammar accepts, outcome matches):
- leaf families (`programs.py`): judge, classify, extract, crisp_scalar, with
  undetermined instances that end in a blocker;
- fixed shapes and the **compositional synthesizer** (`synth.py`): programs
  sampled move by move under kind and alignment constraints, two dialects,
  4 to 23 steps, nothing computed for nothing; exact steps use the executed
  code as oracle;
- the **hand-written code bases** (`codebases.py`): an input generator over a
  latent world and a reference script per pseudocode function. Leaves that
  generate text have template gold for now and are left out of the corpus
  until the teacher writes their references.
- As the model sees a sample: tools at most ~1.3k tokens, messages at most
  ~2.5k.

**Conformance**: 23 programs, all code-base style, each with a replayable
`reference:`; graded by `natlang/checks.py`.

**Measured with real models** (program 23, the triage code base): Bonsai 27B
follows every structural step (11 episodes, 638 s, 4 rejected calls corrected);
the untuned 350M writes one made-up record in one action and never calls. The
web server served real HTTP requests with Bonsai interpreting every step:
correct pages, a form submission persisted across requests, a 404; 4 to 6
minutes per request, which is the gap the tuned small model has to close.

**The whole loop, proven on this laptop (2026-09-19).** `export_sft.py` (pairs
rendered by the model's own template, exactly as at inference) → `train_lora.py`
(LoRA r=32 on a bf16 base, gradient checkpointing, one sequence at a time:
3.2 GiB of GPU memory, 6 s per optimizer step) → `to_gguf.sh` → `serve.sh` →
`eval_turns.py` / `baseline.py`. 150 steps (15 minutes, 2,656 pairs seen):
held-out loss 2.68 → 0.033.

| LFM2.5-350M, unseen programs (seed 777) | untuned | after 15 min of LoRA |
|---|---|---|
| next turn exactly right | 29% | **75%** |
| right tool | 43% | **88%** |
| `call` turns exactly right (shapes / composed / code bases) | 0 / 0 / 0% | 100 / 60 / 40% |
| conformance suite (21 programs; judge checks ungraded) | 2 correct; never calls | 6 correct + 3 ungraded; calls the named functions |
| time per program | ~1 s | 0.1 to 6 s (Bonsai 27B: 25 to 640 s) |

What the tuned model still gets wrong on the conformance suite: leaf
judgments themselves (program 06 is structurally perfect, one `call` over the
list, and wrong on the labels); folds and repeats (rare in the corpus: 53 and
75 of 1,800 calls); blockers; long programs (23: 19 rejected calls). These are
corpus-mix and training-length issues, not harness ones.

### 10.3 Findings worth keeping

*Serving and formats*
- The chat template embedded in LiquidAI's GGUF ignores `tool_calls`; serve
  with the official `chat_template.jinja`. History must be in the model's
  native call format. llama.cpp does not enforce argument schemas for LFM's
  format, hence our own grammar over the native call text. `n_probs` reports
  probabilities before the grammar mask, so P(the model starts a call) and the
  mask-bind rate are measurable over HTTP. Argument names must not be Python
  keywords. Prebuilt CUDA binaries need a newer glibc than this machine has:
  serve from Docker (`scripts/serve.sh`); ~26,000 tok/s prefill and 224 tok/s
  decode for the 350M on an RTX 4060.
- **Bonsai 27B on the 8 GB GPU**: Prism ML's llama.cpp fork in a CUDA 12.8
  runtime image, the lab's official template, q4_0 KV cache, 12k context:
  6.1 GB of VRAM, ~23 tok/s, 6–8 s per turn with a 512-token thinking budget
  (judge calls run with thinking off). **Host RAM**: `--no-mmap`,
  `--cache-ram 1024` and a 5 GB container cap bring it from ~6 GB to under
  1 GB; `scripts/watch_bonsai.sh` restarts it on a failed health check.
- Bonsai's XML call format delivers parameters as text and sometimes wraps a
  value as `{"value": X}`; the harness parses and unwraps. **Its server cannot
  emit a tool literally named `call`**, so decoders take per-model tool
  aliases (`call=call_function`).

*The untuned 350M* (small samples): strong at a complete typed answer in one
shot, weak field by field; parrots nearby prompt text (hence no value-like
placeholders and a short prompt); tool-set size dominates its choice of tool;
picks the cheapest well-typed value (`[]`) for a list task; invents optional
fields; mean P(call) at the start of a turn ~0.35–0.5. Under native decoding
every call is valid and a run takes well under a second.

*Bugs the runs found*
- **Tool-schema growth**: sub-task alternatives offered for every element of
  a filled list (6k → 21k characters), and the internals of a call in progress
  exposed as writable slots (→ 45k), pushed requests past the teacher's
  context. Alternatives are now offered for named slots only and never inside
  a pending node.
- **The workspace listing showed only the first line of a multi-line text, in
  quotes, as if complete.** The teacher decided conformance program 16 from a
  truncated rubric. Short texts are now shown whole and cut-offs say so.
- A long text input was probed as a file name; a generator test depended on
  wording that verified paraphrases change.

*Grading and the teacher*
- With checks-based grading (`natlang/checks.py`: value, status, crisp and
  judge checks) Bonsai reaches **19 of 20** on the leaf-style conformance
  programs with no rejected calls; the miss was the schema-growth crash. It
  solved 19 of 20 in a single episode: a 27B does not decompose on its own,
  which is one of the observations behind stating structure in the program.
- **Bonsai follows the triage pseudocode**: `call classify over tickets`,
  `run_code` for the `not_spam` flags, `select_by_flags`, `call is_urgent over
  real`, with rejected calls corrected from their hints. A worker correctly
  reported a blocker for a ticket the rubric did not cover, and the root passed
  it upward.

### 10.4 Known limitations of the current code

- Callees run sequentially; no batching. Copies are deep copies; the tree is
  in memory.
- `run_code` and crisp code are not statically checked; the QuickJS binding
  cannot call into Python while a time limit is set, so effectful code runs
  without the time limit (a subprocess worker should replace this).
- Conformance programs 03–22 predate code bases and are graded on outcome
  only.

### 10.5 To be measured, not argued

- Per-turn accuracy of the base and tuned model on each skill, and
  whole-program success against the number of calls. Go/no-go for the thesis.
- One-shot versus interpreted, over input size, for the same model.
- Which pseudocode dialects the model reads best; paraphrases per program.
- Mask-bind rate and any quality loss from grammar constraints.
- Tool-set sweep (is `edit` worth its place for the untuned model?).
- Keep or drop the caller's context while blocked in a call.
- Base versus instruction-tuned checkpoint; 350M versus 230M.
- Engine fit for LFM2's hybrid architecture: per-request grammars, logprobs,
  prefix caching across the turns of an episode.

### 10.6 Deferred

- Voting, confidence thresholds, calibration, escalation, speculative
  execution (pending logged finite-type write distributions).
- Postconditions (`ensures`) and type refinements; narrowing the result type
  of a call.
- Provenance idioms: `rerun`, staleness, dataflow; memoization.
- On-device versus server; budget (default: one GPU and time, no API spend).
- Security model for code with real file and network access.
