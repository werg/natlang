# The natlang compiler: functions into weights, experience into weights

Status: design brief, 2026-11-25. Owner direction (verbatim, 2026-11-25):

> "Since we seem to be struggling a lot with getting the model to measurably improve on just vanilla general
> capability, I think we should branch out somewhat."
>
> "Rethink, rework, and update our approach to per function, per code base tuning. What are all the technique
> options that we have there? It would be really, really, really cool if we were able to come up with a meta-learner
> that could take a natlang program and perhaps some samples and optimize its parts sample efficiently. Perhaps even
> something that plays with the fact that we have a programming language here. Perhaps something in the direction of
> playing with the futamura projection or in general just soft prompts, weight space programming and adapters."
>
> "Leveraging the architecture and perhaps some trained state of ~/hindsight-kb to train general capability and allow
> for per task / task group knowledge base tunes."
>
> "We want to be unreasonably ambitious. And we want to really lean on priors and think closely about what we can
> prioritize based on our first principles thinking."

"Write down this design brief as markdown doc. Be unreasonably ambitious."

This brief is the build plan. It is not a proposal list: every part below is to be built, in dependency order, and
each part enters use as soon as it works (AGENTS.md: "A finding is not done until it is acted on"; "Don't measure
what first principles already settle").

---

## 1. Thesis

**A natlang program is a program, and a 1.7B model is an interpreter for it.** Everyone else treats a small model as
a general-purpose assistant and tries to make it better at everything. That is the hardest problem in the field, and
it is the one our held-v3 loop has been grinding against (every trained policy about +4 over base, nothing after round
1 measurably added; DECISIONS 2026-11-25 14:20).

Natlang gives us something almost no one else has: **the work arrives pre-decomposed into typed call sites**. A call
site is a fixed instruction, a signature, types, documentation, a position in a call graph, and a stream of inputs.
Making a model excellent *at one call site* is a low-dimensional specialization problem: a vastly easier problem than
general capability, and exactly where small models close the gap to large ones.

So we stop asking the interpreter to be good at everything and start **compiling programs into the interpreter**:

1. **The compiler** (`F → W`): a learned network that turns a natlang function into model weights in one forward
   pass. The second Futamura projection, made real: a specializer of the interpreter, itself a learned program.
2. **Experience compilation** (`E → W`): the same network fed with retrieved verified experience instead of program
   text: hindsight-kb's adapter reader, unified with the compiler. Per-task, per-family, per-user knowledge bases
   become weight-space scopes.
3. **A JIT runtime**: cold call sites interpret, warm sites run compiled weights, hot sites get gradient-tuned
   weights, and synthesized NL literals are compiled the moment they are created. **Runtime function synthesis gets a
   compiled form.** No other system can do this, because no other system has semantic functions as first-class values.
4. **A data engine that does not need task oracles**: call-level distillation from strong teachers across millions of
   NL call sites, which feeds the compiler, the judges and the base model's own SFT at once.

The end state: a user writes a natlang program; the first call already runs on weights compiled for it; every
further call makes it better; a 1.7B model runs that program at a level a frontier model reaches only by reading the
whole program every time.

---

## 2. The Futamura frame, concretely

| Futamura | Classic | natlang |
|---|---|---|
| Interpreter `I(p, x)` | runs program `p` on input `x` | the model `M` running call site `p` (instruction, signature, types) on input `x` |
| 1st projection `S(I, p)` | specializing `I` to `p` gives a compiled program | **per-site adapter** `W_p` such that `M + W_p` runs `p` well, even without reading `p` carefully |
| 2nd projection `S(S, I)` | specializing the specializer to `I` gives a compiler | **the compiler network** `C(p) → W_p`: one forward pass, no optimization |
| 3rd projection `S(S, S)` | a compiler generator | **the compiler that improves the compiler**: `C` trained on its own compiled outputs' outcomes, across models (§10) |
| Static argument specialization | partial evaluation on known inputs | **`.partial(x)`, `.in(...)`, `withContext`** compile to `W_{p,x}`: the bound argument folded into weights |
| Inlining / deforestation | fusing producer and consumer | **fusion** (`ts-host/src/fusion`): adjacent NL calls fused into one call with one compiled adapter |
| JIT tiers | interpreter, baseline JIT, optimizing JIT | instruction-in-context, compiled adapter, gradient-tuned adapter (§6) |
| Dynamic code generation | `eval`, closures | **NL literals built from data** (`nl\`...\`` with captures): compiled on creation |

Today we already have the first projection, done slowly: gradient-tuned per-site LoRA (`natlang tune`). This brief
builds the second projection and puts it in the runtime.

---

## 3. Inventory: what exists and what each part becomes

Every piece below is already in the repository. The brief extends them; it does not start parallel stacks (AGENTS.md
"One implementation per concept").

| Existing | Where | Becomes |
|---|---|---|
| Per-site LoRA scope tree, per-call stacks, release store, activation/rollback | `ts-host/src/adaptation/weights/` | the weight store for compiled and tuned adapters, keyed by site identity |
| `natlang tune` (collect, split, train, export, validate, activate), incremental builds, families, distill | `ts-host/src/tune/`, `training/neuralese/natlang_neuralese/tune/` | **tier 2** (hot sites); its fits are also teacher targets for the compiler (§5.3) |
| Persistent site identities, rename migrations | `ts-host/src/tune/sites.ts` | compiled-adapter cache keys |
| Soft instruction and skill states (text-initialized Neuralese) | `tune/soft.py`, `tune/soft_export.py`, llama fork `--native-state` | the key-value half of compiled output; the free "tier 0.5" compiler (text-init = the model reading its own instruction) |
| Learned compaction operators | `ts-host/src/native/agent.ts` (`compaction`), `operator-frames.ts`, `model/learned_operators.py` | compression of retrieved experience and long site context before it is compiled (§7) |
| Memetic guidance | `rl/memetic_program.py`, `ts-host/src/guidance/` | the text-level competitor that compiled experience subsumes; its development/confirmation protocol is reused for per-scope knowledge bases |
| GEPA instruction optimization | `ts-host/src/gepa/` | the text-level optimizer at tier 2 (instruction rewriting stays available alongside weights) |
| Fusion planner | `ts-host/src/fusion/` | the call-graph partial evaluator; fused sites are compiled as one site |
| Per-sequence LoRA serving: llama fork `--lora-per-sequence`, wllama per-request LoRA, SGLang multi-LoRA, Python native | `~/llama.cpp-neuralese`, `ts-host/vendor/wllama`, `serve/sglang_actor.py` | serving of compiled adapters on every engine, unchanged protocol (AGENTS.md "One serving feature, every engine") |
| LoRA bank and program-tuning lanes | `/home/werg/data/program-tune-20261117/`, DECISIONS 11-24 | the bank seeds the compiler's training set (site → fitted adapter pairs) |
| hindsight-kb: KB storage, IVF index, consolidation, relations, adapter reader, KV reader, experience objectives | `~/hindsight-kb/src/hindsight_kb/{kb,models/adapters.py,experience}` | the experience half (§7): the store, the index and the reader architecture, ported to LFM3 and merged with the compiler head |
| Expert-iteration verified-trajectory pool | `scripts/neuralese_data/expert_iteration.py`, `ei-*` registry roles | the first experience records |
| Authored programs (28 and growing), broad-external rows, pool tasks | `scripts/rl_tasks/sources/authored_programs.py`, registry | call-site corpus for the data engine |

**Open defect that gates tier 2 (not the compiler):** program-tuning merged arms measured −23 to −78 points against
base on four of six programs (2026-11-25 11:00 log, p 0.0005). A hunt is running. Gradient-tuned adapters are both
tier 2 and a teacher target for the compiler, so its fix lands in both.

---

## 4. The compiler `C`

### 4.1 Input: the compilation unit

A call site's **header**, rendered as the model sees it in a real call (AGENTS.md "Text-init in context", "Learned
operators start from text"):

- the instruction text (after captures are resolved, for NL literals);
- the signature and the types, including doc comments, literal unions, refinements (`Is<>`, `ensures`);
- the call-graph context: the caller's instruction path (root → … → this site), and the site's role in a combinator
  (the `match:` of a `semanticMatch`, the `check` of a `semanticRepair`, the step of an `iterateOn`);
- optionally: k example (input, output) pairs; static arguments bound by `.partial`/`.in`/`withContext`;
- optionally: retrieved experience records (§7).

### 4.2 Architecture

- **Reader: the frozen policy itself.** LFM3 reads the header in the real rendered call. Its hidden states are the
  representation of the function. This is the text-init principle, so the compiler's zero point is "the model's own
  reading of the instruction", never a random adapter.
- **Head: a learned weight writer.** Per layer, a small set of learned queries cross-attends to the reader's hidden
  states and emits:
  - **LoRA factors** (rank 8-16) on all 9 LFM3 projection families. LoRA, not only a key-value prefix: LFM3's
    convolution blocks carry much of the computation and a prefix never reaches them.
  - **A primed key-value state** for the attention layers (the Neuralese family: one mechanism with soft
    instructions, skills and memetic states; owner 11-20).
- **Additive.** The instruction stays in context; the compiled weights correct on top of it. Nothing the model would
  read is withheld, so the worst case is the uncompiled behaviour. (A "header-free" mode for throughput, where the
  compiled weights replace the instruction, is a later serving option, not the default.)
- **Shared with hindsight-kb.** The head is the same object as hindsight-kb's adapter reader `D_A` and KV reader
  `D_KV`. One weight writer, two input kinds (program text, retrieved experience).

### 4.3 Objectives (all on at once; engineering, not science)

1. **Context distillation.** Student: the call with the compiled weights. Teacher: the best available answer for the
   same call: a frontier model answering it (§5), or our own model with many demonstrations and guidance in context,
   or a gradient-tuned tier-2 adapter for that site. Loss on the student's output tokens only (prompts and inputs are
   zero-weight context, AGENTS.md).
2. **Downstream reward.** Where the call sits in a task with an oracle (pool tasks, authored programs with exact
   checks), the episode outcome trains the compiler through the compiled weights. The GRPO learner already trains
   per-call sequences; compiled weights are one more parameterization of the same call.
3. **Adapter regression.** Where a tier-2 fit exists for a site, `C(p)` is also pulled toward the fitted adapter's
   function (output distribution on the site's rows, not raw factor matching: factor space has symmetries).
4. **Few-shot refinement (the meta-learner).** Train `C` so that its output is the best *starting point* for k
   gradient steps on n samples (Reptile-style outer loop). Compiled weights are good alone and become excellent with a
   handful of samples. This is the sample-efficient optimizer of program parts the owner asked for.

### 4.4 Composition rules (weight-space programming)

- **Static arguments:** `.partial(x)` → compile `(p, x)`; the bound argument is part of the header. Partial evaluation.
- **Combinator roles:** the compiled weights of a `match:` judge, a `check`, an `iterateOn` step know their role.
- **Fusion:** a fused site is one compilation unit whose header is the fused call's header.
- **Adapter algebra:** scoped stacks (program base + family + site) already exist in `adaptation/weights`; compiled
  site adapters stack on learned program and family bases exactly like tuned ones.
- **Synthesized functions:** an NL literal built from data at runtime is compiled when created, keyed by its resolved
  text. Functions returned by functions (`scopeJudge`, `noteWriter` in the authored programs) arrive with weights.

---

## 5. The data engine: call-level distillation at scale

**The supply finding** (DECISIONS 2026-11-25 14:20): round 1 trained on 19.6M new teacher-derived tokens; recent rounds
got 1-2.5M. Whole-episode supervision needs task oracles and long rollouts: that is the bottleneck.

**A single NL call needs neither.** It is short, cheap for a teacher, and checkable without an oracle: types parse,
refinements hold, independent teacher samples agree, the downstream caller consumes it.

### 5.1 Sources of call sites

- every NL function and every NL literal in the authored programs (28 now; the campaign keeps producing);
- every model call in collected episodes (own and teacher) across the pool: the site header is in the rendered
  prompt;
- broad-external rows (68,887 eligible) restated as typed NL functions;
- the program-inventor idea bank and the semantic-FP catalogue as generators of *new* sites (owner rule: no
  procedural generation of task worlds; sites written by authors, inputs from real data);
- synthesized literals: run programs and record every literal they create.

### 5.2 Teachers

Frontier models answer each call in the real rendered form (Anthropic and OpenAI credits, free models; teacher choice
by measured agreement, `ts-host/src/tune/teachers.ts`). Teacher reasoning stays zero-weight context; the answer is
the target (owner 11-24).

### 5.3 What the corpus feeds, simultaneously

1. the compiler's context-distillation targets (§4.3.1);
2. the base model's SFT mix (a new tier in `sft_sources`: call-level teacher answers, weighted and deduplicated like
   the others), directly addressing the supply gap;
3. the judges (§8);
4. tier-2 fits on hot sites.

**Scale target: tens of millions of call-level target tokens per week**, limited by teacher spend, which is the
owner's decision. This is the one place where more teacher budget converts directly into progress.

---

## 6. The JIT runtime

The tier choice is policy, so it is a natlang program with a crisp twin behind one interface
(`ts-host/src/runtime/pluggable.ts`; AGENTS.md "Policy is natural language; mechanism is crisp").

| Tier | When | Weights | Cost |
|---|---|---|---|
| 0 interpret | first sight of a site, or the compiler unavailable | none (instruction in context, as today) | 0 |
| 0.5 primed | always available | text-initialized key-value state of the instruction | one prefill |
| 1 compiled | warm site, or any synthesized literal | `C(header)` | one forward pass of `C`, cached by site hash |
| 1.5 refined | a site with a few successful calls | `C(header)` + k gradient steps on its own verified calls | seconds |
| 2 tuned | hot site with a stream of calls | `natlang tune` fit, warm-started from tier 1.5 | today's program tuning |

- **Promotion signals:** call counts, refinement-failure rates (`holds` false), `decide` confidence, user feedback,
  and repair rates. The tracing JIT already records hot paths.
- **Caching:** compiled weights keyed by (site identity, header hash, compiler version, policy sha). Code identity
  is provenance (AGENTS.md).
- **Serving:** per-sequence LoRA on every engine already exists. The compiler runs beside the actor on the same GPU
  (it *is* the actor's forward plus a small head).
- **Every engine:** managed llama-server, wllama in-page, Python native, SGLang. A compiled adapter is an ordinary
  per-sequence adapter; the compiler itself runs where the policy runs (in the browser too: the head is small).

---

## 7. Experience compilation: hindsight-kb, unified

### 7.1 What it becomes

The record store holds **verified experience**: successful trajectories, successful calls, repairs that worked,
failures with their corrections. At each call site, the runtime retrieves relevant records and the shared weight
writer turns them into weights (LoRA delta plus key-value state) for that call. This is hindsight-kb's design, with
three decisions:

1. **Records come from what we already verify.** The expert-iteration pool, authored trajectories, call-level teacher
   answers that passed checks, and the user's own successful runs.
2. **Scopes are knowledge bases.** A record store per program, per family, per user, per project. A "per task-group
   knowledge-base tune" is a scope; adding experience to it needs no training run.
3. **One writer.** hindsight-kb's `D_A` and `D_KV` readers merge with the compiler head (§4.2). Program text and
   experience are two inputs to one network.

### 7.2 Build order (each stage enters use when it works; no measurement gates between them)

1. **Retrieval as demonstrations.** Retrieve similar verified calls and trajectories and put them in context as
   examples. Uses the existing index code (`hindsight_kb/kb/ivf.py`, `retrieval.py`) with the policy's own hidden
   states as keys. Strong prior: retrieved demonstrations reliably help agents.
2. **Retrieval as compacted Neuralese.** The retrieved records pass through the learned compaction operators (now
   being turned on) so many records fit in little context.
3. **Retrieval as weights.** The shared writer turns retrieved records into weights; the records leave the context
   entirely.
4. **Learned addressing.** hindsight-kb's hindsight writer and query encoder, trained on our trajectories. This is
   its six-stage curriculum, run last, on a system that already works.

### 7.3 What we keep from hindsight-kb and what we skip

- **Keep:** storage (`kb/sqlite.py`, `packed_store.py`), IVF and resident index, consolidation and relations,
  provenance, the adapter and KV reader architectures (`models/adapters.py`), the experience objectives
  (`experience/objectives.py`).
- **Port:** the readers to LFM3 (hybrid convolution plus attention; the Qwen3.5 port already handles recurrent span
  composition).
- **Defer:** the hindsight writer's curriculum, until stage 4.

### 7.4 General capability through experience

This is the route to the owner's general-capability goal that does not depend on squeezing everything into 1.7B
weights. The base model learns to *use* compiled experience; the experience itself lives in the store and grows
without retraining. Every verified success anywhere becomes available to every similar call everywhere.

---

## 8. Judges as first-class models

`holds`, `beats`, `settled`, `relates`, `meets`, and `decide` with its confidence carry every natlang program's
control flow: repair loops, fixed points, matches, rankings and best-of-n.

- **Train them hard and specifically.** Call-level distillation (§5) gives millions of judge calls with teacher
  verdicts, and agreement between teacher samples gives calibrated confidence.
- **Compile them.** Every judge site gets compiled weights for its criterion (a `match:` judge compiled from its
  rule, an `ensures` predicate compiled from its predicate).
- **Calibrated confidence becomes real.** `decide(...).confidence` becomes meaningful, which every authored-program
  builder hit as a gap (wave 2 notes).
- **Test-time multiplier.** A strong judge turns sampling budget into accuracy: best-of-n, repair, verification. This
  is capability that grows with compute at inference, independent of base-model gains.

---

## 9. Decomposition (natlang-native)

Per AGENTS.md, every policy part is a natlang program; crisp code is mechanism.

| Unit | Natural-language policy | Crisp mechanism |
|---|---|---|
| Tier choice per call | `chooseTier`: given site stats, refinement failures and confidence, pick 0-2 (pluggable; crisp twin) | counters, cache lookup |
| Promotion to tuning | `worthTuning`: is this site hot and failing enough to fit | `natlang tune` pipeline |
| Site header rendering | none (exact rendering of the real call) | renderer shared with the runtime |
| Retrieval scope selection | `whichScopes`: which knowledge bases apply to this call | index lookup |
| Record admission | `worthRemembering`: is this run's experience a record, and in which scope | store transaction |
| Teacher routing for the data engine | `pickTeacher` by measured agreement per site kind | API clients |
| Call consistency check | `teachersAgree` (semantic agreement of teacher samples) | type parsing, refinements |
| Compiler, writer, readers | none | the networks and their training |

---

## 10. The ambitious end states

1. **Compile-on-write.** Saving a natlang file compiles every function in it. The program runs at tier 1 on its first
   call.
2. **Self-compiling programs.** A program improves itself: its own verified runs enter its scope's store, its hot
   sites promote, and its compiled weights refine continuously in the background. Memetic learning becomes
   weight-space learning.
3. **Compiled synthesis.** A program that synthesizes functions from data (NL literals) runs those functions on
   compiled weights. Semantic functional programming with real closures in weight space.
4. **The third projection.** The compiler trains on its own outcomes: compiled sites that fail produce gradient-tuned
   corrections, and those become the compiler's training targets. The compiler that improves the compiler.
5. **Cross-model compilation.** One compiler head per backbone, trained on shared call-site corpora; a new backbone
   (LFM3 successors) inherits the whole compiled program library by re-running the head's training, not every fit.
6. **Browser-native.** The compiler head is small; wllama runs it in the page. Users get compiled programs locally.
7. **Natlang improving natlang.** The runtime's own decisions (tier choice, fusion planning, recovery) are natlang
   programs, so they are compiled too: the runtime runs faster and better on its own compiled policies.

---

## 11. Work streams (start in parallel)

1. **Data engine** (§5): site extraction from authored programs and collected episodes; teacher answering at scale;
   consistency checks; registry entries; the new `sft_sources` tier for the base model. *Starts now.* Teacher budget is
   the owner's call.
2. **Compiler v1** (§4): reader = frozen LFM3, LoRA plus key-value head, context-distillation objective, few-shot
   refinement outer loop. Trainer on the shared LoRA+rows core (`train/lfm3_learner.py`, `lfm3_fast.py`). The
   tune bank's (site, adapter) pairs seed it once the program-tuning regression is fixed.
3. **Runtime tiers** (§6): `chooseTier` pluggable, compiled-adapter cache keyed by site identity, per-sequence serving on
   every engine; NL literals compiled on creation.
4. **Experience store** (§7): port hindsight-kb's store and index into the natlang training tree; retrieval as
   demonstrations (stage 1), then as compacted Neuralese (stage 2), then the shared writer (stage 3).
5. **Judges** (§8): judge-site corpus from the data engine, calibrated confidence, compiled judge sites.
6. **Tier-2 repair**: the program-tuning regression hunt (running), then tier 2 as warm-start from tier 1.5.

The held-v3 expert-iteration loop keeps running in the background with its fixes (neutral rounds continue, supply
work); it stops absorbing coordination attention. Its pool is the experience store's first content.

---

## 12. Risks, named so they are designed around, not used as gates

| Risk | Design answer |
|---|---|
| Hypernetwork generalization to unseen sites is modest in published text-to-LoRA work | additive weights (worst case = uncompiled), few-shot refinement (tier 1.5), compiler trained on millions of sites rather than hundreds |
| Factor-space symmetries make adapter regression ill-posed | regress on output behaviour, never on raw factors |
| Retrieval leaks held data | records are admitted through `held_eval.exclusion_ids()`; scopes never include held ids or siblings |
| Teacher answers are wrong | teacher-sample agreement, type and refinement checks, downstream consumption; volume over per-sample polish (owner 11-25) |
| Compiled weights drift with policy updates | cache keyed by policy sha; recompile is one forward pass |
| One more serving path | none: compiled adapters are ordinary per-sequence adapters on every engine |
