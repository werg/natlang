# Big real-world examples: compilers, a natural-language database, a pi harness

Owner request (2026-10-07): several optimizing compilers for existing languages, a natural-language SQL database
(with neuralese storage and indexing as an extension), and a port of the pi agentic harness. In all of them, keep
crisp code to the minimum and let natural-language functions carry the program; build real structure from
composable subtasks; update the authoring skills with what we learn; improve primitives when something is missing.

## Shared ground rules

- **What stays crisp.** Host shells (CLI, server), services that stand for the outside world (assembler, linker,
  process runner, IR verifier, storage commit, model endpoints), and the verification harnesses. Everything that is
  the program — parsing, analysis, optimization, code generation, query planning, transaction semantics, agent
  judgment — is natural language. The interpreter may still write TypeScript in its eval scope at run time; that
  is the model's work, not ours.
- **Every stage has a checkable contract**, and mechanical feedback (verifier and test output) goes back into a
  monitored repair step (`iterateOn`) whose steps are recorded. A stage whose output fails its check is not kept.
- **Development model.** The DGX teacher (Qwen3.6-35B-A3B, profile `dgx-qwen`, :8082). It is serving the s73
  collection, so one call takes minutes; pipelines run independent calls concurrently. Wiring tests use scripted
  models so CI stays model-free.
- **Memory.** While the Maple warm-up trains, every job goes through the ledger at ≤1.5 GB (peer agreement);
  natlang runs are HTTP clients of the teacher and fit.

## 1. Optimizing compilers (`applications/compilers`)

The structure of a modern compiler (LLVM/GCC), each box a natural-language function or a small family of them.
Front ends for several languages share one middle end and back end.

```
source ──► front end (per language)          ──► middle end (shared)              ──► back end (per target)
           split   top-level declarations         plan     pass pipeline per function    select    instruction selection
           declare symbols, types, layouts        mem2reg  SSA construction                         (virtual registers)
           check   diagnostics                    simplify instcombine + SCCP + folding  allocate  register allocation
           lower   per function → IR              gvn      redundancy elimination         frame     ABI, prologue, stack
           runtime language runtime as IR         licm     loop-invariant motion          peephole  machine peepholes
                                                  loops    unroll, strength reduction     emit      assembly text
                                                  inline   interprocedural inlining
                                                  dce      dead code, CFG cleanup
```

- **IR: textual LLVM IR** (opaque pointers). Models know it deeply, and it is the state-of-the-art middle-end IR.
  It is checked by real LLVM: `llvmlite` (LLVM 22, pip-installed) parses and verifies modules and JIT-runs them. A
  `toolchain` service (crisp, Python + gcc) offers `verify(ir)`, `run_ir(ir, stdin)`, `assemble_and_run(asm,
  stdin)`, and timing. The service checks; it never generates code.
- **Translation validation by testing.** After every pass, the module must verify and produce the same output as
  before on the program's test inputs (whole-program differential execution via the JIT). A failing pass gets one
  repair round with the verifier's message; if still wrong, the function keeps its previous IR and the rejection is
  recorded. The back end is validated the same way against the IR's output.
- **Pass manager in natural language.** `plan` reads a function and decides which passes to run in which order
  (like learned pass ordering); the host applies them, concurrently across functions.
- **Front ends.** C (a C99 subset: integers, floating point, pointers, arrays, structs, loops, recursion, libc
  printf/malloc) and Python (a typed subset in the style of Codon/mypyc: annotated functions over int, float, bool,
  str, list). The Python front end emits its runtime (lists, strings, printing) as IR into the module, so it is
  optimized with the program. Lua (dynamic typing, tables) is the third target if time allows.
- **Back end.** AArch64 (this machine; gcc assembles and links). x86-64 is the same back end with another target
  description, for Pop.
- **Benchmarks.** Programs with stdin and expected stdout (fib, sieve, matrix multiply, n-body, quicksort, string
  building, struct-heavy code). Report correctness, then run time against gcc -O0/-O2 (C) and CPython (Python),
  and which passes were accepted or rejected per function.

## 2. A natural-language database (`applications/nldb`)

Requests are plain language: "keep track of our customers: name, city, when they signed up", "Ana from Lisbon
signed up today", "move 50 from Ana's account to Ben's, all or nothing", "which cities have more than two
customers?". The engine interprets them as a database would: schema, constraints, transactions, plans, indexes.

**File format (the stored data structures), readable from natlang and by people:**

```
shop.nldb/
  FORMAT.md                    the format itself, so any reader (model or person) knows the layout
  catalog.json                 tables: description, columns {name, type, description, nullable, unique,
                               references}, primary key, indexes, row count, next row id, schema version
  tables/<table>/<page>.jsonl  rows {"_id": n, ...}; at most 256 rows per page; pages ordered by _id
  indexes/<table>/<column>.json sorted [key, _id] pairs (paged when large)
  log/<sequence>.json          one record per committed transaction: the request, the statements it was read
                               as, and every row change (before/after) — redo and audit
```

- **Pure version (end to end in natural language).** `define` (schema from a description), `transact` (a
  `directory-reducer` over the database folder: a transaction is one reducer call, so its file changes are kept
  only if it finishes — atomicity by construction; a constraint violation fails the call and nothing is kept),
  `query` (read-only; returns columns, rows and how the answer was obtained), with helpers for planning, constraint
  checks and index maintenance. The host serializes transactions per database (an `EventLoop`) and gives reads the
  last committed state. No crisp code beyond the shell.
- **Optimized version.** Same natural-language interface over SQLite (`node:sqlite`): `translate` compiles a
  request to SQL against the schema (checked with `EXPLAIN` and against the request's classification: a question
  may not write); SQLite executes exactly; one request is one SQLite transaction. Semantic conditions ("complaints
  that mention a late delivery") become a user-defined function backed by a natural-language call, memoized by
  argument.
- **Neuralese storage and indexing (extension).** A column of type `Neuralese<T>` holds content-addressed block IDs
  (`nz1_…`); payloads live in `blocks/<dialect>/<id>.safetensors` (pure) or a blob table (optimized). An index is
  per dialect: a pooled, normalized vector per block in an IVF index (coarse centroids, inverted lists). Queries:
  nearest blocks to a given block or text (encoded by the serving model), and semantic filters that read blocks as
  `Neuralese` arguments. Integrity: IDs are content hashes; a dialect is pinned per column; a backbone change
  requalifies the channel and rebuilds the index (the foundation rules in AGENTS.md).

## 3. pi on natlang (`applications/pi`)

pi (earendil-works/pi, MIT): a minimal harness — four tools (read, write, edit, bash), a short system prompt,
AGENTS.md context files, on-demand skills, JSONL session trees, compaction, steering and follow-up queues, and
**codemode**: the model writes a script that calls tools and non-LLM classifiers (Jev) so only the script's output
enters its context.

**Port.** The outer loop is pi's: the big model with pi's tools through `pi-ai` (already our dependency), sessions
as JSONL trees, AGENTS.md discovery, compaction. Codemode is natlang eval: the big model writes TypeScript in which
inline `` nl`…` `` calls are typed judgments run by a small model. The big model becomes a natlang author at run
time and the small model its interpreter.

**System One: where natlang helps the big model** (own analysis first, then checked against how people use Jev and
other decision models in harnesses — LangChain's Jev harness, "Jev engineering", OpenHarness's decision gate,
REFLEX). Each is a natural-language function; finite ones use `readout: decision` (one scoring pass, a probability
per allowed value); code acts on the probabilities with per-action floors, escalating below them:

| Extension point (pi event) | Decision / helper | Type |
|---|---|---|
| `tool_call` (bash) | risk of a command: safe / ask / deny | decision, floors 0.35/0.7 |
| before each turn | route: does this step need the big model? | decision, low confidence goes up a tier |
| `tool_result` | digest long output against the current goal; full text kept | small-model summary |
| task start | scout: which files matter for the request | small-model ranking |
| after `edit` | does the diff do what was intended, nothing else | decision + reason |
| each turn | progress: progressing / repeating / blocked | decision → steering note |
| `agent_end` | done: is the request actually complete | decision → continue or finish |
| compaction | structured summary (goal, decisions, files, next steps) | small-model summary |

Evaluation: small coding tasks in scratch repositories (fix a failing test, add a feature, refactor), with and
without the System One layer: success, big-model tokens and turns, blocked risky commands, false asks.

## Status (2026-10-07)

Built and pushed; wiring tested with scripted models; live runs on the development model wait for the teacher
window (the s73 collection holds the teacher until about 2026-10-08 10:00).

- **Compilers.** All stages in the table above except `split`/`check` (folded into `declare`) and `frame`/`emit`
  (folded into `select`). Two drivers over the same stages: the checked host driver (`index.ts`) and the pure
  pipeline (`compiler.nl`, the pass manager in natural language; the host only checks its final program). Lua and
  x86-64 not started. Benchmarks: C fib, sieve, matmul, quicksort, points; Python fib, sieve, collatz.
- **nldb.** Pure folder engine and SQLite engine as designed; redo log and recovery; questions cannot write;
  semantic conditions judged once per value. Neuralese: content-addressed safetensors blocks, per-dialect IVF index
  (tested with synthetic payloads); reading blocks in conditions waits for a neuralese-capable server.
- **pi.** pi's loop, tools, prompt, sessions and compaction; every System One row above, routing opt-in; codemode
  through natlang eval. Six evaluation tasks with hidden checks; variants plain, system-one, codemode, route.
- **Primitives.** `runtime.decide` (decision distributions to the host); `types.ts` inherited by nested named
  functions, doc comments on type fields reach the opening; neuralese store exports in both entry points. "More than
  one model per runtime" was not needed: the pi host keeps the big model as a plain driver, and drivers compose
  (limit, route, answer a known call).
- **Skills.** patterns.md: checked stages, judgments in synchronous code, decisions with floors, models that write
  natlang, one set of stages for two drivers. hosts.md: folder transactions, `runtime.decide`, composing drivers.

## Primitive changes this needs (candidates, decided while building)

1. **Decision distributions to the caller.** `readout: decision` records its distribution in the trace only; the
   Jev pattern needs code to act on probabilities. Host API, not a model-facing name.
2. **More than one model per runtime.** A program routes calls: a big model for hard passes or the agent, a small
   one for decisions. Today one runtime has one model.
3. Whatever the examples show to be missing; each goes into the shared runtime (Node and browser) with tests and
   the skills updated in the same push.

## Work split

Forked agents (owner allowed forks for this work), each in its own worktree and branch, merged by the main
session: compilers; database. The main session: pi port, primitive changes, skills, integration, coordination.
Lessons from each project are collected in `applications/<app>/LESSONS.md` and folded into the skills.
