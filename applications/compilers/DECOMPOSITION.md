# Compilers: decomposition, part by part

Every part of every stage, with a decision:

- **fn**: its own natural-language function.
- **inline**: instructions inside its caller.
- **implicit**: left to the model.
- **crisp**: a TypeScript helper in a callable folder.
- **service**: the outside world.

The executors are small, fast models, so instructions spell rules out, and each function is one task a small model
can finish reliably.

## Policy

- **Natural language: the algorithmic content of a compiler.** That is everything a compiler textbook teaches:
  - parsing and semantic analysis, and IR generation;
  - every analysis (dominators, loops, liveness, induction variables) and every transformation;
  - instruction selection, register assignment, frame layout.
  A deterministic algorithm is still natural language when it *is* the compiler's content (dominators, phi
  placement). The model runs it in eval from the steps the instructions give.
- **Crisp: text plumbing around that content.**
  - lexing;
  - splitting a module into functions;
  - reading a function's CFG edges out of IR text;
  - escaping and numbering string constants;
  - printing assembly directives;
  - applying an already-decided map to code text (registers, frame offsets).
  None of these involve a decision a compiler book discusses.
- **Implicit: only what small models do reliably.** That means JSON, sorting and arithmetic in eval. Language
  semantics with rules (C's integer promotions, Rust's wrapping, Python's floor division) are spelled out.
- **Service: the checks.** The verifier, the IR interpreter, the assembler and the runner stand outside the
  compiler, and they never generate code.
- **Analyses are computed by the pass manager and passed to the passes that use them.** A function can call only its
  own folder, so this is the only way to share them, and it is also how LLVM's analysis manager works.
- **Passes split as LLVM splits them.** SCCP, InstCombine, SimplifyCFG, LoopSimplify, LoopRotate, IndVarSimplify,
  LSR and LoopUnroll are separate passes there, and they are separate functions here.

## Front end (per language)

| Part | Decision | Unit | Why |
|---|---|---|---|
| Lexing: source to tokens with lines (Python: INDENT/DEDENT; Rust: lifetimes, `..=`) | crisp | `<lang>/lex.ts` | A regular-language scan with no decisions. Small models mis-tokenize long sources. |
| Split into top-level declarations by bracket matching | crisp | `<lang>/lex.ts` | Mechanical. It lets each declaration be parsed on its own. |
| Parse one top-level declaration (statements, and expressions by a spelled precedence/associativity table) | fn, per declaration, in parallel | `<lang>/parse` | The grammar is the front end's content. One declaration fits a small model's call. |
| File scope: structs, globals, prototypes, functions to a symbol table | fn | `<lang>/scope` | Its own data, needed by every function's analysis. |
| Name resolution: block scopes and shadowing, each name to its declaration | fn, per function | `<lang>/resolve` | Scope bookkeeping is a separate task from typing. Small models conflate them. |
| Typing and implicit conversions: each expression's type by spelled rules (C: usual arithmetic conversions, promotions, decay; Python: one type per local, int/float rules; Rust: literal inference, method resolution, auto-ref) | fn, per function | `<lang>/types` | The densest rules of the front end. They get a call of their own. |
| Rust borrow checking: moves and borrows along the control flow | fn, per function | `rust/borrows` | A distinct dataflow algorithm with its own diagnostics. |
| Diagnostics: errors with lines | inline | each analysis | Reported where they are found. |
| Struct layout and type mapping (C: int i32, long i64, …; Python: list as `%list`; Rust: widths, `%vec`) | inline | `<lang>/declare` | A spelled table, small. |
| Global initializers, constant-folded | inline | `<lang>/declare` | Few. |
| String literals: collect, number `@.str.N`, escape to `c"…\00"` | crisp | `<lang>/strings.ts` | Escaping and numbering are mechanical. Small models miscount lengths. |
| Library and runtime declarations, with each runtime function's contract | inline | `<lang>/declare` | A spelled list of the allowed contracts. |
| Signatures of every function | inline | `<lang>/declare` | Follows from the type table. |
| IR generation for one function: allocas, statements, expressions, short-circuit, loops, returns, with a template per construct | fn, per function, in parallel | `<lang>/lower` | The core translation. One function's body must come out as one well-formed IR function. |
| Module assembly: header, then functions, with duplicate declarations dropped | crisp | driver | Text plumbing. |
| Runtime library: one definition per declared `@rt_` contract | fn, per contract, in parallel | `runtime` | Each one is independent and can be verified on its own. Today a single call writes them all. |

## Middle end (opt/)

| Part | Decision | Unit | Why |
|---|---|---|---|
| Pipeline per level (O1: sroa, mem2reg, instcombine, simplifycfg, adce; O2: adds inline, sccp, gvn, loop-simplify, licm, loop-rotate, indvars, lsr, unroll; O3: a second round), skipping passes that cannot apply | crisp | `opt/pipeline.ts` | LLVM's pipelines are fixed lists. Applicability (no allocas, no loops, no calls) is a mechanical test. Today a model call picks it. |
| CFG edges from IR text: blocks, successors, predecessors | crisp | `opt/cfg.ts` | Parsing terminators is plumbing. |
| Dominator tree (Cooper–Harvey–Kennedy, reverse postorder) | fn | `opt/dominators` | Analysis content, with a spelled iterative algorithm. |
| Dominance frontiers | fn | `opt/frontiers` | Its own small algorithm. Only mem2reg needs it. |
| Natural loops: back edges, bodies, latches, exits, preheaders, nesting | fn | `opt/loops-info` | Its own data, used by five passes. |
| Induction variables and trip counts (SCEV, restricted to affine add recurrences) | fn | `opt/scev` | Its own data, used by indvars, lsr and unroll. |
| sroa: split aggregates accessed at constant offsets | fn | `opt/sroa` | Its own pass in LLVM. Small. |
| mem2reg: which allocas are promotable | fn | `opt/mem2reg/promotable` | A rule check with its own result, which is data. |
| mem2reg: phi placement at the iterated dominance frontier of each alloca's stores | fn | `opt/mem2reg/place` | The textbook algorithm. Its result is data (block, alloca). |
| mem2reg: renaming over the dominator tree with value stacks, rewriting loads and stores | fn | `opt/mem2reg/rename` | The IR rewrite. It takes the placements as input. |
| mem2reg: prune dead and trivial phis | inline | `opt/mem2reg/rename` | A short closing step. |
| sccp: lattice solve with executable edges (constants, dead edges) | fn | `opt/sccp` | Its own pass. Its rewrite is a short closing step. |
| instcombine: a spelled rule table (identities, power-of-two multiplies and divides, reassociation, comparisons, casts) applied to a fixpoint | fn | `opt/instcombine` | One pass, with its rules listed. |
| simplifycfg: merge blocks, thread jumps, diamonds to select, remove unreachable blocks | fn | `opt/simplifycfg` | Its own pass. |
| gvn: value numbering of pure expressions over the dominator tree | fn | `opt/gvn` | Its own data. |
| gvn: redundant loads by memory dependence, with spelled alias rules | fn | `opt/gvn-loads` | A different analysis, about memory. Kept separate for small models. |
| loop-simplify: preheaders, a single latch, dedicated exits | fn | `opt/loop-simplify` | LLVM's LoopSimplify. licm, rotate, indvars and unroll rely on it. |
| licm: invariance in dominator order, safety rules, hoisting | fn | `opt/licm` | One pass. |
| loop-rotate: guard and bottom test, phi fix-up | fn | `opt/loop-rotate` | Its own pass. |
| indvars: canonical induction variables, exit test rewrite | fn | `opt/indvars` | Its own pass. Uses scev. |
| lsr: strength reduction of derived induction variables | fn | `opt/lsr` | Its own pass. Uses scev. |
| unroll: full unrolling (constant trip ≤ 8, body ≤ 20) and runtime by 2 with a remainder | fn | `opt/unroll` | Its own pass. Uses scev. |
| inline: cost per call site (instruction count, constant arguments, single caller) | inline | `opt/inline` | Spelled arithmetic. Small. |
| inline: clone with unique renaming, splice with a return phi | fn | `opt/inline` | The rewrite. |
| adce: mark from live roots, sweep | fn | `opt/adce` | Its own pass. CFG cleanup belongs to simplifycfg. |
| After each pass: verify, and run on the inputs | service | `toolchain` | The check. |
| Retry with the problem, else keep the previous version | inline | pass manager | Orchestration. |

## Back end (aarch64/)

| Part | Decision | Unit | Why |
|---|---|---|---|
| Data: initializer layout (padding, endianness, doubles as bits) | fn | `aarch64/data` | Layout rules. |
| Data: section directives and symbol naming | crisp | `aarch64/print.ts` | Formatting. |
| Calling convention: AAPCS64 argument and result registers, variadic calls, the stack at calls | inline | `aarch64/select` | A spelled table. It cannot be separated from selection. |
| Instruction selection: a pattern per IR instruction, over virtual registers, emitting PHI pseudo-instructions | fn, per function | `aarch64/select` | The core. Pattern tables spelled out. |
| PHI elimination: copies in predecessors, critical edges split | fn | `aarch64/phi-elim` | LLVM's PHIElimination. A separate step for small models. |
| Live intervals by backward dataflow; calls crossed | fn | `aarch64/liveness` | Analysis. |
| Linear-scan assignment: pools, expiry, spill choice, giving the register or slot of each virtual register | fn | `aarch64/assign` | The allocation decision. Its result is data. |
| Rewrite with the assignment: physical registers, reload and store around spills | crisp | `aarch64/rewrite.ts` | Applying a decided map is mechanical. |
| Frame layout: frame record, saved registers in pairs, slots by alignment, 16-byte total | fn | `aarch64/frame` | Layout decisions. Its result is data (offsets). |
| Prologue and epilogue insertion, slot references to `[sp, #off]`, large offsets through x16 | crisp | `aarch64/apply-frame.ts` | Applying a decided layout, from fixed templates. |
| Peepholes: a spelled rule table over windows of 2–3 instructions | fn | `aarch64/peephole` | One pass, with its rules listed. |
| Emission: directives, labels, `.size` | crisp | `aarch64/print.ts` | Formatting. Today this is a model call (`emit`). |
| Running the program and comparing outputs | service | `toolchain` | The check. |

## Drivers

`compiler.nl` sequences the stages in natural language (the pass manager), and `index.ts` is the checked driver. Both
call the crisp helpers through the callable folder, and both check with the `toolchain` service.

## Changes from today

- New functions of their own:
  - front end: scope, resolve, types, borrows, and the runtime written per contract;
  - analyses: dominators, frontiers, loops-info, scev;
  - mem2reg's parts: promotable, place, rename;
  - what `simplify` bundled: sccp, instcombine, simplifycfg;
  - what `loops` bundled: loop-rotate, indvars, lsr, unroll;
  - middle end, other passes: sroa, gvn-loads, loop-simplify;
  - back end: phi-elim, assign.
- New crisp plumbing:
  - lex, strings and cfg;
  - the pipeline;
  - rewrite, apply-frame and print (emission and data directives);
  - module assembly.
- `plan` and `emit` stop being model calls.
