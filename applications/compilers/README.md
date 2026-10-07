# Optimizing compilers in natural language

C (a C99 subset), typed Python (annotated functions over int, float, bool, str, list) and Rust (a subset: scalars,
structs with methods, `Vec`, borrow checking included) compiled to AArch64 Linux through LLVM IR. The compiler is
natural language from end to end. Its stages follow a production compiler's, each a function whose instructions
state its algorithm. Crisp code only checks what the stages produce.

```
front end, per language (c/, python/, rust/)
  parse      parser: source to a syntax tree (clang's, Python ast's or syn's node kinds)
  analyze    semantic analysis: scopes, name resolution, type inference, implicit conversions, errors;
             Rust's borrow checking
  declare    IR generation for the module: types, globals, strings, runtime contracts, signatures
  lower      IR generation for one function, from its checked tree (clang -O0 style)
runtime      Python's lists and Rust's vectors, input and panics, written from the contracts the header declares

middle end (opt/), shared, on LLVM IR
  plan       pass manager: the pipeline for one function at -O1/-O2/-O3
  flow       analysis: CFG, dominator tree (Cooper-Harvey-Kennedy), dominance frontiers, natural loops
  mem2reg    SROA, then phi placement at iterated dominance frontiers and renaming over the dominator tree
  simplify   SCCP over a lattice with executable edges, InstCombine rules, SimplifyCFG
  gvn        value numbering over the dominator tree; redundant loads by memory dependence
  licm       preheaders, invariance in dominator order, safe hoisting
  loops      rotation, induction variables and strength reduction, trip counts, full and by-2 unrolling
  inline     cost model, clone with renaming, splice with a return phi
  dce        mark-and-sweep from live roots, CFG cleanup

back end (aarch64/)
  data       data emission: globals, constants, strings with LLVM's layout
  select     instruction selection to machine code over virtual registers: AAPCS64, patterns, phi elimination
  liveness   analysis: live intervals by backward dataflow, calls crossed
  allocate   linear-scan register allocation with spilling; caller- or callee-saved by calls crossed
  frame      frame lowering: layout, prologue and epilogue, frame indices to sp offsets
  emit       assembly printing: directives and labels
  peephole   machine peepholes: addressing modes, cbz/tbz, madd/msub, ldp/stp
```

The stages pass typed values (`types.ts`):

- `Syntax` and `Checked`: trees of `Node`s;
- `ModuleFrame`: the header, plus each function's signature and tree;
- `Flow`: control flow;
- `MachineCode`: virtual-register code;
- `Liveness`: live intervals.

Two drivers run the stages:

- **`compiler.nl`, the compiler.** This is the default. The pass manager is written as instructions. It parses and
  analyzes the program, lays out the module, and generates each function's IR. Then it plans and runs passes per
  function, computing control flow when a pass needs it, and runs the back end chain. It checks every stage with
  the `toolchain` service, retries a stage once with the problem, and otherwise keeps the previous version. The
  host only checks the final program against gcc, CPython or rustc.
- **`index.ts`, the checked driver (`--checked`).** It does the same orchestration in TypeScript and checks each
  result. After each stage, the module must pass LLVM's verifier and print the same output on the test inputs; this
  is translation validation by testing. A rejected stage is sent back once with the reason. Code generation works
  on virtual registers until frame lowering, so each function's chain from select to emit is checked as a whole.
  Each emitted function is spliced into LLVM's own assembly for the rest of the program, so a fault is traced to one
  function. Stage calls run concurrently, at most 4 in flight.

`toolchain.ts` and `toolchain.py` stand for the outside world. They parse, verify and JIT-run IR with llvmlite
(LLVM 22), and assemble and link with gcc. They check code; they never generate it.

## Running

```sh
python3 -m venv ~/llvm-venv && ~/llvm-venv/bin/pip install llvmlite
export NATLANG_LLVM_PYTHON=~/llvm-venv/bin/python
natlang run applications/compilers -- compile prog.c|prog.py|prog.rs -O2 --input in.txt [--checked] [--out DIR]
natlang run applications/compilers -- bench [fib sieve ...] [--checked]
```

`bench` compiles `bench/c/*.c`, `bench/python/*.py` and `bench/rust/*.rs` and checks them against gcc -O0 (C),
CPython (Python) or rustc -O (Rust; `$RUSTC`, `~/.cargo/bin` or PATH). It then times them against gcc -O0/-O2,
CPython or rustc -O. Results go to `natlang-cc-out/` (`*.ll`, `*.s`, `*.report.json`, `bench.json`).

## Status

Every stage, both drivers and the retry/reject path are tested with scripted models and the real toolchain
(`ts-host/test/compilers.test.mjs`). The live benchmark run on the development model is pending; results will be
recorded here.
