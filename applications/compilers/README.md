# Optimizing compilers in natural language

C (a C99 subset), typed Python (annotated functions over int, float, bool, str, list) and Rust (a subset: scalars,
structs with methods, `Vec`, borrow checking included) compiled to AArch64 Linux through LLVM IR. Every compiler
stage is a natural-language function. Crisp code only checks what the stages produce.

```
source ─► front end (per language) ─► middle end (shared, LLVM IR) ─► back end (AArch64)
          c/declare     python/declare  opt/plan     pass manager       aarch64/data      data section
          c/lower       python/lower    opt/mem2reg  SSA construction   aarch64/select    instruction selection
          rust/declare  rust/lower      opt/simplify InstCombine, SCCP  aarch64/allocate  register allocation
                                        opt/gvn      redundancy         aarch64/peephole  machine peepholes
                                        opt/licm     invariant motion
                                        opt/loops    unrolling, strength reduction
                                        opt/inline   inlining
                                        opt/dce      dead code, CFG cleanup
          runtime: Python's lists and Rust's vectors, input and panics, written from the contracts the header declares
```

The stages live in `compiler/`, the callable folder of `compiler.nl`. Two drivers run them:

- **`compiler.nl`, the pure pipeline.** The pass manager is written as instructions. It reads the program, lowers
  each function, plans and runs passes per function, and runs the back end. It checks every stage with the
  `toolchain` service, retries a stage once with the problem, and otherwise keeps the previous version. The host
  only checks the final program against gcc or CPython. Use `--pure`.
- **`index.ts`, the checked driver.** It does the same orchestration in TypeScript. After each stage the module
  must pass LLVM's verifier and print the same output on the test inputs; this is translation validation by
  testing. A rejected stage is sent back once with the reason. In the back end, each generated function is spliced
  into LLVM's own assembly for the rest of the program, so a fault is traced to one function. Stage calls run
  concurrently, at most 4 in flight.

`toolchain.ts` and `toolchain.py` stand for the outside world. They parse, verify and JIT-run IR with llvmlite
(LLVM 22), and assemble and link with gcc. They check code; they never generate it.

## Running

```sh
python3 -m venv ~/llvm-venv && ~/llvm-venv/bin/pip install llvmlite
export NATLANG_LLVM_PYTHON=~/llvm-venv/bin/python
natlang run applications/compilers -- compile prog.c|prog.py|prog.rs -O2 --input in.txt [--pure] [--out DIR]
natlang run applications/compilers -- bench [fib sieve ...] [--pure]
```

`bench` compiles `bench/c/*.c`, `bench/python/*.py` and `bench/rust/*.rs` and checks them against gcc -O0 (C),
CPython (Python) or rustc -O (Rust; `$RUSTC`, `~/.cargo/bin` or PATH). It then times them against gcc -O0/-O2,
CPython or rustc -O. Results go to `natlang-cc-out/` (`*.ll`, `*.s`, `*.report.json`, `bench.json`).

## Status

Every stage, both drivers and the retry/reject path are tested with scripted models
(`ts-host/test/compilers.test.mjs`). The live benchmark run on the development model is pending; results will be
recorded here.
