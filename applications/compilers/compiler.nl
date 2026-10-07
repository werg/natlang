---
description: An optimizing compiler for C, typed Python and Rust, end to end in natural language. Front end, middle end and AArch64 back end, every stage checked by running the program.
args:
  source: string
  language: "'c' | 'python' | 'rust'"
  level: Level
  inputs: string[]
returns: Compiled
---
Compile source, a program in language, the way an optimizing compiler does, with the stages in your folder. Each stage
is a function you call, and toolchain checks what the stages produce. inputs are the program's test inputs: run the
program once with each as standard input.

Front end, with the stages of the program's language (c, python or rust):
1. parse(source) builds the syntax tree. analyze(syntax) resolves names, infers types and checks the program. If either
   reports diagnostics, stop and return them, with empty ir and assembly.
2. declare(checked) lays out the module: the header (types, globals, constants, declarations) and each function's
   signature and tree. The header must pass toolchain.verify. When it declares runtime functions (`@rt_…`), runtime
   writes them, and they join the program's own functions.
3. lower(fn, context) generates each function's IR, all at once. A function's context is the header plus a `declare`
   line for every other function of the module.
The module is the header followed by every function. It must pass toolchain.verify. Its outputs under
toolchain.runIR on each input are the reference behaviour.

Middle end, per function, with the stages in opt:
1. opt.plan chooses the passes for level.
2. Run them in that order. mem2reg, gvn, licm and loops take the function's flow: compute it with opt.flow on the
   function as it is, and again only after a pass has changed it. inline also takes the definitions of the functions
   it calls, as callees.
3. After each pass, the module with the new version of the function must verify and give the reference output on every
   input. If it does not, call the same pass once more with problem saying what went wrong. If it is still wrong, keep
   the previous version.
Functions are independent within a pass: work on them concurrently.

Back end, with the stages in aarch64:
1. data turns the header into the data section.
2. For each function of the optimized module: select, then liveness of the result, then allocate with that liveness,
   then frame, then emit with the function's name. Then peephole the emitted function.
3. The program is the data section followed by every function's assembly. Under toolchain.runAssembly it must give the
   reference output on every input. When it does not, find the function at fault from the error and redo that
   function's stages, from select, with problem. If only peephole broke a function, use its emitted version.

Return the optimized module as ir and the program as assembly. Add a log line for every stage you ran on every
function: the stage, the function, accepted or kept the previous version, and why.
