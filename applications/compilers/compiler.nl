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

Checking a stage. A stage's answer is accepted when it passes the check named for it below. When it does not, call the
same stage once more with problem saying what went wrong (the verifier's message, or how the output changed). If the
second answer is also wrong, the stage has failed: in the middle end keep the previous version, and in the front end
stop and return the diagnostics. A call that raises an error counts as a wrong answer, with the error as the problem.
The check for a function of the middle end is that the module with the new version of the function passes
toolchain.verify, defines the same name, and gives the reference output on every input.

Front end, with the stages of the program's language (c, python or rust):
1. checked = analyze(parse(source)): parse builds the syntax tree, and analyze resolves names, infers types and checks
   the program. The diagnostics of both arrive in checked.diagnostics. If checked reports diagnostics, stop and return
   them, with empty ir and assembly.
2. declare(checked) lays out the module: the header (types, globals, constants, declarations) and each function's
   signature and tree. The header must pass toolchain.verify (checked and retried like every stage). When it declares runtime functions (`@rt_…`), runtime
   writes them, and they join the program's own functions; with the header they must pass toolchain.verify.
3. lower(fn, context) generates each function's IR, all at once; each must define fn's name and pass toolchain.verify
   with its context. A function's context is the header plus a `declare`
   line for every other function of the module.
The module is the header followed by every function. It must pass toolchain.verify. Its outputs under
toolchain.runIR on each input are the reference behaviour.

Middle end, per function, with the stages in opt:
1. opt.plan chooses the passes for level. When plan fails, use mem2reg, simplify, dce.
2. Run them in that order. mem2reg, gvn, licm and loops take the function's flow: compute it with opt.flow on the
   function as it is, and again only after a pass has changed it. inline also takes the definitions of the functions
   it calls, as callees.
3. After each pass, the module with the new version of the function must verify and give the reference output on every
   input (the check above); a wrong answer is retried once and then dropped, keeping the previous version.
4. When every function has had all of its passes, the whole module must verify and give the reference output. Passes
   were checked one at a time, so when the whole module is wrong, go back through the functions in source order,
   restoring each one's lowered version, until the module is right.
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
