---
description: An optimizing compiler for C and typed Python, end to end in natural language. Front end, middle end and AArch64 back end, every stage checked by running the program.
args:
  source: string
  language: "'c' | 'python'"
  level: Level
  inputs: string[]
returns: Compiled
---
Compile source, a program in language, the way an optimizing compiler does, with the stages in your folder. Each
stage is a function you call, and toolchain checks what the stages produce. inputs are the program's test inputs:
run it once with each as standard input.

Front end. Read the program with c.declare or python.declare. If its diagnostics are not empty, stop and return
them, with empty ir and assembly. For Python, python.runtime writes the runtime library the header declares, and
its functions join the program's own. Lower every function with c.lower or python.lower, all at once. A function's
context is the header plus a `declare` line for every other function of the module. The module is the header
followed by every function. It must pass toolchain.verify, and its outputs under toolchain.runIR on each input are
the reference behaviour.

Middle end. For each function, opt.plan chooses passes for level; run them in that order (opt.inline also takes the
definitions of the functions it calls as callees). After each pass, the module with the new version of the function
must verify and give the reference output on every input. If it does not, call the same pass once more with problem
saying what went wrong; if it is still wrong, keep the previous version. Functions are independent within a pass:
work on them concurrently.

Back end. aarch64.data turns the header into the data section. For each function of the optimized module, run
aarch64.select, then aarch64.allocate, then aarch64.peephole. The program is the data section followed by every
function's assembly. Under toolchain.runAssembly it must give the reference output on every input. When it does
not, find the function at fault from the error and redo that function's stages with problem. If peephole broke a
function, use its allocated version.

Return the optimized module as ir and the program as assembly. Add a log line for every stage you ran on every
function: the stage, the function, accepted or kept the previous version, and why.
