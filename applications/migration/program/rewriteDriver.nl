---
description: Rewrite a legacy driver whose steps are mechanism as a TypeScript helper and one instruction that calls it.
args:
  name: string
  instructions: string
  steps: KindedStep[]
  signature: string
  callable: string[]
  types: string
  helperName: string
  previous: DriverRewrite | null
  problems: string[]
returns: DriverRewrite
---
name is a natural-language function with signature whose instructions run a fixed sequence of steps. steps are the
instructions split into steps, each with what it does. callable lists the functions in its folder with their
signatures; a TypeScript module in that folder imports each one as `import f from './f.nl';` and awaits it. types is
the program's types module, which declares every type the signatures name.

1. helper: the TypeScript source of the module `helperName.ts` in the folder. Its default export is an async
   function whose parameters are the arguments of signature, in order (`args: { a: A, b: B }` declares `(a: A, b: B)`),
   and that does the steps in order: each step that calls a function becomes `const x = await f(a, b)`, passing the
   arguments in the order callable declares them, each "mechanism" step becomes plain code over those values, and it
   returns the result.
   A "judgment" step that is not a call of a function in callable stays a question to the model: write it as
   ``await nl`...`(values)`` with `import { nl } from '@natlang/node';`. Import types from the types module the folder
   sees (`import type { T } from '../types.js';`, one `../` per folder level).
2. instructions: one sentence telling the function to return the result of `helperName` called with its arguments.

previous is your last answer, when there was one, and problems are what an exact check (the TypeScript compiler and
natlang check) found wrong with it; fix every one of them and keep the rest. Return { helper, instructions }.
