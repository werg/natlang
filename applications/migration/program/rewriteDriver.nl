---
description: Rewrite a legacy driver whose steps are mechanism as a TypeScript helper and one instruction that calls it.
args:
  name: string
  instructions: string
  steps: KindedStep[]
  signature: string
  callable: string[]
  helperName: string
  problems: string[]
returns: DriverRewrite
---
name is a natural-language function with signature whose instructions run a fixed sequence of steps. steps are the
instructions split into steps, each with what it does. callable lists the functions in its folder with their
signatures; a TypeScript module in that folder imports each one as `import f from './f.nl';` and awaits it.

1. helper: the TypeScript source of the module `helperName.ts` in the folder. Its default export is an async
   function with the parameters of signature that does the steps in order: each step that calls a function becomes
   `const x = await f(...)`, each "mechanism" step becomes plain code over those values, and it returns the result.
   A "judgment" step that is not a call of a function in callable stays a question to the model: write it as
   ``await nl`...`(values)`` with `import { nl } from '@natlang/node';`. Import types from the types module the folder
   sees (`import type { T } from '../types.js';`, one `../` per folder level).
2. instructions: one sentence telling the function to return the result of `helperName` called with its arguments.

problems are what an exact check found wrong with your last answer; fix every one of them. Return { helper,
instructions }.
