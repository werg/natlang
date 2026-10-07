#!/usr/bin/env node
/**
 * What the recursion guard costs eval code: a million-element map with a callback that makes no calls (unguarded),
 * one that calls a built-in (guarded), and a plain counted loop for reference. Run after `npm run build`.
 */
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const runner = fileURLToPath(new URL('../test/support/eval-case.mjs', import.meta.url));
const code = `const xs = Array.from({ length: 1_000_000 }, (_, i) => i);
const time = (run: () => unknown) => { const start = performance.now(); run(); return Math.round(performance.now() - start); };
const loop = time(() => { let total = 0; for (let i = 0; i < xs.length; i++) total += xs[i] * 2; return total; });
const callFree = time(() => xs.map(x => x * 2));
const calling = time(() => xs.map(x => Math.abs(x)));
return JSON.stringify({ loopMs: loop, callFreeMapMs: callFree, guardedMapMs: calling });`;
const output = JSON.parse(execFileSync(process.execPath, [runner, JSON.stringify({ code })], { encoding: 'utf8' }).trim().split('\n').at(-1));
if (output.error) throw new Error(output.error);
console.log(output.ok);
