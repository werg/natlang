/**
 * An npm dependency declared in this directory's package.json, used from handwritten callable-folder
 * TypeScript (`main/is_numeric.ts`) and from a natural-language function's eval (scripted here).
 */
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { ApplicationPackages, createNatlangRuntime, loadNatlang } from '../../ts-host/dist/index.js';

const workspace = dirname(fileURLToPath(import.meta.url));
if (process.argv.includes('--install')) await new ApplicationPackages(workspace).prepareDependencies();
let turn = 0;
const runtime = createNatlangRuntime({ workspace, network: true,
  model: () => ++turn === 1 ? { calls: [['eval', { code: 'import isNumber from "is-number"; result = isNumber(value);' }], ['mark_lines', { start: 1, end: 2 }]] }
    : { text: 'done' } });
const main = loadNatlang(join(workspace, 'main.nl'));
const handwritten = await runtime.run(() => main.is_numeric('42'));
const natural = await runtime.run(() => main('42'));
if (handwritten !== true || natural !== true) throw new Error(JSON.stringify({ handwritten, natural }));
console.log(JSON.stringify({ handwritten, natural, workspace }));
