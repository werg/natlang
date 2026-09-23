/**
 * An npm package installed in this directory (`npm install` here first), used from handwritten
 * callable-folder TypeScript (`main/is_numeric.ts`) and from a natural-language function's eval (scripted here).
 */
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createNatlangRuntime, loadNatlang } from '../../ts-host/dist/index.js';

const workspace = dirname(fileURLToPath(import.meta.url));
let turn = 0;
const runtime = createNatlangRuntime({ workspace,
  model: () => ++turn === 1 ? { calls: [['eval', { code: 'import isNumber from "is-number"; return isNumber(value);' }]] }
    : { text: 'done' } });
const main = loadNatlang(join(workspace, 'main.nl'));
const handwritten = await runtime.run(() => main.is_numeric('42'));
const natural = await runtime.run(() => main('42'));
if (handwritten !== true || natural !== true) throw new Error(JSON.stringify({ handwritten, natural }));
console.log(JSON.stringify({ handwritten, natural, workspace }));
