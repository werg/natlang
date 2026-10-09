/** Writes the byte-equality snapshot: node test/support/capture-builtin-prompts.mjs <out.json> */
import { writeFileSync } from 'node:fs';
import { renderBuiltinPrompts } from './builtin-prompts.mjs';

const first = await renderBuiltinPrompts(), second = await renderBuiltinPrompts();
for (const key of Object.keys(first)) {
  if (JSON.stringify(first[key]) === JSON.stringify(second[key])) continue;
  const a = JSON.stringify(first[key], null, 1).split('\n'), b = JSON.stringify(second[key], null, 1).split('\n');
  const at = a.findIndex((line, index) => line !== b[index]);
  throw new Error(`the rendering of ${key} is not deterministic at ${a[at]} <> ${b[at]}`);
}
for (const [name, value] of Object.entries(first))
  console.log(name, Array.isArray(value) ? `${value.length} entries` : typeof value === 'object' ? Object.keys(value).join(',') : `${String(value).length} chars`);
if (process.argv[2]) writeFileSync(process.argv[2], JSON.stringify(first, null, 1) + '\n');
