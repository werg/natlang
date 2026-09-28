#!/usr/bin/env node
// Retired compatibility entry point. Technique-only preferences are not evidence of a worse result.
// Use build-preference-pairs.mjs for failures established by replay.
import { writeFile } from 'node:fs/promises';
import { parseArgs } from 'node:util';

const { values, positionals } = parseArgs({ allowPositionals: true, options: { out: { type: 'string' } } });
if (!positionals.length || !values.out) throw new Error('usage: pairs.mjs RESULTS.jsonl... --out PAIRS.jsonl');
// Technique differences alone do not establish a worse answer. Old pairs of this kind are retired.
await writeFile(values.out, '');
console.log(JSON.stringify({ pairs: 0, retired: 'technique-only preference pairs',
  replacement: 'build-preference-pairs.mjs verifies actual failures by replay' }));
