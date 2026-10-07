#!/usr/bin/env node
import { resolve } from 'node:path';
import { writeGuidedSoftIterateCandidate } from './semantic-iterate-worlds-v15-soft-guided-builder.mjs';

const args = process.argv.slice(2);
if (args.length !== 2 || args[0] !== '--out' || !args[1] || args[1].startsWith('--'))
  throw new Error('usage: node build-semantic-iterate-worlds-v15-soft-guided.mjs --out FRESH_DIRECTORY');
const built = await writeGuidedSoftIterateCandidate({ out: resolve(args[1]) });
console.log(JSON.stringify({ out: resolve(args[1]), source_sha256: built.sourceSha,
  worlds: built.rows.length, train: 6, test: 6, admission_granted: false,
  proof_kind: 'scaffold-assisted authored CPU reference; not unassisted teacher success' }, null, 2));
