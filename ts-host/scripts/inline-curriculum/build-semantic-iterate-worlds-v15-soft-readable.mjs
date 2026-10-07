#!/usr/bin/env node
import { resolve } from 'node:path';
import { writeSoftIterateCandidate } from './semantic-iterate-worlds-v15-soft-builder.mjs';

const args = process.argv.slice(2);
if (args.length !== 2 || args[0] !== '--out' || !args[1] || args[1].startsWith('--'))
  throw new Error('usage: node build-semantic-iterate-worlds-v15-soft-readable.mjs --out FRESH_DIRECTORY');
const built = await writeSoftIterateCandidate({ out: resolve(args[1]) });
console.log(JSON.stringify({ out: resolve(args[1]), source_sha256: built.sourceSha,
  worlds: built.rows.length, train: 6, test: 6, topology_changed: false, admission_granted: false }, null, 2));
