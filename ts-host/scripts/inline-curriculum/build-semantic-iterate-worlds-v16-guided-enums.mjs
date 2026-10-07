#!/usr/bin/env node
import { resolve } from 'node:path';
import { worlds } from './semantic-iterate-worlds-v15-enum-data.mjs';
import { writeGuidedSoftIterateCandidate } from './semantic-iterate-worlds-v15-soft-guided-builder.mjs';

const args = process.argv.slice(2);
if (args.length !== 2 || args[0] !== '--out' || !args[1] || args[1].startsWith('--'))
  throw new Error('usage: node build-semantic-iterate-worlds-v16-guided-enums.mjs --out FRESH_DIRECTORY');
const revision = 'authored-semantic-iterate-worlds-v15/16-guided-soft-enum-final-types';
const out = resolve(args[1]);
const built = await writeGuidedSoftIterateCandidate({ out, worlds, revision, shapeVersion: 'v16' });
console.log(JSON.stringify({ out, source_sha256: built.sourceSha, worlds: built.rows.length,
  train: 6, test: 6, admission_granted: false,
  proof_kind: 'scaffold-assisted frozen CPU reference; not unassisted teacher success' }, null, 2));
