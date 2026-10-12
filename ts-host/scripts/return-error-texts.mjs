#!/usr/bin/env node
// The model-facing text of return_result's value rejections in every wording (native/return-errors.ts), for the live
// measurement of scripts/error_text_replay.py. Input (argv[2], JSON): { "<context id>": { "type": "Judgment",
// "problem": { "kind": "misplaced", "fields": ["verdict"] } } }. Output: { "<variant>": { "<context id>": "<line>" } },
// the line exactly as Reject renders it after the tool result's "rejected" status line.
import { readFileSync } from 'node:fs';
import { RETURN_ERROR_TEXTS, returnValueRejection } from '../dist/native/return-errors.js';
import { Reject } from '../dist/native/values.js';

const spec = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const out = {};
for (const variant of RETURN_ERROR_TEXTS) {
  out[variant] = {};
  for (const [id, { type, problem }] of Object.entries(spec))
    out[variant][id] = new Reject([returnValueRejection(variant, problem, type)]).message;
}
process.stdout.write(JSON.stringify(out, null, 1) + '\n');
