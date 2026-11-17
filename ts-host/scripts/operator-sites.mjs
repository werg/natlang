#!/usr/bin/env node
/**
 * The runtime's operator sites as a batch call, for training rows built outside a run
 * (scripts/neuralese_data/operator_rows.py; plans/neuralese/OPERATORS_RESTORE_20261117.md §9): one JSON request per
 * input line (`OperatorSiteRequest`), one JSON reply per output line, the
 * runtime's own `operatorSite` (native/operator-site.ts): the operator call's messages and tools, the consumer block
 * with the state in place, the crisp text and whether the runtime compacts the value. Needs the built dist (`npm run build:node`).
 */
import { createInterface } from 'node:readline';
import { operatorSite } from '../dist/native/operator-site.js';

const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
for await (const line of lines) {
  if (!line.trim()) continue;
  let reply;
  try { reply = operatorSite(JSON.parse(line)); } catch (error) { reply = { error: String(error?.message ?? error) }; }
  process.stdout.write(JSON.stringify(reply) + '\n');
}
