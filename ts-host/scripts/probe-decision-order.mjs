#!/usr/bin/env node
/**
 * Does a model's decision follow the order its options are listed in? Scores one captured readout request with the
 * options listed in each cyclic rotation and prints every option's log-probability per rotation. An option that wins
 * whenever it is listed first is read by position, not by the evidence; a readout that reads the evidence picks the same
 * option in every row, and its scores move by a few nats, not by ten.
 *
 *   node scripts/probe-decision-order.mjs REQUEST.json [--endpoint URL] [--frame value|result-object] [--model ID]
 *
 * REQUEST.json is a readout request (`{ messages, options }`), or a /v1/natlang/score body whose first item is one (its
 * `continuations` are the options) whose last message is the runtime's `decisionPrompt` for its options. The server is the
 * natlang llama.cpp fork's (default: the shared server a running natlang session published). Reads only; starts nothing.
 */
import { readFileSync } from 'node:fs';
import { decisionPrompt, framedDecision } from '../dist/decision-frame.js';

const args = process.argv.slice(2);
const option = name => { const at = args.indexOf(name); return at < 0 ? undefined : args[at + 1]; };
const file = args.find((arg, index) => !arg.startsWith('--') && !args[index - 1]?.startsWith('--'));
if (!file || !option('--endpoint')) { console.error('usage: probe-decision-order.mjs REQUEST.json --endpoint URL [--frame value|result-object] [--model ID]'); process.exit(2); }
const frame = option('--frame') ?? 'value';
const parsed = JSON.parse(readFileSync(file, 'utf8'));
const captured = parsed.body ?? parsed;
const item = captured.items?.[0] ?? captured;
const options = item.options ?? item.continuations;
const last = item.messages.at(-1);
if (last.content !== decisionPrompt(options)) throw new Error('the last message is not the readout prompt of the options (the runtime\'s decisionPrompt)');

const rows = [];
for (let shift = 0; shift < options.length; shift++) {
  const listed = [...options.slice(shift), ...options.slice(0, shift)];
  // The options keep their index; only the order they are listed in the prompt changes.
  const request = framedDecision({ messages: [...item.messages.slice(0, -1), { ...last, content: decisionPrompt(listed) }], options: listed }, frame);
  const reply = await fetch(`${option('--endpoint').replace(/\/$/, '')}/v1/natlang/score`, { method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model: option('--model') ?? captured.model, items: [{ messages: request.messages, continuations: request.options }] }) });
  if (!reply.ok) throw new Error(`score failed: HTTP ${reply.status} ${(await reply.text()).slice(0, 300)}`);
  const scores = (await reply.json()).results[0];
  if (scores.error) throw new Error(scores.error);
  const byOption = new Map(listed.map((value, index) => [value, scores.log_probs[index]]));
  rows.push({ listed, logProbs: options.map(value => byOption.get(value)) });
}
const width = Math.max(...options.map(value => value.length), 8);
console.log(`frame ${frame}; log-probability of each option, one row per listing order, the winner of a row marked <`);
console.log(['listed first'.padEnd(width + 2), ...options.map(value => value.padStart(width + 2))].join(''));
for (const row of rows) {
  const best = row.logProbs.indexOf(Math.max(...row.logProbs));
  console.log([row.listed[0].padEnd(width + 2), ...row.logProbs.map((value, index) => `${value.toFixed(1)}${index === best ? '<' : ' '}`.padStart(width + 2))].join(''));
}
const winners = rows.map(row => row.logProbs.indexOf(Math.max(...row.logProbs)));
const firstWins = rows.filter((row, shift) => winners[shift] === options.indexOf(row.listed[0])).length;
console.log(`winner follows the listing in ${firstWins} of ${rows.length} orders; ${new Set(winners).size === 1 ? 'one option wins in every order' : 'the winner changes with the order'}`);
