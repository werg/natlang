#!/usr/bin/env node
/**
 * Gate 3 of plans/BATCHED_EXECUTION.md: N decision-readout classifications over K labels, scored three ways on one
 * Neuralese server: (A) N*K requests, one option each, through /v1/neuralese/decide; (B) N requests with K options
 * (one prompt pass per item); (C) the wired path: `neuraleseServerModelTurn(...).decide` issued with Promise.all, which
 * coalesces into POST /v1/natlang/score. Scores must agree; throughput and request counts are reported.
 *
 *   node scripts/bench_batched_score.mjs ENDPOINT N [OUT.json]
 */
import { writeFileSync } from 'node:fs';
import { neuraleseServerModelTurn } from '../ts-host/dist/model/neuralese-server.js';

const [endpoint, count = '1000', out] = process.argv.slice(2);
const N = Number(count);
const labels = ['billing', 'technical', 'spam'];
const rubric = 'billing: charges, invoices and payments; technical: outages, errors and bugs; spam: advertising';
const subjects = ['charged twice for invoice', 'login fails with error', 'free cruise offer', 'card declined at checkout', 'export crashes the app',
  'cheap watches limited stock', 'refund for annual plan', 'API timeouts for customers', 'boost your SEO now'];
const items = Array.from({ length: N }, (_, index) => ({
  messages: [{ role: 'system', content: 'Pick the label from the rubric that fits the ticket. Answer with the label only.' },
    { role: 'user', content: `rubric: ${rubric}\nticket: ${subjects[index % subjects.length]} #${index}` }],
  options: labels }));

const post = async (path, body) => {
  const response = await fetch(endpoint + path, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status} ${(await response.text()).slice(0, 300)}`);
  return response.json();
};
let requests = 0;
const counted = async (path, body) => { requests++; return post(path, body); };
const timed = async run => { requests = 0; const start = performance.now(); const result = await run(); return { result, ms: performance.now() - start, requests }; };
const percentile = (values, p) => [...values].sort((a, b) => a - b)[Math.min(values.length - 1, Math.floor(p * values.length))];

const latencies = { A: [], B: [] };
const A = await timed(async () => {
  const scores = [];
  for (const item of items) {
    const row = [];
    for (const option of item.options) {
      const start = performance.now();
      row.push((await counted('/v1/neuralese/decide', { messages: item.messages, options: [option] })).log_probs[0]);
      latencies.A.push(performance.now() - start);
    }
    scores.push(row);
  }
  return scores;
});
const B = await timed(async () => {
  const scores = [];
  for (const item of items) {
    const start = performance.now();
    scores.push((await counted('/v1/neuralese/decide', { messages: item.messages, options: item.options })).log_probs);
    latencies.B.push(performance.now() - start);
  }
  return scores;
});
// C: the driver. Requests are counted by wrapping fetch.
const realFetch = globalThis.fetch;
let cRequests = 0, cPaths = new Set();
globalThis.fetch = (url, init) => { cRequests++; cPaths.add(new URL(String(url)).pathname); return realFetch(url, init); };
const driver = neuraleseServerModelTurn({ endpoint, model: 'natlang-neuralese' });
const C0 = performance.now();
const settled = await Promise.allSettled(items.map(item => driver.decide(item)));
const cMs = performance.now() - C0;
globalThis.fetch = realFetch;
const C = settled.map(result => result.status === 'fulfilled' ? result.value.log_probs : null);
const failed = settled.filter(result => result.status === 'rejected').length;

let maxAB = 0, maxBC = 0, argmaxAgree = 0;
items.forEach((_, i) => {
  A.result[i].forEach((v, k) => { maxAB = Math.max(maxAB, Math.abs(v - B.result[i][k])); });
  if (C[i]) {
    B.result[i].forEach((v, k) => { maxBC = Math.max(maxBC, Math.abs(v - C[i][k])); });
    const top = scores => scores.indexOf(Math.max(...scores));
    if (top(A.result[i]) === top(C[i])) argmaxAgree++;
  }
});
const row = (name, ms, reqs, lat) => ({ name, total_s: +(ms / 1000).toFixed(2), requests: reqs, decisions_per_s: +(N / (ms / 1000)).toFixed(2),
  ...(lat ? { latency_p50_ms: +percentile(lat, 0.5).toFixed(1), latency_p95_ms: +percentile(lat, 0.95).toFixed(1) } : {}) });
const report = { endpoint, items: N, options_per_item: labels.length,
  rows: [row('A: N*K single-option /v1/neuralese/decide', A.ms, A.requests, latencies.A),
    row('B: N per-item /v1/neuralese/decide (K options)', B.ms, B.requests, latencies.B),
    { ...row('C: driver decide + Promise.all -> /v1/natlang/score', cMs, cRequests), paths: [...cPaths], failed }],
  agreement: { max_abs_diff_A_vs_B: maxAB, max_abs_diff_B_vs_C: maxBC, argmax_agree_A_vs_C: `${argmaxAgree}/${N}` },
  speedup_C_over_A: +(A.ms / cMs).toFixed(2), speedup_C_over_B: +(B.ms / cMs).toFixed(2) };
console.log(JSON.stringify(report, null, 2));
if (out) writeFileSync(out, JSON.stringify(report, null, 2) + '\n');
