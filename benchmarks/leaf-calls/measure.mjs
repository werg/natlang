#!/usr/bin/env node
/**
 * Live measurement of the leaf-call surfaces (benchmarks/leaf-calls/README.md).
 *
 *   node benchmarks/leaf-calls/measure.mjs --variant tools|direct|decision --samples 2 --jobs 4 --out results/tools.json
 *   node benchmarks/leaf-calls/measure.mjs --compare results/tools.json results/direct.json
 *
 * A variant is a set of runtime model settings on the configured runtime `natlang run` uses (its selected profile,
 * NATLANG_PROFILE or NATLANG_CONFIG_HOME choose the model) with the profile's sampling. Results pair by (case, sample)
 * across variants. The output keeps every outcome, its turns and its tokens.
 */
import { appendFileSync, existsSync, readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { isDeepStrictEqual } from 'node:util';

const here = dirname(fileURLToPath(import.meta.url));
const dist = join(here, '..', '..', 'ts-host', 'dist');
const VARIANTS = {
  tools: { leafCalls: 'tools', decisionReadout: 'declared' },
  direct: { leafCalls: 'direct', decisionReadout: 'declared' },
  decision: { leafCalls: 'direct', decisionReadout: 'finite-returns' },
};

const args = process.argv.slice(2);
const flag = (name, fallback) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : fallback; };

if (args[0] === '--compare') compare(args[1], args[2]);
else await measure(flag('--variant'), Number(flag('--samples', '2')), flag('--out'), Number(flag('--jobs', '1')));

async function measure(variant, samples, out, jobs) {
  if (!VARIANTS[variant] || !out) throw new Error('usage: measure.mjs --variant tools|direct|decision --samples N --out FILE');
  const { natlangRuntime, loadVirtualNatlang } = await import(join(dist, 'index.js'));
  const { callableMeta } = await import(join(dist, 'runtime', 'callable.js'));
  const suite = JSON.parse(readFileSync(join(here, 'suite.json'), 'utf8'));
  const configured = await natlangRuntime({});
  // The variant's settings over the profile's: the model config the runtime was built with is the one calls use.
  Object.assign(configured.runtime.options.model, VARIANTS[variant]);
  // Each finished call is appended to OUT.jsonl as it completes, so a stopped run resumes where it was.
  mkdirSync(dirname(out), { recursive: true });
  const partial = `${out}.jsonl`;
  const results = existsSync(partial) ? readFileSync(partial, 'utf8').split('\n').filter(Boolean).map(line => JSON.parse(line)) : [];
  const done = new Set(results.map(row => `${row.case}/${row.sample}`));
  const started = Date.now();
  try {
    const work = [];
    for (let sample = 0; sample < samples; sample++) for (const [index, item] of suite.cases.entries())
      if (!done.has(`${index}/${sample}`)) work.push({ sample, index, item });
    // --jobs calls at a time (the server's slots); each call's own trace gives its turns and tokens.
    const runOne = async ({ sample, index, item }) => {
      const fn = loadVirtualNatlang({ [item.function]: suite.functions[item.function] }, item.function);
      const params = callableMeta(fn).definition.params;
      const began = Date.now();
      let value, error, mine = [];
      // The task's own trace sink: the root call's events (its turns, tokens and any readout).
      const own = trace => { if (trace.parentCallId === null) mine = trace.events; };
      try {
        value = await configured.runtime.run(() => fn(...params.map(param => item.inputs[param.name])), { trace: own });
      } catch (failure) { error = failure instanceof Error ? failure.message : String(failure); }
      const trace = mine;
      const requests = trace.filter(event => event.kind === 'model_request' && event.phase === 'end');
      const readout = trace.some(event => event.kind === 'decision_readout' && event.phase === 'scored');
      const row = { case: index, sample, function: item.function, expected: item.expected, value: value ?? null, error: error ?? null,
        correct: error === undefined && isDeepStrictEqual(value, item.expected), seconds: (Date.now() - began) / 1000,
        turns: requests.length, readout: readout ? 'decision' : null,
        prompt_tokens: requests.reduce((sum, event) => sum + (event.prompt_tokens ?? 0), 0),
        cached_tokens: requests.reduce((sum, event) => sum + (event.cached_tokens ?? 0), 0),
        completion_tokens: requests.reduce((sum, event) => sum + (event.completion_tokens ?? 0), 0) };
      results.push(row);
      appendFileSync(partial, JSON.stringify(row) + '\n');
      process.stderr.write(`${variant} s${sample} #${index} ${item.function} ${row.correct ? 'ok' : 'WRONG'} ${row.seconds.toFixed(1)}s ` +
        `${row.turns}t ${row.prompt_tokens}p ${row.completion_tokens}c ${row.error ? row.error.slice(0, 120) : JSON.stringify(row.value)}\n`);
    };
    await Promise.all(Array.from({ length: jobs }, async () => { while (work.length) await runOne(work.shift()); }));
  } finally { await configured.close(); }
  results.sort((a, b) => a.sample - b.sample || a.case - b.case);
  const correct = results.filter(row => row.correct).length;
  writeFileSync(out, JSON.stringify({ variant, settings: VARIANTS[variant], model: configured.selection.choice.model,
    profile: configured.selection.name, samples, seconds: (Date.now() - started) / 1000, correct, total: results.length, results }, null, 2) + '\n');
  process.stderr.write(`${variant}: ${correct}/${results.length} correct in ${((Date.now() - started) / 1000).toFixed(0)} s\n`);
}

/** McNemar exact (two-sided binomial on discordant pairs) plus per-variant cost. */
function compare(a, b) {
  const [left, right] = [a, b].map(path => JSON.parse(readFileSync(path, 'utf8')));
  const key = row => `${row.case}/${row.sample}`;
  const rightByKey = new Map(right.results.map(row => [key(row), row]));
  let onlyLeft = 0, onlyRight = 0, both = 0, neither = 0;
  for (const row of left.results) {
    const other = rightByKey.get(key(row));
    if (!other) continue;
    if (row.correct && other.correct) both++; else if (row.correct) onlyLeft++; else if (other.correct) onlyRight++; else neither++;
  }
  const n = onlyLeft + onlyRight, k = Math.min(onlyLeft, onlyRight);
  let tail = 0; for (let i = 0; i <= k; i++) tail += binomial(n, i) / 2 ** n;
  const p = n ? Math.min(1, 2 * tail) : 1;
  const cost = run => {
    const rows = run.results, mean = field => rows.reduce((sum, row) => sum + row[field], 0) / rows.length;
    return { correct: `${run.correct}/${run.total}`, mean_seconds: +mean('seconds').toFixed(2), mean_turns: +mean('turns').toFixed(2),
      mean_prompt_tokens: Math.round(mean('prompt_tokens')), mean_completion_tokens: Math.round(mean('completion_tokens')) };
  };
  console.log(JSON.stringify({ [left.variant]: cost(left), [right.variant]: cost(right),
    pairs: both + neither + n, both, neither, [`only_${left.variant}`]: onlyLeft, [`only_${right.variant}`]: onlyRight, mcnemar_p: +p.toFixed(4) }, null, 2));
}
function binomial(n, k) { let r = 1; for (let i = 1; i <= k; i++) r = r * (n - k + i) / i; return r; }
