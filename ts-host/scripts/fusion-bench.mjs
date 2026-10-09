#!/usr/bin/env node
/**
 * Fusion benchmark (plans/FUSED_PIPELINES.md): run one application's stage chain over a set of cases twice, as text
 * (fusion off) and fused (fusion on), against the same model, and report tokens, latency and output agreement, with
 * the per-application gate.
 *
 *   node scripts/fusion-bench.mjs --app ../applications/compilers --entry compiler.nl --cases cases.jsonl \
 *     --neuralese-endpoint http://127.0.0.1:8090 --model NAME --certificate cert.json --weights SHA256 \
 *     [--planner crisp|nl|shadow] [--mode both|text|fused] [--repeat 1] [--out report.json] [--dry-run]
 *
 * cases.jsonl: one JSON object per line, `{ "id": "c1", "args": [ ...arguments of the entry function... ], "expected": ... }`.
 * `expected` is optional; with it the report adds accuracy of each mode. The entry function runs with the effects of
 * the application, so point it at a sandboxed copy.
 *
 * Needs a qualified Neuralese model: a server speaking the certificate's dialect (reference server, or the llama.cpp
 * fork) and the certificate the training pipeline issued for those exact weights. Without them the fused run falls
 * back to text, the report says so (`engaged: false`) and the gate fails. `--dry-run` prints the plan and the
 * configuration and calls no model. Nothing here loads a model itself.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve, dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const dist = new URL('../dist/index.js', import.meta.url).href;
const fusionDist = new URL('../dist/fusion/index.js', import.meta.url).href;
const canonical = value => JSON.stringify(value, (_key, item) => item && typeof item === 'object' && !Array.isArray(item) ?
  Object.fromEntries(Object.entries(item).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)) : item);
const sum = values => values.reduce((total, value) => total + value, 0);
const percentile = (values, p) => values.length ? [...values].sort((a, b) => a - b)[Math.min(values.length - 1, Math.floor(p * values.length))] : null;

/** The gate for turning fusion on for an application (plans/FUSED_PIPELINES.md, "Gate"). */
export const DEFAULT_GATE = { minAgreement: 0.98, maxTokenRatio: 1.0, maxLatencyRatio: 1.0, maxFallbacks: 0, maxUnconsumed: 0 };

function countingDriver(driver, meter) {
  const wrapped = async (request, signal) => {
    const started = performance.now();
    const turn = await driver(request, signal);
    meter.turns++;
    meter.promptTokens += turn.prompt_tokens ?? 0;
    meter.completionTokens += turn.completion_tokens ?? 0;
    meter.modelMs += performance.now() - started;
    return turn;
  };
  for (const key of Object.keys(driver)) wrapped[key] = driver[key];
  if (driver.neuralese) wrapped.neuralese = true;
  return wrapped;
}

/**
 * Run the benchmark. `options.driver` (a model driver, Neuralese-capable for fused runs), `options.neuralese`
 * ({ store, port }), `options.createDriver(mode)` (a fresh driver per mode) and `options.runtime` (extra runtime options)
 * are injectable; the CLI builds them from flags.
 */
export async function runBench(options) {
  const lib = await import(options.dist ?? dist);
  const fusion = await import(options.fusionDist ?? fusionDist);
  const { createNatlangRuntime, loadNatlang, nodeSourceFiles } = lib;
  const { fusionFacts, crispPlan, fusedEdges, fusionReport, parseFusionSettings, readFusionCertificate } = fusion;
  const root = resolve(options.app);
  const settings = { ...parseFusionSettings(undefined), planner: options.planner ?? 'crisp',
    ...(options.certificate ? { certificate: options.certificate } : {}), ...(options.weights ? { weights: options.weights } : {}) };
  const facts = fusionFacts(root, nodeSourceFiles(root));
  const probe = createNatlangRuntime({ calls: false, ...(options.runtime ?? {}) });
  const plan = settings.planner === 'crisp' ? crispPlan(facts) : await fusion.planFusionWith(probe, facts, settings.planner);
  const edges = fusedEdges(facts, plan);
  const report = { schema: 'natlang.fusion-bench/1', app: root, entry: options.entry, plan: fusionReport(facts, plan, settings).counts,
    edges: edges.map(edge => edge.id), cases: options.cases.length, modes: {}, gate: { ...DEFAULT_GATE, ...(options.gate ?? {}) } };
  if (options.dryRun) return { ...report, dryRun: true };
  const entry = loadNatlang(join(root, options.entry), root);
  const modes = options.mode === 'both' || !options.mode ? ['text', 'fused'] : [options.mode];
  const outputs = {};
  for (const mode of modes) {
    const meter = { turns: 0, promptTokens: 0, completionTokens: 0, modelMs: 0 };
    const traces = [];
    const certificate = options.certificateValue ?? readFusionCertificate(root, settings);
    const runtime = createNatlangRuntime({ calls: false, ...(options.runtime ?? {}), model: countingDriver(options.createDriver ? options.createDriver(mode) : options.driver, meter),
      trace: trace => traces.push(trace),
      ...(options.neuralese ? { neuralese: options.neuralese } : {}),
      ...(mode === 'fused' ? { fusion: { mode: 'on', edges, certificate, weights: settings.weights, ...(options.emulation ? { emulation: options.emulation } : {}) } } : {}) });
    const results = [];
    for (const item of options.cases) {
      const before = { ...meter };
      const started = performance.now();
      let value, error;
      try { value = await runtime.run(() => entry(...item.args)); } catch (caught) { error = caught instanceof Error ? caught.message : String(caught); }
      results.push({ id: item.id, ms: performance.now() - started, error, value,
        tokens: meter.promptTokens + meter.completionTokens - before.promptTokens - before.completionTokens, turns: meter.turns - before.turns });
    }
    runtime.close();
    const events = traces.flatMap(trace => trace.events.filter(event => typeof event.kind === 'string' && event.kind.startsWith('fusion')));
    outputs[mode] = results;
    report.modes[mode] = {
      tokens: { prompt: meter.promptTokens, completion: meter.completionTokens, total: meter.promptTokens + meter.completionTokens },
      turns: meter.turns, latencyMs: { total: sum(results.map(item => item.ms)), p50: percentile(results.map(item => item.ms), 0.5), p95: percentile(results.map(item => item.ms), 0.95) },
      errors: results.filter(item => item.error).length,
      ...(options.cases.some(item => 'expected' in item) ? { accuracy: results.filter((item, index) => 'expected' in options.cases[index] &&
        canonical(item.value) === canonical(options.cases[index].expected)).length / options.cases.filter(item => 'expected' in item).length } : {}),
      ...(mode === 'fused' ? { fusion: {
        fusedEdges: events.filter(event => event.kind === 'fusion_edge' && event.status === 'fused').length,
        consumed: events.filter(event => event.kind === 'fusion_edge' && event.status === 'consumed').length,
        fallbacks: events.filter(event => event.kind === 'fusion_fallback').map(event => event.reason),
        emulation: events.some(event => event.emulation === true) } } : {}) };
  }
  if (outputs.text && outputs.fused) {
    const same = outputs.text.filter((item, index) => !item.error && !outputs.fused[index].error && canonical(item.value) === canonical(outputs.fused[index].value)).length;
    const text = report.modes.text, fused = report.modes.fused;
    const engaged = fused.fusion.fusedEdges > 0;
    report.comparison = { agreement: options.cases.length ? same / options.cases.length : null, engaged,
      tokenRatio: text.tokens.total ? fused.tokens.total / text.tokens.total : null,
      latencyRatio: text.latencyMs.total ? fused.latencyMs.total / text.latencyMs.total : null };
    const gate = report.gate, c = report.comparison;
    report.gate.result = {
      engaged, agreement: c.agreement !== null && c.agreement >= gate.minAgreement,
      tokens: c.tokenRatio !== null && c.tokenRatio <= gate.maxTokenRatio, latency: c.latencyRatio !== null && c.latencyRatio <= gate.maxLatencyRatio,
      fallbacks: fused.fusion.fallbacks.length <= gate.maxFallbacks, unconsumed: fused.fusion.fusedEdges - fused.fusion.consumed <= gate.maxUnconsumed,
      emulation: !fused.fusion.emulation };
    report.gate.pass = Object.values(report.gate.result).every(Boolean);
  }
  return report;
}

function parseArguments(argv) {
  const values = new Map(), flags = new Set();
  for (let index = 0; index < argv.length; index++) {
    const item = argv[index];
    if (!item.startsWith('--')) throw new Error(`unexpected argument ${item}`);
    if (['--dry-run', '--help'].includes(item)) flags.add(item);
    else values.set(item, argv[++index]);
  }
  return { values, flags };
}

async function main() {
  const { values, flags } = parseArguments(process.argv.slice(2));
  if (flags.has('--help') || !values.get('--app') || !values.get('--entry') || !values.get('--cases')) {
    console.log(readFileSync(fileURLToPath(import.meta.url), 'utf8').split('*/')[0].replace(/^[#/* ]+/gm, '').trim());
    return flags.has('--help') ? 0 : 2;
  }
  const cases = readFileSync(resolve(values.get('--cases')), 'utf8').split('\n').filter(Boolean).map((line, index) => ({ id: `case${index}`, ...JSON.parse(line) }));
  const lib = await import(dist);
  let driver, neuralese;
  if (!flags.has('--dry-run')) {
    const endpoint = values.get('--neuralese-endpoint');
    if (!endpoint) throw new Error('--neuralese-endpoint is required: the text baseline and the fused run use the same Neuralese-capable server');
    const info = await (await lib.fetchModel(`${endpoint.replace(/\/$/, '')}/v1/neuralese/info`)).json();
    const dialect = values.get('--dialect') ?? info.dialects?.[0] ?? info.dialect;
    const width = Number(values.get('--width') ?? info.width);
    const { MemoryNeuraleseStore } = lib, { StandInNeuralesePort, hashingEmbedder } = await import(new URL('../dist/native/neuralese-store.js', import.meta.url).href);
    const store = new MemoryNeuraleseStore();
    driver = (await import(new URL('../dist/model/index.js', import.meta.url).href)).neuraleseServerModelTurn({ endpoint, model: values.get('--model'), store, headers: {} });
    neuralese = { store, port: new StandInNeuralesePort(store, hashingEmbedder(width), width, dialect) };
  }
  const report = await runBench({ app: values.get('--app'), entry: values.get('--entry'), cases, driver, neuralese,
    planner: values.get('--planner'), certificate: values.get('--certificate'), weights: values.get('--weights'),
    mode: values.get('--mode'), dryRun: flags.has('--dry-run') });
  if (values.get('--out')) writeFileSync(resolve(values.get('--out')), JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify(report, null, 2));
  return report.gate?.pass === false ? 1 : 0;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) main().then(code => process.exit(code), error => { console.error(error.message); process.exit(1); });
