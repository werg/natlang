#!/usr/bin/env node
/**
 * A/B replay of one natural-language function: old against new prompt bytes on recorded inputs.
 *
 * The inputs are calls the machine recorded (`natlang traces export --definition NAME`, or a JSONL file of such
 * records). Each variant is a copy of the program folder: a git revision of it (`--old-rev REV`), a prepared folder
 * (`--old-dir DIR`), or the working tree (the default for new). Both variants run the same inputs against one executor
 * (a model profile), `--samples` calls each (default 48, inputs taken round robin). A call is scored by the runtime's own
 * checks (the declared return type, refinements and checkers: a call that ends without a valid value fails) and by an
 * exact comparison of the value with the recorded output. Writes a JSON summary; the row for MODEL_FACING_CHANGES is
 * printed and kept in the summary as `row`.
 *
 *   node ts-host/scripts/ab-replay.mjs --program applications/logs --function investigate/summarize.nl \
 *     --old-rev 12c4501a^ [--new-rev REV | --new-dir DIR] [--records FILE.jsonl] [--samples 48] [--concurrency 8] \
 *     [--profile dgx-qwen] [--label TEXT] --out summary.json
 *
 * Records are filtered to the function's source path (`definition.source` ends with the function's path) and
 * deduplicated by inputs. Calls of the function's own stages (its companion folder) are made by the model as usual.
 */
import { execFileSync, spawnSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync, rmSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve, relative, basename, dirname } from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';

const dist = new URL('../dist/index.js', import.meta.url).href;

/** Canonical JSON: object keys sorted, undefined dropped. */
export function canonical(value) {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value && typeof value === 'object') return `{${Object.keys(value).sort().filter(key => value[key] !== undefined)
    .map(key => `${JSON.stringify(key)}:${canonical(value[key])}`).join(',')}}`;
  return JSON.stringify(value ?? null);
}

/** Recorded calls of the function at `functionPath` (relative to the program), one per distinct input. */
export function selectRecords(records, functionPath, { name } = {}) {
  const wanted = functionPath.replace(/\\/g, '/');
  const seen = new Set(), out = [];
  for (const record of records) {
    const source = record?.definition?.source ?? '';
    if (!(source === wanted || source.endsWith(`/${wanted}`) || wanted.endsWith(`/${source}`) && source.includes('/')) &&
      !(name && record?.definition?.name === name && source === basename(wanted))) continue;
    if (record.audit_of) continue;
    const key = canonical(record.inputs);
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(record);
  }
  return out;
}

/** A copy of `programDir` (relative to the git root `repo`) at revision `rev`. */
export function materializeRevision(repo, programDir, rev, into) {
  const rel = relative(repo, resolve(programDir)).split('\\').join('/');
  mkdirSync(into, { recursive: true });
  const archive = spawnSync('git', ['-C', repo, 'archive', rev, rel], { maxBuffer: 1 << 30 });
  if (archive.status !== 0) throw new Error(`git archive ${rev} ${rel}: ${archive.stderr}`);
  const untar = spawnSync('tar', ['-x', '-C', into], { input: archive.stdout });
  if (untar.status !== 0) throw new Error(`tar: ${untar.stderr}`);
  return join(into, rel);
}

function countingDriver(driver, meter) {
  return Object.assign(async (request, signal, options) => {
    meter.turns++;
    const reply = await driver(request, signal, options);
    const usage = reply?.usage;
    if (usage) { meter.promptTokens += usage.prompt_tokens ?? usage.input ?? 0; meter.completionTokens += usage.completion_tokens ?? usage.output ?? 0; }
    return reply;
  }, driver.decide ? { decide: driver.decide } : {}, driver.contextWindow ? { contextWindow: driver.contextWindow } : {});
}

async function pool(items, limit, work) {
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (next < items.length) { const index = next++; await work(items[index], index); }
  }));
}

/** Run one variant: `samples` calls of the function in `root` over `inputs` (round robin). */
async function runVariant({ lib, root, functionPath, records, samples, concurrency, driver, runtimeOptions }) {
  const { createNatlangRuntime, loadNatlang } = lib;
  const fn = loadNatlang(join(root, functionPath), root);
  const params = fn[Symbol.for('natlang.callable')].definition.params;
  const rows = [];
  const jobs = Array.from({ length: samples }, (_, index) => records[index % records.length]);
  await pool(jobs, concurrency, async (record, index) => {
    const meter = { turns: 0, promptTokens: 0, completionTokens: 0 };
    const runtime = createNatlangRuntime({ calls: false, ...(runtimeOptions ?? {}), model: { ...(runtimeOptions?.model ?? {}), driver: countingDriver(driver, meter) } });
    const args = params.map(param => record.inputs?.[param.name]);
    while (args.length && args[args.length - 1] === undefined) args.pop();
    const started = performance.now();
    let value, error;
    try { value = await runtime.run(() => fn(...args)); } catch (caught) { error = caught instanceof Error ? caught.message : String(caught); }
    finally { runtime.close?.(); }
    rows.push({ index, call_id: record.call_id, ok: error === undefined, exact: error === undefined && canonical(value) === canonical(record.output),
      error: error?.slice(0, 500), value, ms: Math.round(performance.now() - started), ...meter });
  });
  rows.sort((a, b) => a.index - b.index);
  const n = rows.length, sum = key => rows.reduce((total, row) => total + (typeof row[key] === 'boolean' ? Number(row[key]) : row[key]), 0);
  const errors = {};
  for (const row of rows) if (row.error) { const kind = row.error.replace(/["'`][^"'`]*["'`]/g, '…').slice(0, 120); errors[kind] = (errors[kind] ?? 0) + 1; }
  return { n, ok: sum('ok'), exact: sum('exact'), mean_turns: n ? +(sum('turns') / n).toFixed(2) : 0,
    mean_prompt_tokens: n ? Math.round(sum('promptTokens') / n) : 0, mean_completion_tokens: n ? Math.round(sum('completionTokens') / n) : 0,
    mean_ms: n ? Math.round(sum('ms') / n) : 0, errors, rows };
}

const FITS = 'ab-replay: the inputs reached the model';
/** Recorded inputs the function at `root` still accepts: each is invoked once with a model that stops at its first turn. */
export async function fittingRecords(lib, root, functionPath, records) {
  const fn = lib.loadNatlang(join(root, functionPath), root);
  const params = fn[Symbol.for('natlang.callable')].definition.params;
  const fit = [], misfit = {};
  for (const record of records) {
    const runtime = lib.createNatlangRuntime({ calls: false, model: { driver: () => { throw new Error(FITS); } } });
    const args = params.map(param => record.inputs?.[param.name]);
    while (args.length && args[args.length - 1] === undefined) args.pop();
    let message = '';
    try { await runtime.run(() => fn(...args)); } catch (caught) { message = caught instanceof Error ? caught.message : String(caught); }
    finally { runtime.close?.(); }
    if (message.includes(FITS)) fit.push(record);
    else { const kind = message.slice(0, 160) || 'returned without a model'; misfit[kind] = (misfit[kind] ?? 0) + 1; }
  }
  return { fit, misfit };
}

/**
 * The A/B replay. `options.driver` is the executor (a model driver); `options.oldRoot`/`options.newRoot` are the two
 * program folders; `options.records` the recorded calls (already selected or raw).
 */
export async function abReplay(options) {
  const lib = await import(options.dist ?? dist);
  const selected = selectRecords(options.records, options.functionPath, { name: options.name });
  if (!selected.length) throw new Error(`no recorded calls of ${options.functionPath}`);
  // Only inputs both variants accept (a recorded call from an older signature is left out and counted).
  const oldFit = await fittingRecords(lib, options.oldRoot, options.functionPath, selected);
  const { fit: records, misfit } = await fittingRecords(lib, options.newRoot, options.functionPath, oldFit.fit);
  for (const [kind, count] of Object.entries(oldFit.misfit)) misfit[`old: ${kind}`] = count;
  if (!records.length) throw new Error(`none of the ${selected.length} recorded inputs of ${options.functionPath} fits both variants: ${JSON.stringify(misfit)}`);
  const samples = options.samples ?? 48, concurrency = options.concurrency ?? 8;
  const common = { lib, functionPath: options.functionPath, records, samples, concurrency, driver: options.driver, runtimeOptions: options.runtime };
  const [oldVariant, newVariant] = await Promise.all([
    runVariant({ ...common, root: options.oldRoot }), runVariant({ ...common, root: options.newRoot })]);
  const label = options.label ?? options.functionPath;
  const brief = v => `${v.ok}/${v.n} valid, ${v.exact}/${v.n} equal to recorded, ${v.mean_turns} turns`;
  return { version: 'natlang.ab-replay/1', label, function: options.functionPath, old: options.oldLabel, new: options.newLabel,
    executor: options.executor, distinct_inputs: records.length, recorded_inputs: selected.length, misfit, samples, created_at: new Date().toISOString(),
    row: `${label}: old ${brief(oldVariant)}; new ${brief(newVariant)} (n=${samples}/variant, ${records.length} recorded inputs, ${options.executor ?? 'executor'})`,
    variants: { old: oldVariant, new: newVariant } };
}

function exportRecords(name, limit) {
  const natlang = process.env.NATLANG_BIN ?? 'natlang';
  const out = execFileSync(natlang, ['traces', 'export', '--definition', name, '--limit', String(limit)], { maxBuffer: 1 << 30, encoding: 'utf8' });
  return out.split('\n').filter(Boolean).map(line => JSON.parse(line));
}

async function main(argv = process.argv.slice(2)) {
  const values = new Map();
  for (let i = 0; i < argv.length; i++) if (argv[i].startsWith('--')) values.set(argv[i], argv[i + 1]?.startsWith('--') || argv[i + 1] === undefined ? 'true' : argv[++i]);
  if (!values.has('--program') || !values.has('--function') || !(values.has('--old-rev') || values.has('--old-dir')) || !values.has('--out')) {
    console.log(readFileSync(fileURLToPath(import.meta.url), 'utf8').split('*/')[0].replace(/^[#/* !]+/gm, '').trim());
    return 2;
  }
  const program = resolve(values.get('--program')), functionPath = values.get('--function');
  const repo = execFileSync('git', ['-C', program, 'rev-parse', '--show-toplevel'], { encoding: 'utf8' }).trim();
  const work = mkdtempSync(join(values.get('--work') ?? tmpdir(), 'ab-replay-'));
  const variant = (which) => values.has(`--${which}-dir`) ? { root: resolve(values.get(`--${which}-dir`)), label: values.get(`--${which}-dir`) } :
    values.has(`--${which}-rev`) ? { root: materializeRevision(repo, program, values.get(`--${which}-rev`), join(work, which)), label: values.get(`--${which}-rev`) } :
      { root: program, label: 'working tree' };
  const oldV = variant('old'), newV = variant('new');
  const name = basename(functionPath, '.nl');
  const records = values.has('--records') ? readFileSync(values.get('--records'), 'utf8').split('\n').filter(Boolean).map(line => JSON.parse(line)) :
    exportRecords(values.get('--definition') ?? name, Number(values.get('--limit') ?? 5000));
  const lib = await import(dist);
  const config = lib.loadModelConfiguration(values.get('--profile'), {});
  const session = lib.createResolvedModelSession(config.choice, process.env, process.stderr, {});
  const driver = Object.assign((request, signal, options) => session.turn(request, signal, options),
    { decide: (request, signal) => session.decide(request, signal), contextWindow: () => session.contextWindow() });
  const { seed, ...settings } = config.choice.runtime ?? {};
  await session.prepare();
  try {
    const summary = await abReplay({ records, functionPath, name, oldRoot: oldV.root, newRoot: newV.root, oldLabel: oldV.label, newLabel: newV.label,
      samples: Number(values.get('--samples') ?? 48), concurrency: Number(values.get('--concurrency') ?? 8), label: values.get('--label'),
      executor: `${config.choice.model ?? ''} @ ${config.choice.endpoint ?? values.get('--profile') ?? 'default profile'}`.trim(), driver,
      runtime: { model: { ...settings, promptRendering: config.choice.promptRendering ?? 'full' }, ...(seed ? { seed } : {}) } });
    mkdirSync(dirname(resolve(values.get('--out'))), { recursive: true });
    writeFileSync(values.get('--out'), `${JSON.stringify(summary, null, 1)}\n`);
    console.log(summary.row);
  } finally { await session.close(); if (!values.has('--keep')) rmSync(work, { recursive: true, force: true }); }
  return 0;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) main().then(code => process.exit(code), error => { console.error(error.stack ?? error.message); process.exit(1); });
