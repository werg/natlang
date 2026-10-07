/**
 * `natlang run applications/compilers -- compile PROGRAM.c|PROGRAM.py [-O1|-O2|-O3] [--input FILE]... [--out DIR]
 *   [--no-backend] [--pure]`: compile with the natural-language compiler, checking every stage on the inputs.
 * `natlang run applications/compilers -- bench [NAME...] [--out DIR] [--pure]`: the benchmarks in bench/, with
 *   timings against gcc -O0/-O2 (C) or CPython (Python).
 * Without --pure the host driver (index.ts) runs the stages and checks each one; with --pure the whole pipeline is
 * compiler.nl, a pass manager in natural language, and the host only checks its final program against gcc/CPython.
 */
import { mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { basename, extname, join, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import type { TargetContext } from '@natlang/node';
import { NatlangRuntime, type ModelDriver } from '@natlang/node';
import { compile, limiter, type Compilation, type Language, type StageRecord } from './index.js';
import compiler from './compiler.nl';
import { gccReference, toolchain, toolchainAvailable, toolchainDeclaration } from './toolchain.js';
import type { Level } from './types.js';

const benchDirectory = fileURLToPath(new URL('../../compilers/bench', import.meta.url));

function option(args: string[], name: string): string[] {
  return args.flatMap((arg, i) => arg === name && args[i + 1] !== undefined ? [args[i + 1]!] : []);
}
const best = async (times: number, work: () => Promise<{ ms: number, ok: boolean }>) => {
  let fastest = Infinity;
  for (let i = 0; i < times; i++) { const result = await work(); if (!result.ok) return null; fastest = Math.min(fastest, result.ms); }
  return fastest;
};

export default async function main(context: TargetContext): Promise<number> {
  const [command, ...args] = context.args;
  if (!toolchainAvailable()) { context.io.error.write('the toolchain needs llvmlite: set NATLANG_LLVM_PYTHON\n'); return 2; }
  const level = (args.find(arg => /^-O[123]$/.test(arg))?.slice(1) ?? 'O2') as Level;
  const run = <T>(fn: () => Promise<T>) => context.runtime.run(fn, { services: { toolchain },
    serviceDeclarations: { toolchain: toolchainDeclaration } });
  // The pure pipeline's stages are its callable folder; it runs them and may not rewrite them. It decides itself
  // which calls run at once, so the model requests in flight are limited at the driver.
  const configured = context.runtime.options.model;
  const driver = typeof configured === 'function' ? configured : configured?.driver ?? context.model;
  const inFlight = limiter(4);
  const limited: ModelDriver = Object.assign((request: Parameters<ModelDriver>[0], signal?: AbortSignal) => inFlight(async () => driver(request, signal)), driver);
  const fixed = new NatlangRuntime({ ...context.runtime.options, codeEdits: 'deny',
    model: typeof configured === 'object' && configured ? { ...configured, driver: limited } : limited });
  /** compiler.nl end to end; its program is then checked against the reference outputs here. */
  const pure = async (source: string, language: Language, inputs: string[], expected: string[]): Promise<Compilation> => {
    const started = performance.now();
    const result = await fixed.run(() => compiler(source, language, level, inputs), { services: { toolchain },
      serviceDeclarations: { toolchain: toolchainDeclaration } });
    const runs = result.assembly ? await Promise.all(inputs.map(input => toolchain.runAssembly(result.assembly, input, 120_000))) : [];
    const wrong = runs.findIndex((run, i) => !run.ok || run.stdout !== expected[i]);
    const problem = !result.assembly ? 'no assembly' : wrong >= 0 ? `input ${wrong}: ${runs[wrong]!.error ?? `printed ${JSON.stringify(runs[wrong]!.stdout.slice(0, 200))}`}` : undefined;
    for (const line of result.log) context.io.error.write(`     ${line}\n`);
    return { ok: !result.diagnostics.length && !problem, diagnostics: [...result.diagnostics, ...problem ? [problem] : []],
      records: [{ function: '*', stage: 'compiler.nl', accepted: !problem, attempts: 1, ms: performance.now() - started, problem }],
      ir: result.ir, assembly: result.assembly || undefined, log: result.log };
  };
  const reference = async (source: string, language: Language, input: string, file?: string) => language === 'c'
    ? (await gccReference(source, '-O0', input)).stdout
    : spawnSync('python3', file ? [file] : ['-c', source], { input, encoding: 'utf8' }).stdout;
  const log = (record: StageRecord) => context.io.error.write(`${record.accepted ? 'ok  ' : 'NO  '} ${record.function} ${record.stage}` +
    ` (${record.attempts} attempt${record.attempts > 1 ? 's' : ''}, ${(record.ms / 1000).toFixed(0)} s)${record.problem ? `: ${record.problem.slice(0, 200)}` : ''}\n`);
  const out = resolve(context.workspace, option(args, '--out')[0] ?? 'natlang-cc-out');
  mkdirSync(out, { recursive: true });
  const save = (name: string, result: Compilation) => {
    if (result.unoptimized) writeFileSync(join(out, `${name}.O0.ll`), result.unoptimized);
    if (result.ir) writeFileSync(join(out, `${name}.ll`), result.ir);
    if (result.assembly) writeFileSync(join(out, `${name}.s`), result.assembly);
    writeFileSync(join(out, `${name}.report.json`), JSON.stringify({ ok: result.ok, diagnostics: result.diagnostics, records: result.records, log: result.log }, null, 2));
  };

  if (command === 'compile') {
    const path = args.find(arg => /\.(c|py)$/.test(arg));
    if (!path) { context.io.error.write('usage: compile PROGRAM.c|PROGRAM.py [-O2] [--input FILE]... [--out DIR]\n'); return 2; }
    const source = readFileSync(resolve(context.workspace, path), 'utf8');
    const inputs = option(args, '--input').map(file => readFileSync(resolve(context.workspace, file), 'utf8'));
    const language: Language = extname(path) === '.py' ? 'python' : 'c';
    const runInputs = inputs.length ? inputs : [''];
    const result = args.includes('--pure')
      ? await pure(source, language, runInputs, await Promise.all(runInputs.map(input => reference(source, language, input))))
      : await compile(source, { language, level, inputs: runInputs, run, backend: !args.includes('--no-backend'), onRecord: log });
    save(basename(path, extname(path)), result);
    context.io.output.write(`${result.ok ? 'compiled' : 'failed'}: ${result.diagnostics.join('; ') || 'all stages checked'} (${out})\n`);
    return result.ok ? 0 : 1;
  }

  if (command === 'bench') {
    const names = args.filter(arg => !arg.startsWith('-') && !option(args, '--out').includes(arg));
    const programs = ['c', 'python'].flatMap(language => readdirSync(join(benchDirectory, language))
      .filter(file => /\.(c|py)$/.test(file) && (!names.length || names.includes(basename(file, extname(file)))))
      .map(file => ({ language: language as Language, file: join(benchDirectory, language, file) })));
    const limit = limiter(4);
    const rows = await Promise.all(programs.map(async ({ language, file }) => {
      const name = basename(file, extname(file));
      const source = readFileSync(file, 'utf8'), input = readFileSync(file.replace(/\.(c|py)$/, '.in'), 'utf8');
      const expected = await reference(source, language, input, file);
      const result = args.includes('--pure') ? await pure(source, language, [input], [expected])
        : await compile(source, { language, level, inputs: [input], expected: [expected], run, limit, backend: true, onRecord: log });
      save(name, result);
      const natlang = result.ok ? await best(3, () => toolchain.runAssembly(result.assembly!, input, 120_000)) : null;
      const baselines = language === 'c'
        ? { 'gcc -O0': await best(3, () => gccReference(source, '-O0', input, 120_000)), 'gcc -O2': await best(3, () => gccReference(source, '-O2', input, 120_000)) }
        : { cpython: await best(1, async () => { const started = performance.now(); const r = spawnSync('python3', [file], { input }); return { ok: r.status === 0, ms: performance.now() - started }; }) };
      const stages = result.records.reduce((counts, r) => ({ ...counts, [r.accepted ? 'accepted' : 'rejected']: (counts[r.accepted ? 'accepted' : 'rejected'] ?? 0) + 1 }), {} as Record<string, number>);
      return { name, language, ok: result.ok, natlangMs: natlang, ...baselines, stages, diagnostics: result.diagnostics };
    }));
    writeFileSync(join(out, 'bench.json'), JSON.stringify(rows, null, 2));
    for (const row of rows) context.io.output.write(`${JSON.stringify(row)}\n`);
    return rows.every(row => row.ok) ? 0 : 1;
  }

  context.io.error.write('usage: compile PROGRAM [-O1|-O2|-O3] [--input FILE]... [--out DIR] [--no-backend] | bench [NAME...] [--out DIR]\n');
  return 2;
}
