/**
 * natlang optimizing compilers: C, typed-Python and Rust front ends, a shared LLVM-IR middle end, an AArch64 back end.
 * Every stage is a natural-language function, and compiler.nl is the compiler itself, with its pass manager in natural
 * language: the order of stages, what is checked after each, the retry with the problem, the fallbacks. This module
 * holds no compilation policy. It runs compiler.nl and verifies what comes back with the toolchain service (LLVM's
 * verifier, the IR interpreter, the assembler and runner): the IR must verify, and the IR and the assembly must print
 * what the program is expected to print on every input.
 */
import compiler from './compiler.nl';
import { toolchain } from './toolchain.js';
import type { Level } from './types.js';

export type Language = 'c' | 'python' | 'rust';
export type StageRecord = { function: string, stage: string, accepted: boolean, attempts: number, ms: number,
  problem?: string, size?: number };
export type CompileOptions = {
  language: Language,
  level?: Level,
  /** stdin for each test run; the program must behave the same on each. */
  inputs?: string[],
  /** Expected stdout per input (benchmarks); otherwise the optimized IR's output is the reference for the assembly. */
  expected?: string[],
  /** Runs natural-language work in a task that provides the toolchain service. */
  run: <T>(fn: () => Promise<T>) => Promise<T>,
  /** Whether the assembly is verified too (default true); the IR always is. */
  backend?: boolean,
  onRecord?: (record: StageRecord) => void,
};
export type Compilation = { ok: boolean, diagnostics: string[], records: StageRecord[], unoptimized?: string,
  ir?: string, assembly?: string, problem?: string,
  /** The pass manager's own account of its stages (compiler.nl). */
  log?: string[] };

const quote = (text: string) => JSON.stringify(text.length > 400 ? `${text.slice(0, 400)}…` : text);

/** One module: drop declarations of symbols the module defines, and repeated declarations. */
export function normalize(text: string): string {
  const defined = new Set([...text.matchAll(/^define[^@\n]*@([\w.$-]+)\s*\(/gm)].map(m => m[1]));
  const declared = new Set<string>();
  return text.split('\n').filter(line => {
    const symbol = /^declare[^@]*@([\w.$-]+)\s*\(/.exec(line)?.[1];
    if (!symbol) return true;
    if (defined.has(symbol) || declared.has(symbol)) return false;
    declared.add(symbol);
    return true;
  }).join('\n');
}

export type Limiter = <T>(work: () => Promise<T>) => Promise<T>;
export function limiter(size: number): Limiter {
  let active = 0;
  const waiting: (() => void)[] = [];
  return async <T>(work: () => Promise<T>): Promise<T> => {
    if (active >= size) await new Promise<void>(resolve => waiting.push(resolve));
    active++;
    try { return await work(); } finally { active--; waiting.shift()?.(); }
  };
}

type Ran = { stdout: string, exitCode: number | null, error?: string };

/** Run compiler.nl, then verify its module and program with the toolchain. */
export async function compile(source: string, options: CompileOptions): Promise<Compilation> {
  const { language, level = 'O2', inputs = [''], run } = options;
  const records: StageRecord[] = [];
  const record = (stage: string, started: number, problem?: string | null) => {
    const entry: StageRecord = { function: '*', stage, accepted: !problem, attempts: 1, ms: Math.round(performance.now() - started),
      ...problem ? { problem } : {} };
    records.push(entry); options.onRecord?.(entry);
    return problem ?? null;
  };

  const compiledAt = performance.now();
  let result;
  try { result = await run(() => compiler(source, language, level, inputs)); } catch (error) {
    const problem = `compilation failed: ${String((error as Error)?.message ?? error).slice(0, 600)}`;
    record('compiler.nl', compiledAt, problem);
    return { ok: false, diagnostics: [problem], records };
  }
  const log = result.log ?? [];
  record('compiler.nl', compiledAt, result.diagnostics.length ? result.diagnostics.join('; ') : null);
  if (result.diagnostics.length) return { ok: false, diagnostics: result.diagnostics, records, log };

  const ir = normalize(result.ir);
  const fail = (diagnostic: string): Compilation => ({ ok: false, diagnostics: [diagnostic], records, ir, log,
    ...result.assembly ? { assembly: result.assembly } : {}, problem: diagnostic });

  const verifyAt = performance.now();
  const verdict = await toolchain.verify(ir);
  if (record('verify', verifyAt, verdict.ok ? null : `the verifier rejected the module: ${verdict.error}`)) {
    return fail(`the verifier rejected the module: ${verdict.error}`);
  }

  /** The first input on which a program does not print the reference output, described; null when there is none. */
  const differs = (ran: Ran[], reference: { stdout: string, exitCode?: number | null }[]) => {
    for (const [i, input] of inputs.entries()) {
      const got = ran[i]!, wanted = reference[i];
      if (!wanted) {
        if (got.error && got.exitCode === null) return `on input ${quote(input)} the program failed: ${got.error}`;
        continue;
      }
      if (got.stdout !== wanted.stdout || (wanted.exitCode !== undefined && got.exitCode !== wanted.exitCode)) {
        return `on input ${quote(input)} the program printed ${quote(got.stdout)} (exit ${got.exitCode}${got.error ? `, ${got.error}` : ''}) instead of ${quote(wanted.stdout)}`;
      }
    }
    return null;
  };
  const expected = options.expected?.map(stdout => ({ stdout }));
  const irAt = performance.now();
  const irRuns: Ran[] = await Promise.all(inputs.map(input => toolchain.runIR(ir, input)));
  const irProblem = record('run ir', irAt, differs(irRuns, expected ?? []));
  if (irProblem) return fail(irProblem);
  if (options.backend === false) return { ok: true, diagnostics: [], records, ir, log };
  if (!result.assembly) return fail('the back end did not produce a program');

  const assemblyAt = performance.now();
  const reference = expected ?? irRuns.map(r => ({ stdout: r.stdout, exitCode: r.exitCode }));
  const asmRuns: Ran[] = await Promise.all(inputs.map(input => toolchain.runAssembly(result.assembly, input)));
  const asmProblem = record('run assembly', assemblyAt, differs(asmRuns, reference));
  if (asmProblem) return fail(`the generated program misbehaves: ${asmProblem}`);
  return { ok: true, diagnostics: [], records, ir, assembly: result.assembly, log };
}
