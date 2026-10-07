/**
 * natlang optimizing compilers: C, typed-Python and Rust front ends, a shared LLVM-IR middle end, an AArch64 back end.
 * Every stage is a natural-language function; compiler.nl is the compiler itself, with its pass manager in natural
 * language. This is the checked driver: it runs the same stages and checks each result. A stage's output must verify
 * and make the program behave as before on the given inputs (translation validation by testing), or it is sent back
 * once with the reason and otherwise rejected, keeping the previous version.
 */
import compiler from './compiler.nl';
import { llvmAssembly, toolchain } from './toolchain.js';
import type { Flow, Level, ModuleFrame, Pass } from './types.js';

export type Language = 'c' | 'python' | 'rust';
export type StageRecord = { function: string, stage: string, accepted: boolean, attempts: number, ms: number,
  problem?: string, size?: number };
export type CompileOptions = {
  language: Language,
  level?: Level,
  /** stdin for each test run; the program must behave the same on each after every stage. */
  inputs?: string[],
  /** Expected stdout per input (benchmarks); otherwise the front end's output is the reference. */
  expected?: string[],
  /** Runs natural-language work in a task that provides the toolchain service. */
  run: <T>(fn: () => Promise<T>) => Promise<T>,
  /** Natural-language calls in flight at once (default 4), or a limiter shared with other compilations. */
  concurrency?: number,
  limit?: Limiter,
  backend?: boolean,
  onRecord?: (record: StageRecord) => void,
};
export type Compilation = { ok: boolean, diagnostics: string[], records: StageRecord[], unoptimized?: string,
  ir?: string, assembly?: string, problem?: string,
  /** The pure pipeline's own account of its stages (compiler.nl). */
  log?: string[] };

// The stages are compiler.nl's callable folder; this driver calls them directly and checks each result.
/** Each language's front end: parser, semantic analysis, module layout, and IR generation for one function. */
const FRONT_ENDS = { c: compiler.c, python: compiler.python, rust: compiler.rust };
const runtimeLibrary = compiler.runtime;
const { plan, flow, mem2reg, simplify, gvn, licm, loops, inline, dce } = compiler.opt;
const { data, select, liveness, allocate, frame: frameLowering, emit, peephole } = compiler.aarch64;

/** Passes that take the function's control flow (opt.flow), and those that do not. */
const FLOW_PASSES: Record<'mem2reg' | 'gvn' | 'licm' | 'loops', (fn: string, context: string, flow: Flow, problem?: string) => Promise<string>> =
  { mem2reg, gvn, licm, loops };
const PLAIN_PASSES: Record<'simplify' | 'dce', (fn: string, context: string, problem?: string) => Promise<string>> = { simplify, dce };

/** An answer without markdown fences. */
const clean = (text: string) => text.trim().replace(/^```[a-z]*\n/i, '').replace(/\n?```$/, '').trim();
const nameOf = (definition: string) => /define[^@]*@([\w.$-]+)\s*\(/.exec(definition)?.[1];
const declaration = (signature: string) => signature.replace(/^define\s+/, 'declare ')
  .replace(/\b(internal|private|dso_local|hidden|protected|linkonce_odr|weak)\s+/g, '')
  .replace(/\s%[\w.$-]+(?=\s*[,)])/g, '');
const calls = (definition: string) => new Set([...definition.matchAll(/call[^@\n]*@([\w.$-]+)\(/g)].map(m => m[1]!));
const instructions = (text: string) => text.split('\n').filter(line => /^\s+[a-z%]/.test(line) && !/^\s*[.;]/.test(line)).length;
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

/** LLVM's assembly split into its functions and everything else (data, directives): the validation scaffold. */
export function splitAssembly(text: string): { functions: Map<string, string>, rest: string } {
  const lines = text.split('\n'), functions = new Map<string, string>(), rest: string[] = [];
  for (let i = 0; i < lines.length; i++) {
    const type = /^\s*\.type\s+([\w.$]+),\s*@function/.exec(lines[i]!);
    if (!type) { rest.push(lines[i]!); continue; }
    let start = i;
    for (let back = 1; back <= 2 && rest.length; back++) {
      if (/^\s*\.(globl|p2align)\b/.test(rest.at(-1)!)) { rest.pop(); start--; }
    }
    let end = i;
    for (let j = i + 1; j < lines.length; j++) { end = j; if (/^\s*\.cfi_endproc/.test(lines[j]!)) break; }
    functions.set(type[1]!, lines.slice(start, end + 1).join('\n'));
    i = end;
  }
  return { functions, rest: rest.join('\n') };
}
const program = (functions: Iterable<string>, rest: string) => `\t.text\n${[...functions].join('\n\n')}\n\n${rest}\n`;

export async function compile(source: string, options: CompileOptions): Promise<Compilation> {
  const { language, level = 'O2', inputs = [''], run } = options;
  const limit = options.limit ?? limiter(options.concurrency ?? 4);
  const records: StageRecord[] = [];
  const ask = <T>(work: () => Promise<T>) => limit(() => run(work));

  /** Ask for a stage result; check it; on a problem ask once more with it; record the outcome. */
  async function stage(name: string, fn: string, produce: (problem?: string) => Promise<string>,
    check: (candidate: string) => Promise<string | null>): Promise<string | null> {
    const started = performance.now();
    let problem: string | undefined;
    for (let attempt = 1; attempt <= 2; attempt++) {
      let candidate: string | undefined;
      try { candidate = clean(await produce(problem)); } catch (error) { problem = `the call failed: ${String((error as Error)?.message ?? error).slice(0, 600)}`; }
      if (candidate !== undefined) problem = await check(candidate) ?? undefined;
      const accepted = candidate !== undefined && !problem;
      if (accepted || attempt === 2) {
        const record = { function: fn, stage: name, accepted, attempts: attempt, ms: Math.round(performance.now() - started),
          ...(accepted ? { size: instructions(candidate!) } : { problem }) };
        records.push(record); options.onRecord?.(record);
      }
      if (accepted) return candidate!;
    }
    return null;
  }

  /** A stage that only a later check can judge (an analysis, or a step before the code can run): timed and recorded. */
  async function step<T>(name: string, fn: string, work: () => Promise<T>): Promise<T> {
    const started = performance.now();
    const record = (accepted: boolean, problem?: string) => {
      const entry = { function: fn, stage: name, accepted, attempts: 1, ms: Math.round(performance.now() - started), ...problem ? { problem } : {} };
      records.push(entry); options.onRecord?.(entry);
    };
    try { const value = await work(); record(true); return value; }
    catch (error) { record(false, String((error as Error)?.message ?? error).slice(0, 600)); throw error; }
  }

  const verifies = async (text: string) => { const check = await toolchain.verify(normalize(text)); return check.ok ? null : `the verifier rejected it: ${check.error}`; };
  // Front end: the parser and semantic analysis (a program they find invalid stops here with its diagnostics), the
  // module's layout, whose header must verify (checked and retried like every stage), then each function's IR.
  const front = FRONT_ENDS[language];
  let checked;
  try {
    const syntax = await step('parse', '(module)', () => ask(() => front.parse(source)));
    if (syntax.diagnostics.length) return { ok: false, diagnostics: syntax.diagnostics, records };
    checked = await step('analyze', '(module)', () => ask(() => front.analyze(syntax)));
    if (checked.diagnostics.length) return { ok: false, diagnostics: checked.diagnostics, records };
  } catch (error) { return { ok: false, diagnostics: [`the front end failed: ${String((error as Error)?.message ?? error).slice(0, 600)}`], records }; }
  const analyzed = checked;
  let read: ModuleFrame | undefined;
  const declared = await stage('declare', '(module)', async problem => {
    read = await ask(() => front.declare(analyzed, problem));
    return read.header;
  }, async header => read?.diagnostics.length ? null : verifies(header));
  if (read?.diagnostics.length) return { ok: false, diagnostics: read.diagnostics, records };
  if (declared === null || !read) return { ok: false, diagnostics: [`the front end failed: ${records.at(-1)?.problem ?? 'no answer'}`], records };
  const frame: ModuleFrame = read;
  let header = declared;
  const signatures = new Map(frame.functions.map(f => [f.name, f.signature]));
  const functions = new Map<string, string>();
  if (/^declare[^@\n]*@rt_/m.test(header)) {
    // The runtime library (Python's lists, Rust's vectors, input, panics) is IR written for this program from the
    // contracts its header declares; its functions are optimized and compiled like the program's own.
    const runtime = await stage('runtime', '(runtime)', problem => ask(() => runtimeLibrary(header, problem)), defs => verifies(`${header}\n${defs}`));
    if (runtime === null) return { ok: false, diagnostics: ['the runtime library did not verify'], records };
    header = normalize(`${header}\n${runtime.split('\n').filter(line => line.startsWith('declare')).join('\n')}`);
    for (const definition of runtime.match(/^define[^]*?^\}/gm) ?? []) {
      signatures.set(nameOf(definition)!, definition.slice(0, definition.indexOf('{')).trim());
      functions.set(nameOf(definition)!, definition);
    }
  }
  /** What a stage working on one function sees of the rest: the header and every other function's declaration. */
  const context = (name: string) => normalize([header, ...[...signatures].filter(([other]) => other !== name).map(([, s]) => declaration(s))]
    .join('\n').split('\n').filter(line => !new RegExp(`^declare[^@]*@${name.replace(/[.$]/g, '\\$&')}\\s*\\(`).test(line)).join('\n'));
  const lowered = await Promise.all(frame.functions.map(f => stage('lower', f.name,
    problem => ask(() => front.lower(f, context(f.name), problem)),
    candidate => nameOf(candidate) !== f.name ? Promise.resolve(`the answer must define @${f.name}`) : verifies(`${context(f.name)}\n${candidate}`))));
  if (lowered.some(f => f === null)) return { ok: false, diagnostics: ['a function did not lower to valid IR'], records };
  frame.functions.forEach((f, i) => functions.set(f.name, lowered[i]!));
  const moduleText = (replace?: [string, string]) => normalize([header, ...[...functions].map(([n, t]) => replace?.[0] === n ? replace[1] : t)].join('\n\n'));
  const unoptimized = moduleText();

  // The reference behavior: the expected outputs, or what the front end's program does.
  const reference: { stdout: string, exitCode: number | null }[] = [];
  for (const [i, input] of inputs.entries()) {
    const result = await toolchain.runIR(unoptimized, input);
    if (result.error && result.exitCode === null) return { ok: false, diagnostics: [`the unoptimized program failed: ${result.error}`], records, unoptimized };
    const expected = options.expected?.[i];
    if (expected !== undefined && result.stdout !== expected) {
      return { ok: false, diagnostics: [`the front end's program printed ${quote(result.stdout)} instead of ${quote(expected)}`], records, unoptimized };
    }
    reference.push({ stdout: result.stdout, exitCode: result.exitCode });
  }
  const differs = async (runOne: (input: string) => Promise<{ stdout: string, exitCode: number | null, error?: string }>) => {
    for (const [i, input] of inputs.entries()) {
      const result = await runOne(input);
      if (result.stdout !== reference[i]!.stdout || result.exitCode !== reference[i]!.exitCode) {
        return `on input ${quote(input)} the program printed ${quote(result.stdout)} (exit ${result.exitCode}${result.error ? `, ${result.error}` : ''}) instead of ${quote(reference[i]!.stdout)} (exit ${reference[i]!.exitCode})`;
      }
    }
    return null;
  };
  const behaves = async (text: string) => await verifies(text) ?? await differs(input => toolchain.runIR(text, input));

  // Middle end: each function's pipeline, chosen by the pass manager; functions in parallel. The control-flow analysis
  // is computed for the function as it is when a pass needs it, and again only after the function changed.
  const flows = new Map<string, Promise<Flow>>();
  const flowOf = (name: string, text: string) => {
    if (!flows.has(text)) flows.set(text, step('flow', name, () => ask(() => flow(text))));
    return flows.get(text)!;
  };
  await Promise.all([...functions.keys()].map(async name => {
    const passes: Pass[] = await ask(() => plan(functions.get(name)!, level)).catch(() => ['mem2reg', 'simplify', 'dce'] as Pass[]);
    for (const pass of passes) {
      const produce = async (problem?: string) => {
        const current = functions.get(name)!;
        if (pass in FLOW_PASSES) {
          const facts = await flowOf(name, current);
          return ask(() => FLOW_PASSES[pass as keyof typeof FLOW_PASSES](current, context(name), facts, problem));
        }
        if (pass !== 'inline') return ask(() => PLAIN_PASSES[pass as keyof typeof PLAIN_PASSES](current, context(name), problem));
        const callees = [...calls(current)].filter(c => c !== name && functions.has(c)).map(c => functions.get(c)!).join('\n\n');
        return ask(() => inline(current, callees, context(name), problem));
      };
      const result = await stage(pass, name, produce, candidate =>
        nameOf(candidate) !== name ? Promise.resolve(`the answer must define @${name}`) : behaves(moduleText([name, candidate])));
      if (result !== null) functions.set(name, result);
    }
  }));
  let ir = moduleText();
  if (await behaves(ir)) {
    // Each pass was checked against the others' state at the time; if the combination misbehaves, step back.
    for (const [name, text] of frame.functions.map((f, i) => [f.name, lowered[i]!] as const)) {
      functions.set(name, text);
      ir = moduleText();
      if (!await behaves(ir)) break;
    }
  }
  if (!options.backend) return { ok: true, diagnostics: [], records, unoptimized, ir };

  // Back end: each generated function is checked inside LLVM's own assembly of the same module (the scaffold), then
  // the program made only of generated code is checked as a whole.
  const scaffold = await llvmAssembly(ir);
  if (!scaffold.ok) return { ok: false, diagnostics: [`no scaffold: ${scaffold.error}`], records, unoptimized, ir };
  const { functions: llvmFunctions, rest } = splitAssembly(scaffold.assembly!);
  const runs = (assembly: string) => differs(input => toolchain.runAssembly(assembly, input));
  const spliced = (name: string, text: string) => program([...llvmFunctions].map(([n, t]) => n === name ? text : t), rest);
  const dataText = await stage('data', '(data)', problem => ask(() => data(header, problem)),
    candidate => runs(program(llvmFunctions.values(), candidate)));
  // Code generation runs on virtual registers until frame lowering, so a function's chain (selection, liveness,
  // allocation, frame lowering, emission) is checked as a whole once it is emitted, and redone with the problem.
  const machine = new Map<string, string>();
  await Promise.all([...functions].map(async ([name, fnIR]) => {
    const emitted = await stage('codegen', name, async problem => {
      const code = await step('select', name, () => ask(() => select(fnIR, header, problem)));
      const live = await step('liveness', name, () => ask(() => liveness(code)));
      const allocated = await step('allocate', name, () => ask(() => allocate(code, live, problem)));
      const body = await step('frame', name, () => ask(() => frameLowering(allocated, problem)));
      return step('emit', name, () => ask(() => emit(name, body, problem)));
    }, candidate => runs(spliced(name, candidate)));
    if (emitted === null) return;
    const improved = await stage('peephole', name, problem => ask(() => peephole(emitted, problem)), candidate => runs(spliced(name, candidate)));
    machine.set(name, improved ?? emitted);
  }));
  if (dataText === null || machine.size !== functions.size) {
    return { ok: false, diagnostics: ['the back end did not produce every part'], records, unoptimized, ir };
  }
  const assembly = program(machine.values(), `${dataText}\n\t.section\t".note.GNU-stack","",@progbits`);
  const problem = await runs(assembly);
  return { ok: !problem, diagnostics: problem ? [`the generated program misbehaves: ${problem}`] : [], records, unoptimized, ir, assembly,
    ...(problem ? { problem } : {}) };
}
