/**
 * `migrateProgram`: migrate one legacy prose program into idiomatic semantic FP, or delete it
 * (plans/semantic-fp/07_CORPUS_MIGRATION.md; DECOMPOSITION.md "Program migration"). This module is the fixed sequence
 * and the exact checks; every decision is a typed judgment in this folder. `propose` runs before the equivalence
 * measurement, `settle` after it; `scripts/semantic_fp/migrate_corpus.py` batches the measurement between them.
 */
import { spawn } from 'node:child_process';
import { cpSync, existsSync, mkdtempSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync, mkdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, dirname, join } from 'node:path';
import { semanticRepair } from '@natlang/node/std';
import stepKind from './stepKind.nl';
import worthKeeping from './worthKeeping.nl';
import nextAttempt from './nextAttempt.nl';
import rewriteLeaf from './rewriteLeaf.nl';
import rewriteDriver from './rewriteDriver.nl';
import type { Baseline, Comparison, DriverRewrite, Finding, KindedStep, Keep, Next } from './types.js';

/** The codes of the idiom floor (`natlang check`, ts-host/src/compiler/idiom.ts) plus the untrusted-instruction error. */
export const FLOOR_CODES = ['idiom-decision-readout', 'idiom-code-in-prose', 'idiom-fixed-cursor-iterate', 'untrusted-instruction'];

export type CheckResult = { ok: boolean, diagnostics: (Finding & { severity: string })[], functions: { source: string, signature: string }[] };

/** One program to migrate, as the corpus driver prepares it. */
export type Unit = { program: string, root: string, brief: string, baseline: Baseline, problems?: string[], history?: string[] };

export type FileChange = { path: string, before: string | null, after: string | null, role: 'leaf' | 'driver' | 'helper',
  steps?: KindedStep[], repairAttempts?: number, problems?: string[] };

export type Proposal = {
  schema: 'natlang.semantic-fp-migration-proposal/1', program: string, root: string,
  status: 'clean' | 'delete' | 'candidate' | 'unchecked',
  keep: Keep | null, keepBy: 'floor' | 'judgment' | null, findings: Finding[], changes: FileChange[], problems: string[],
};

// ---- crisp plumbing ----------------------------------------------------------------------------------------------------

/** `natlang check --json DIR` (the CLI on PATH, or NATLANG_BIN). Exit 1 still prints the JSON report. */
export function checkProgram(directory: string): Promise<CheckResult> {
  return new Promise((resolveCheck, reject) => {
    const bin = process.env.NATLANG_BIN || 'natlang';
    const child = spawn(bin, ['check', '--json', directory], { stdio: ['ignore', 'pipe', 'pipe'] });
    let out = '', err = '';
    child.stdout.on('data', chunk => { out += chunk; });
    child.stderr.on('data', chunk => { err += chunk; });
    child.on('error', reject);
    child.on('close', () => {
      try {
        const parsed = JSON.parse(out) as Partial<CheckResult>;
        resolveCheck({ ok: !!parsed.ok, diagnostics: parsed.diagnostics ?? [], functions: parsed.functions ?? [] });
      } catch { reject(new Error(`natlang check gave no JSON report for ${directory}: ${err.slice(-400)}`)); }
    });
  });
}

/** The front matter (between the opening and closing `---` lines, without them) and the body of a `.nl` file. */
export function splitSource(text: string): { front: string, body: string } {
  const match = /^---\n([\s\S]*?)\n---\n?([\s\S]*)$/.exec(text);
  if (!match) throw new Error('a .nl file starts with front matter between --- lines');
  return { front: match[1]!, body: match[2]! };
}

/** The body's steps: its numbered items when it has them (text before the first one is a step of its own), else its
 * paragraphs. A fenced block stays inside the step it belongs to. */
export function stepsOf(body: string): string[] {
  const lines = body.replace(/\s+$/, '').split('\n');
  const numbered = lines.some(line => /^\d+\.\s/.test(line));
  const steps: string[][] = [];
  let fenced = false;
  for (const line of lines) {
    const opens = !fenced && (numbered ? /^\d+\.\s/.test(line) : line.trim() !== '' && (steps.length === 0 || steps.at(-1)!.at(-1)?.trim() === ''));
    if (opens || steps.length === 0) steps.push([]);
    steps.at(-1)!.push(line);
    if (/^\s*```/.test(line)) fenced = !fenced;
  }
  return steps.map(step => step.join('\n').trim()).filter(Boolean);
}

/** The file with `readout: decision` added to its front matter (unchanged when it has a readout already) and `body`. */
export function assemble(front: string, body: string, readout: boolean): string {
  const withReadout = readout && !/^readout:/m.test(front) ? `${front}\nreadout: decision` : front;
  return `---\n${withReadout}\n---\n${body.trim()}\n`;
}

/** The front matter without its `readout:` line: what must not change, since the cases call the function as declared. */
const declaration = (front: string) => front.split('\n').filter(line => !/^readout:/.test(line)).join('\n').trim();

function stripFence(text: string): string {
  const fenced = /^\s*```[a-z]*\n([\s\S]*?)\n```\s*$/.exec(text);
  return (fenced ? fenced[1]! : text).trim();
}

/** Copy `root` into a fresh directory and write `files` ({path: text, or null to remove}) over it. */
export function materialize(root: string, files: Record<string, string | null>): string {
  const directory = mkdtempSync(join(tmpdir(), 'natlang-program-migration-'));
  cpSync(root, directory, { recursive: true, filter: source => !/(^|\/)(node_modules|\.natlang|dist)(\/|$)/.test(source.slice(root.length)) });
  for (const [path, text] of Object.entries(files)) {
    const target = join(directory, path);
    if (text === null) { rmSync(target, { force: true }); continue; }
    mkdirSync(dirname(target), { recursive: true });
    writeFileSync(target, text);
  }
  return directory;
}

const floorFindings = (check: CheckResult): Finding[] => check.diagnostics
  .filter(d => FLOOR_CODES.includes(d.code))
  .map(({ file, line, code, message }) => ({ file, line, code, message }));

/**
 * The exact problems of a candidate: `natlang check` errors, idiom-floor findings in the changed files, a changed
 * declaration (front matter other than `readout:`), and a changed list of function signatures.
 */
export async function problemsOf(root: string, original: CheckResult, files: Record<string, string | null>): Promise<string[]> {
  const problems: string[] = [];
  for (const [path, text] of Object.entries(files)) {
    if (text === null || !path.endsWith('.nl')) continue;
    const before = readFileSync(join(root, path), 'utf8');
    let after;
    try { after = splitSource(text); } catch (error) { problems.push(`${path}: ${(error as Error).message}`); continue; }
    if (declaration(after.front) !== declaration(splitSource(before).front))
      problems.push(`${path}: its front matter changed; keep args and returns exactly as they are.`);
    if (!after.body.trim()) problems.push(`${path}: the instructions are empty.`);
  }
  const directory = materialize(root, files);
  try {
    const check = await checkProgram(directory);
    for (const d of check.diagnostics) {
      if (d.severity === 'error') problems.push(`${d.file}:${d.line}: ${d.message}`);
      else if (FLOOR_CODES.includes(d.code) && Object.hasOwn(files, d.file)) problems.push(`${d.file}:${d.line}: ${d.message}`);
    }
    const signatures = (c: CheckResult) => c.functions.map(f => `${f.source} ${f.signature}`).sort().join('\n');
    if (signatures(check) !== signatures(original)) problems.push('the functions of the program changed their signatures; keep every signature as it is.');
    const helpers = Object.keys(files).filter(path => path.endsWith('.ts') && files[path] !== null);
    if (helpers.length) problems.push(...await typeProblems(directory, helpers));
  } finally { rmSync(directory, { recursive: true, force: true }); }
  return problems;
}

/** The ts-host whose compiler and `@natlang/node` declarations type-check helpers: NATLANG_TS_HOST, else the one beside
 * the natlang CLI (NATLANG_BIN is `<repo>/scripts/natlang`). */
function tsHost(): string {
  if (process.env.NATLANG_TS_HOST) return process.env.NATLANG_TS_HOST;
  if (process.env.NATLANG_BIN) return join(dirname(dirname(process.env.NATLANG_BIN)), 'ts-host');
  throw new Error('set NATLANG_TS_HOST or NATLANG_BIN to type-check helpers');
}

/**
 * Type-check callable-folder helpers (`natlang check` does not): each `.nl` a helper imports gets a declaration taken
 * from its root function's generated `.d.nl.ts`, and a helper that is the default export of a driver's folder must
 * have the driver's parameters and result. Problems are the compiler's errors, one per line.
 */
export async function typeProblems(directory: string, helpers: string[]): Promise<string[]> {
  const host = tsHost();
  const checks: string[] = [];
  for (const helper of helpers) {
    const folder = dirname(helper), parts = folder.split('/');
    const rootName = parts[0]!;
    if (folder === '.' || !existsSync(join(directory, `${rootName}.nl`))) continue;
    const up = (depth: number) => '../'.repeat(depth) || './';
    const path = (name: string) => [...parts.slice(1), name].map(p => `[${JSON.stringify(p)}]`).join('');
    for (const entry of readdirSync(join(directory, folder))) {
      if (!entry.endsWith('.nl') || existsSync(join(directory, folder, `${entry.slice(0, -3)}.d.nl.ts`))) continue;
      writeFileSync(join(directory, folder, `${entry.slice(0, -3)}.d.nl.ts`),
        `import type root from '${up(parts.length)}${rootName}.nl';\ndeclare const fn: (typeof root)${path(entry.slice(0, -3))};\nexport default fn;\n`);
    }
    // The driver calls its helper with its own arguments and returns what it returns.
    const driver = parts.length === 1 ? rootName : null;
    if (driver && basename(helper, '.ts') === 'run') {
      const check = `${folder}/__signature_check.ts`;
      writeFileSync(join(directory, check), `import type driver from '../${driver}.nl';\nimport run from './run.js';\n` +
        `export const sameSignature: (...args: Parameters<typeof driver>) => ReturnType<typeof driver> = run;\n`);
      checks.push(check);
    }
  }
  writeFileSync(join(directory, 'tsconfig.migration-check.json'), JSON.stringify({
    compilerOptions: { strict: true, noEmit: true, module: 'NodeNext', moduleResolution: 'NodeNext', target: 'ES2023',
      allowArbitraryExtensions: true, skipLibCheck: true, types: [],
      paths: { '@natlang/node': [join(host, 'dist/index.d.ts')], '@natlang/node/std': [join(host, 'dist/std/index.d.ts')] } },
    files: [...helpers, ...checks] }));
  const output = await new Promise<string>((resolveOutput, reject) => {
    const child = spawn(process.execPath, [join(host, 'node_modules/typescript/bin/tsc'), '-p', 'tsconfig.migration-check.json'],
      { cwd: directory, stdio: ['ignore', 'pipe', 'pipe'] });
    let text = '';
    child.stdout.on('data', chunk => { text += chunk; });
    child.stderr.on('data', chunk => { text += chunk; });
    child.on('error', reject);
    child.on('close', () => resolveOutput(text));
  });
  return output.split('\n').filter(line => /error TS\d+/.test(line))
    .map(line => line.replace(/__signature_check\.ts\(\d+,\d+\): error TS\d+: /, 'run.ts default export: it must take the function\'s arguments in order and return its result: '))
    .slice(0, 20);
}

/** The functions of a callable folder with their declarations: `name: (args) => returns` from each `.nl` front matter. */
function callableOf(folder: string): string[] {
  if (!existsSync(folder) || !statSync(folder).isDirectory()) return [];
  const names: string[] = [];
  for (const entry of readdirSync(folder).sort()) {
    if (!entry.endsWith('.nl')) continue;
    const { front } = splitSource(readFileSync(join(folder, entry), 'utf8'));
    names.push(`${entry.slice(0, -3)}: ${declaration(front).replace(/\n/g, ' ')}`);
  }
  return names;
}

// ---- the stages ------------------------------------------------------------------------------------------------------

/** Keep or delete: the crisp floor first (no passing original case: delete), then the judgment. */
export async function keepOf(unit: Unit, findings: Finding[]): Promise<{ keep: Keep, by: 'floor' | 'judgment' }> {
  if (unit.baseline.scored > 0 && unit.baseline.passes === 0) return { keep: 'delete', by: 'floor' };
  return { keep: await worthKeeping(unit.brief, unit.baseline, findings), by: 'judgment' };
}

async function kinded(path: string, body: string, front: string): Promise<KindedStep[]> {
  const steps = stepsOf(body);
  const kinds = await Promise.all(steps.map(step => stepKind(step, declaration(front), path)));
  return steps.map((text, index) => ({ text, kind: kinds[index]! }));
}

/** A leaf judgment: its instructions rewritten until the exact check passes (or the repair budget is spent), with
 * `readout: decision` when the floor asked for one. */
async function migrateLeaf(root: string, original: CheckResult, path: string, codes: string[], problems: string[]): Promise<FileChange> {
  const before = readFileSync(join(root, path), 'utf8');
  const { front, body } = splitSource(before);
  const steps = await kinded(path, body, front);
  const readout = codes.includes('idiom-decision-readout');
  const fileOf = (instructions: string) => assemble(front, stripFence(instructions), readout);
  const first = await rewriteLeaf(body, steps, declaration(front), problems);
  const repaired = await semanticRepair(first, {
    check: (instructions: string) => problemsOf(root, original, { [path]: fileOf(instructions) }),
    fix: (_instructions: string, found: string[]) => rewriteLeaf(body, steps, declaration(front), [...problems, ...found]),
    attempts: 3,
  });
  return { path, before, after: fileOf(repaired.value), role: 'leaf', steps, repairAttempts: repaired.attempts, problems: repaired.problems };
}

/** A driver whose steps are mechanism: a TypeScript helper in its callable folder and one instruction calling it. */
async function migrateDriver(root: string, original: CheckResult, path: string, problems: string[]): Promise<FileChange[]> {
  const before = readFileSync(join(root, path), 'utf8');
  const { front, body } = splitSource(before);
  const steps = await kinded(path, body, front);
  const name = basename(path, '.nl');
  const folder = join(dirname(path), name);
  const helperName = 'run';
  const helperPath = join(folder, `${helperName}.ts`);
  const callable = callableOf(join(root, folder));
  const types = existsSync(join(root, 'types.ts')) ? readFileSync(join(root, 'types.ts'), 'utf8') : '';
  const filesOf = (rewrite: DriverRewrite) => ({ [path]: assemble(front, stripFence(rewrite.instructions), false), [helperPath]: `${stripFence(rewrite.helper)}\n` });
  const first = await rewriteDriver(name, body, steps, declaration(front), callable, types, helperName, null, problems);
  const repaired = await semanticRepair(first, {
    check: (rewrite: DriverRewrite) => problemsOf(root, original, filesOf(rewrite)),
    fix: (previous: DriverRewrite, found: string[]) => rewriteDriver(name, body, steps, declaration(front), callable, types, helperName, previous, [...problems, ...found]),
    attempts: 3,
  });
  const files = filesOf(repaired.value);
  return [
    { path, before, after: files[path]!, role: 'driver', steps, repairAttempts: repaired.attempts, problems: repaired.problems },
    { path: helperPath, before: existsSync(join(root, helperPath)) ? readFileSync(join(root, helperPath), 'utf8') : null, after: files[helperPath]!, role: 'helper' },
  ];
}

/** A file is a driver when its finding is prose code and it has a callable folder; otherwise a leaf. */
export function roleOf(root: string, path: string, codes: string[]): 'leaf' | 'driver' {
  const folder = join(root, dirname(path), basename(path, '.nl'));
  return codes.includes('idiom-code-in-prose') && existsSync(folder) && statSync(folder).isDirectory() ? 'driver' : 'leaf';
}

/** Propose a migration of one program: the floor's findings, keep or delete, then one rewrite per flagged file. */
export async function propose(unit: Unit): Promise<Proposal> {
  const original = await checkProgram(unit.root);
  const findings = floorFindings(original);
  const base = { schema: 'natlang.semantic-fp-migration-proposal/1' as const, program: unit.program, root: unit.root, findings };
  if (!findings.length) return { ...base, status: 'clean', keep: null, keepBy: null, changes: [], problems: [] };
  const { keep, by } = await keepOf(unit, findings);
  if (keep === 'delete') return { ...base, status: 'delete', keep, keepBy: by, changes: [], problems: [] };
  const byFile = new Map<string, string[]>();
  for (const finding of findings) byFile.set(finding.file, [...(byFile.get(finding.file) ?? []), finding.code]);
  const changes = (await Promise.all([...byFile].map(([path, codes]) => roleOf(unit.root, path, codes) === 'driver'
    ? migrateDriver(unit.root, original, path, unit.problems ?? [])
    : migrateLeaf(unit.root, original, path, codes, unit.problems ?? []).then(change => [change])))).flat();
  const files = Object.fromEntries(changes.map(change => [change.path, change.after]));
  const problems = await problemsOf(unit.root, original, files);
  return { ...base, status: problems.length ? 'unchecked' : 'candidate', keep, keepBy: by, changes, problems };
}

/** Write a proposal's candidate: the program copied with its changes, at `out`. */
export function writeCandidate(proposal: Proposal, out: string): void {
  const directory = materialize(proposal.root, Object.fromEntries(proposal.changes.map(change => [change.path, change.after])));
  rmSync(out, { recursive: true, force: true });
  mkdirSync(dirname(out), { recursive: true });
  cpSync(directory, out, { recursive: true });
  rmSync(directory, { recursive: true, force: true });
}

/** After the measurement: accepted by the exact rule, or retry / give up by the judgment within the attempts left. */
export async function settle(comparison: Comparison, attemptsLeft: number, history: string[]): Promise<'accept' | Next> {
  if (comparison.accepted) return 'accept';
  if (attemptsLeft <= 0) return 'give-up';
  return nextAttempt(comparison, attemptsLeft, history);
}
