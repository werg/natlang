import { readFileSync, mkdirSync } from 'node:fs';
import { resolve, dirname, join } from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { buildProject } from '../compiler/node-project.js';
import { formatDiagnostics } from '../compiler/project.js';
import { fingerprint, immutable, cloneData } from '../adaptation/identity.js';
import { parseStrictJSON } from '../adaptation/schema.js';
import type { EvaluationSuite, EvaluationCase, PreparedSuite } from './types.js';
import { TOOLS_PROMPT, FUNCTION_TOOLS_PROMPT, NL_DEPTH_LIMIT_NOTICE, directoryReducerPrompt } from '../native/prompt.js';
export function defineEvaluationSuite<I, E>(suite: EvaluationSuite<I, E>): EvaluationSuite<I, E> { return suite; }
export function validateCases(cases: readonly EvaluationCase[]): readonly EvaluationCase[] {
  if (!Array.isArray(cases) || !cases.length) throw new Error('evaluation suite needs cases');
  const ids = new Set<string>(), groups = new Map<string, string>();
  for (const testCase of cases) {
    if (!testCase.id || !testCase.group || !['train', 'validation', 'test'].includes(testCase.split)) throw new Error('case needs id, group, and explicit split');
    if (ids.has(testCase.id)) throw new Error('duplicate case ID: ' + testCase.id); ids.add(testCase.id);
    if (groups.has(testCase.group) && groups.get(testCase.group) !== testCase.split) throw new Error('group leakage: ' + testCase.group);
    groups.set(testCase.group, testCase.split);
  }
  return immutable(cloneData(cases));
}
export function validateBudget(budget: EvaluationSuite['budget']): void {
  for (const [key, value] of Object.entries(budget)) {
    if (key === 'pricing' || key === 'requestBounds') {
      const names = key === 'pricing' ? ['inputPerMillion', 'outputPerMillion'] : ['maxInputTokens', 'maxOutputTokens'];
      if (!value || typeof value !== 'object' || names.some(name => !Number.isFinite((value as Record<string, number>)[name]) || (value as Record<string, number>)[name]! < 0) ||
        Object.keys(value).some(name => !names.includes(name))) throw new Error('invalid budget: ' + key);
      if (key === 'requestBounds' && Object.values(value).some(count => !Number.isSafeInteger(count))) throw new Error('request bounds must be integers');
    } else if (!Number.isFinite(value) || (value as number) < 0) throw new Error('invalid budget: ' + key);
  }
  for (const key of ['maxRollouts', 'maxProposals', 'maxModelCalls'] as const) if (!Number.isSafeInteger(budget[key])) throw new Error('budget needs integer ' + key);
  if (budget.maxCost !== undefined && (!budget.pricing || !budget.requestBounds)) throw new Error('hard monetary caps require configured pricing and bounded requests');
}
/** A module URL is mandatory: fixture functions are imported afresh in workers, never serialized. */
export async function prepareEvaluationSuite(suite: EvaluationSuite, moduleURL: string): Promise<PreparedSuite> {
  validateBudget(suite.budget);
  if (suite.selection) {
    if (suite.selection.tieBreak !== undefined && !['baseline', 'modelCalls', 'latency', 'cost'].includes(suite.selection.tieBreak)) throw new Error('invalid selection tie-break');
    for (const [key, value] of Object.entries(suite.selection)) if (key !== 'tieBreak' &&
      (!['maxMeanLatencyMs', 'maxModelCalls', 'maxCost'].includes(key) || !Number.isFinite(value) || Number(value) < 0)) throw new Error('invalid selection constraint: ' + key);
  }
  if (suite.systemPrompt && fingerprint(suite.systemPrompt, 'natlang.system-prompt/v1') !== suite.policy?.systemPromptHash)
    throw new Error('declare the evaluated application systemPromptHash in suite policy');
  if (!suite.executorIdentity?.id) throw new Error('declare executorIdentity for reproducible artifact binding');
  const modulePath = fileURLToPath(moduleURL); const root = resolve(dirname(modulePath), suite.program.root);
  const cases = validateCases(typeof suite.cases === 'string' ? readFileSync(resolve(root, suite.cases), 'utf8').split(/\r?\n/)
    .filter(line => line.trim()).map(line => parseStrictJSON(line) as EvaluationCase) : suite.cases);
  const outDir = join(root, '.natlang', 'adaptation', 'build'); mkdirSync(outDir, { recursive: true });
  const result = buildProject({ project: root, programId: suite.program.id, guidance: suite.program.guidance, services: suite.services, outDir, write: true,
    writeDeclarations: false, runtimeModule: { specifiers: ['@natlang/node'], url: new URL('../index.js', import.meta.url).href,
      types: fileURLToPath(new URL('../index.d.ts', import.meta.url)) } });
  if (!result.ok) throw new Error(formatDiagnostics(result.diagnostics));
  const program = result.manifest.adaptation!;
  const selected = typeof suite.components === 'string' ? program.components.filter(component => suite.components === 'joint' ||
    (suite.components === 'lambda-only' ? component.kind === 'lambda.instructions' : component.kind === 'program.guidance')).map(component => component.key) : [...suite.components];
  if (!selected.length || new Set(selected).size !== selected.length || selected.some(key => !program.components.some(component => component.key === key)))
    throw new Error('selection contains unknown, duplicate, or nonpersistent components; only named .nl, compiler-authored inline sites, and program guidance are persistent; generated/delegate and unregistered callables are excluded');
  const entry = resolve(outDir, suite.program.entry.replace(/\.m?ts$/, '.js'));
  const suiteHash = fingerprint({ id: suite.id, module: readFileSync(modulePath, 'utf8'), program: program.buildHash,
    protocol: program.protocol, prompt: { interpreter: TOOLS_PROMPT, functions: FUNCTION_TOOLS_PROMPT, depth: NL_DEPTH_LIMIT_NOTICE,
      folder: directoryReducerPrompt(), guidance: suite.program.guidance ?? '' },
    cases, selected, identity: suite.executorIdentity, policy: suite.policy ?? { codeEdits: 'deny', settings: {} },
    fixture: suite.fixture.create.toString(), score: suite.score.toString(), feedback: suite.feedback?.toString() ?? null,
    gates: suite.requiredGates ?? [], selection: suite.selection ?? null, services: suite.services ?? null, systemPrompt: suite.systemPrompt ?? null });
  return { suite, moduleURL, cases, program, suiteHash, components: selected.sort(), compiledEntry: pathToFileURL(entry).href };
}
export async function loadEvaluationSuite(path: string): Promise<PreparedSuite> {
  const sourcePath = resolve(path);
  let moduleURL = pathToFileURL(sourcePath).href;
  if (/\.m?ts$/.test(sourcePath)) {
    const root = dirname(sourcePath), outDir = join(root, '.natlang/adaptation/suite');
    const result = buildProject({ project: root, outDir, write: true, writeDeclarations: false,
      runtimeModule: { specifiers: ['@natlang/node'], url: new URL('../index.js', import.meta.url).href,
        types: fileURLToPath(new URL('../index.d.ts', import.meta.url)) } });
    if (!result.ok) throw new Error(formatDiagnostics(result.diagnostics));
    moduleURL = pathToFileURL(join(outDir, sourcePath.slice(root.length + 1).replace(/\.m?ts$/, '.js'))).href;
  }
  const imported = await import(moduleURL);
  const suite = imported.default as EvaluationSuite;
  return prepareEvaluationSuite({ ...suite, program: { ...suite.program, root: resolve(dirname(sourcePath), suite.program.root) } }, moduleURL);
}
