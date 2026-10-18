/**
 * Acceptance for TypeScript authoring tasks: the project folder a directory reducer edited is mounted as a
 * callable folder and its authored export is called on test inputs. Its `nl` children are answered by an
 * oracle keyed by fragments of the child's opening (its argument values), since the author's instruction
 * wording is not known in advance; an unmatched child fails, which fails the check.
 */
import { isDeepStrictEqual } from 'node:util';
import { createHash } from 'node:crypto';
import ts from 'typescript';
import { withCallOf } from '../compiler/inline.js';
import { createNatlangRuntime } from '../runtime/node.js';
import { loadVirtualNatlang } from '../runtime/virtual-project.js';
import type { ModelTurn, ModelTurnOptions, ModelTurnRequest } from '../contracts.js';
import { openingText, type Message } from './opening.js';

export type AuthoringSpec = {
  /** The authored module, a path in the folder (for example "app.ts"), and the export to call. */
  module: string;
  export: string;
  /** Calls to make and the results they must produce. */
  runs: { args: unknown[]; expected: unknown }[];
  /** Answers for the export's natural-language children: every fragment must appear in the child's opening. */
  oracle?: { match: string[]; value: unknown }[];
  /** Exact aliases derived from unchanged declared run arguments and original keyed oracle. */
  argument_aliases?: { match: string[]; value: unknown; source_id: string; source_text: string }[];
  /** Offline independently reviewed literal child criteria; source tests remain unchanged. */
  source_criterion_review?: { schema: 'natlang.reviewed-source-child-criteria/1';
    module_sha256: string; literal_bindings?: { policy: string };
    criteria: { instruction: string; result_kind: 'boolean' | 'decision' }[] };
  /** What the authored source must use. */
  requires?: { nl?: boolean; iterateOn?: boolean; calls?: string[]; noNl?: boolean };
};

export type AuthoringResult = { ok: boolean; problems: string[]; results: unknown[] };

/** Derive text aliases only from exact source Ticket records, never generated answers. */
export function withSourceArgumentAliases(spec: AuthoringSpec, files?: Record<string, string>): AuthoringSpec {
  const { argument_aliases: _supplied, ...original } = spec;
  spec = original;
  const aliases: NonNullable<AuthoringSpec['argument_aliases']> = [];
  const byText = new Map<string, unknown>();
  function visit(value: unknown): void {
    if (Array.isArray(value)) { value.forEach(visit); return; }
    if (!value || typeof value !== 'object') return;
    const record = value as Record<string, unknown>;
    if (Object.keys(record).sort().join(',') === 'id,text' &&
        typeof record.id === 'string' && typeof record.text === 'string' && record.text.length) {
      const keyed = spec.oracle?.filter(item => item.match.length === 1 && item.match[0] === JSON.stringify(record.id)) ?? [];
      for (const item of keyed) {
        const encoded = JSON.stringify(record.text);
        if ((byText.has(record.text) && !isDeepStrictEqual(byText.get(record.text), item.value)) ||
            (spec.oracle ?? []).some(existing => existing.match.every(part => encoded.includes(part)) &&
              !isDeepStrictEqual(existing.value, item.value))) throw new Error('Conflicting source argument oracle aliases.');
        if (!byText.has(record.text)) aliases.push({ match: [encoded], value: item.value,
          source_id: record.id, source_text: record.text });
        byText.set(record.text, item.value);
      }
    }
    Object.values(record).forEach(visit);
  }
  spec.runs.forEach(run => visit(run.args));
  if (spec.source_criterion_review) {
    const review = spec.source_criterion_review, source = files?.[spec.module];
    if (review.schema !== 'natlang.reviewed-source-child-criteria/1' || typeof source !== 'string' ||
        createHash('sha256').update(source).digest('hex') !== review.module_sha256 ||
        !Array.isArray(review.criteria) || !review.criteria.length ||
        review.criteria.some(value => !value || typeof value.instruction !== 'string' || !value.instruction.trim() ||
          (value.result_kind !== 'boolean' && value.result_kind !== 'decision')) ||
        new Set(review.criteria.map(value => value.instruction)).size !== review.criteria.length)
      throw new Error('Reviewed source child criteria do not bind the captured module.');
    const parsed = ts.createSourceFile(spec.module, source, ts.ScriptTarget.Latest, true);
    const literalCriteria = new Set<string>();
    let policy: string | undefined;
    if (review.literal_bindings) {
      if (Object.keys(review.literal_bindings).join(',') !== 'policy' || typeof review.literal_bindings.policy !== 'string')
        throw new Error('Reviewed source interpolation requires one exact policy literal.');
      const declarations: ts.Node[] = [];
      function bindings(node: ts.Node): void {
        if ((ts.isVariableDeclaration(node) || ts.isParameter(node) || ts.isBindingElement(node) ||
            ts.isFunctionDeclaration(node) || ts.isClassDeclaration(node) || ts.isImportSpecifier(node) ||
            ts.isImportClause(node) || ts.isNamespaceImport(node)) && node.name &&
            (ts.isIdentifier(node.name) ? node.name.text === 'policy' : node.name.getText(parsed).includes('policy')))
          declarations.push(node);
        ts.forEachChild(node, bindings);
      }
      bindings(parsed);
      const node = declarations[0];
      if (declarations.length !== 1 || !node || !ts.isVariableDeclaration(node) || !node.initializer ||
          !ts.isStringLiteral(node.initializer) || node.initializer.text !== review.literal_bindings.policy ||
          !ts.isVariableDeclarationList(node.parent) || !(node.parent.flags & ts.NodeFlags.Const) ||
          !ts.isVariableStatement(node.parent.parent) || node.parent.parent.parent !== parsed)
        throw new Error('Reviewed source interpolation is not an unshadowed module const policy.');
      policy = node.initializer.text;
    }
    function collect(node: ts.Node): void {
      if (ts.isTaggedTemplateExpression(node) &&
          ((ts.isIdentifier(node.tag) && node.tag.text === 'nl') || withCallOf(undefined, node.tag))) {
        if (ts.isNoSubstitutionTemplateLiteral(node.template)) literalCriteria.add(node.template.text.trim());
        else if (policy !== undefined && node.template.templateSpans.every(span =>
            ts.isIdentifier(span.expression) && span.expression.text === 'policy'))
          literalCriteria.add((node.template.head.text + node.template.templateSpans.map(span =>
            policy + span.literal.text).join('')).trim());
      }
      ts.forEachChild(node, collect);
    }
    collect(parsed);
    if (review.criteria.some(value => !literalCriteria.has(value.instruction)))
      throw new Error('Reviewed source child criterion is not an exact literal in the captured module.');
    const labels = new Map<string, { value: boolean; decision: 'hold' | 'rollback'; id: string }>();
    for (const run of spec.runs) {
      if (run.args.length !== 1 || !Array.isArray(run.args[0]) || !run.expected ||
          typeof run.expected !== 'object' || Array.isArray(run.expected))
        throw new Error('Reviewed source child requires exact declared service tests.');
      for (const record of run.args[0]) {
        if (!record || typeof record !== 'object' || Object.keys(record).sort().join(',') !== 'id,notes,requests' ||
            typeof record.id !== 'string' || typeof record.notes !== 'string' || !record.notes ||
            !Array.isArray(record.requests) || record.requests.some((r: unknown) => !r || typeof r !== 'object' ||
              Object.keys(r).join(',') !== 'status' || !Number.isInteger((r as { status?: unknown }).status)))
          throw new Error('Reviewed source child requires unchanged full source service records.');
        const expected = (run.expected as Record<string, unknown>)[record.id];
        if (expected !== 'hold' && expected !== 'rollback') continue;
        const failed = record.requests.filter((r: { status: number }) => r.status >= 500).length;
        if (!record.requests.length || failed * 20 <= record.requests.length ||
            !spec.oracle?.some(item => item.value === expected && item.match.includes(record.notes)))
          throw new Error('Reviewed source child lacks agreeing above-threshold source oracle evidence.');
        if (spec.oracle.some(item => item.match.includes(record.notes) &&
            (item.value === 'hold' || item.value === 'rollback') && item.value !== expected))
          throw new Error('Conflicting reviewed source note oracle evidence.');
        const value = expected === 'hold', prior = labels.get(record.notes);
        if (prior && prior.value !== value) throw new Error('Conflicting reviewed source note labels.');
        if (!prior) labels.set(record.notes, { value, decision: expected, id: record.id });
      }
    }
    if (!labels.size) throw new Error('Reviewed source child has no independent source labels.');
    for (const criterion of review.criteria) for (const [notes, label] of labels)
      aliases.push({ match: [criterion.instruction, JSON.stringify(notes)],
        value: criterion.result_kind === 'boolean' ? label.value : label.decision,
        source_id: label.id, source_text: notes });
  }
  return aliases.length ? { ...spec, argument_aliases: aliases } : spec;
}

const HARNESS = 'authoring_check';

function createsNl(source: string): boolean {
  const parsed = ts.createSourceFile('authoring.ts', source, ts.ScriptTarget.Latest, true);
  let found = false;
  function visit(node: ts.Node): void {
    if (ts.isTaggedTemplateExpression(node) &&
        ((ts.isIdentifier(node.tag) && node.tag.text === 'nl') || withCallOf(undefined, node.tag))) found = true;
    ts.forEachChild(node, visit);
  }
  visit(parsed);
  return found;
}

/** Historical requirement accounting only; current admission re-runs checkAuthoring. */
export async function checkAuthoringLegacyRequirements(files: Record<string, string>, spec: AuthoringSpec): Promise<AuthoringResult> {
  return runAuthoring(files, spec, true);
}

/** Run the authored export on every test input. */
export async function checkAuthoring(files: Record<string, string>, spec: AuthoringSpec): Promise<AuthoringResult> {
  return runAuthoring(files, spec, false);
}

async function runAuthoring(files: Record<string, string>, spec: AuthoringSpec, legacyRequirements: boolean): Promise<AuthoringResult> {
  const problems: string[] = [];
  if ((spec.argument_aliases || spec.source_criterion_review) && !isDeepStrictEqual(spec.argument_aliases, withSourceArgumentAliases(spec, files).argument_aliases))
    return { ok: false, problems: ['Source argument oracle aliases differ from the original declared runs.'], results: [] };
  const source = files[spec.module];
  if (source === undefined) return { ok: false, problems: [`${spec.module} does not exist`], results: [] };
  const hasNl = legacyRequirements ? /\bnl\s*(?:<[^`]*>)?\s*`/.test(source) : createsNl(source);
  if (spec.requires?.nl && !hasNl) problems.push(`${spec.module} creates no nl function`);
  if (spec.requires?.iterateOn && !/\biterateOn\b/.test(source)) problems.push(`${spec.module} does not use iterateOn`);
  if (spec.requires?.noNl && hasNl) problems.push(`${spec.module} creates an nl function where an existing one fits`);
  for (const name of spec.requires?.calls ?? []) if (!new RegExp(`\\b${name}\\s*\\(`).test(source)) problems.push(`${spec.module} does not call ${name}`);
  // The project's natlang and TypeScript files become the harness's callable folder.
  const mounted: Record<string, string> = {};
  for (const [path, text] of Object.entries(files)) if (/\.(?:ts|nl)$/.test(path)) mounted[`${HARNESS}/${path}`] = text;
  const moduleName = spec.module.replace(/\.ts$/, '').split('/').join('.');
  mounted[`${HARNESS}.nl`] = `---\nargs:\n  payload: string\nreturns: string\n---\nRun the check.\n`;
  const results: unknown[] = [];
  for (const [index, run] of spec.runs.entries()) {
    let staged = false;
    const model = Object.assign(async (request: ModelTurnRequest, _signal?: AbortSignal, options?: ModelTurnOptions): Promise<ModelTurn> => {
      const opening = String((request.messages[1] as { content?: unknown } | undefined)?.content ?? '');
      if (opening.startsWith(`You are inside this call: ${HARNESS}(`)) {
        // After the eval staged its result, a reply without a tool call returns it; a failed eval ends the call.
        if (staged) {
          const last = String((request.messages.at(-1) as { content?: unknown } | undefined)?.content ?? '');
          return /\nStaged /.test(`\n${last}`) || last.includes('Staged ') ? { text: 'done' } :
            { calls: [['return_result', { status: 'failed', reason: `The authored export failed: ${last.slice(0, 400)}` }]] };
        }
        staged = true;
        return { calls: [['eval', { code: `return JSON.stringify(await ${moduleName}.${spec.export}(...JSON.parse(payload)) ?? null);` }]] };
      }
      const shown = openingText(request.messages as Message[]);
      const matches = [...(spec.oracle ?? []), ...(spec.argument_aliases ?? [])]
        .filter(item => item.match.every(fragment => shown.includes(fragment)))
        .filter(item => options?.acceptsResult?.(item.value) === true);
      const answer = matches[0];
      if (spec.argument_aliases && answer && matches.some(item => !isDeepStrictEqual(item.value, answer.value)))
        return { calls: [['return_result', { status: 'failed', reason: 'Conflicting source argument oracle answers.' }]] };
      return answer ? { calls: [['return_result', { status: 'success', value: answer.value }]] } :
        { calls: [['return_result', { status: 'failed', reason: 'The authoring oracle has no answer for this call.' }]] };
    }, { requiresResultContract: true });
    const runtime = createNatlangRuntime({ model: model as never, seed: { mode: 'backend' } });
    try {
      const fn = loadVirtualNatlang(mounted, `${HARNESS}.nl`);
      const text = await runtime.run(() => (fn as unknown as (payload: string) => Promise<string>)(JSON.stringify(run.args)));
      const value = JSON.parse(String(text));
      results.push(value);
      if (!isDeepStrictEqual(value, run.expected)) problems.push(`run ${index}: got ${JSON.stringify(value)}, expected ${JSON.stringify(run.expected)}`);
    } catch (error) {
      results.push(null);
      problems.push(`run ${index}: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
  return { ok: !problems.length, problems, results };
}
