/**
 * Groups for the specializer (§5.2, §7.5): the calls of one function split by what they did, each group small enough for
 * one focused model call that writes that group's case or says why it cannot. The host finds the groups, measures any
 * guard a writer proposes exactly, and assembles the accepted cases into one cases file. Node and browser safe.
 */
import ts from 'typescript';
import { candidatePredicates, compileGuard, splitOf, type Approach, type Example } from './mining.js';
import { exampleOf, type Study } from './specializer.js';
import type { CallStore } from './store.js';

/** One group: the calls that did one thing (a behavior, or an approach when every call did the same thing). */
export type Group = { id: string; label: string; training: Example[]; heldOut: number; approaches: Approach[] };

/** The label mining separates calls by: the behavior, unless every call had the same behavior (then the approach). */
export function labeler(examples: readonly Example[]): (example: Example) => string {
  const byBehavior = new Set(examples.map(example => example.behavior)).size > 1;
  return example => byBehavior ? example.behavior ?? 'none' : example.approach;
}

/** The groups worth a writer: at least `minSupport` training calls, largest first, at most `max`. */
export function groupsOf(subject: Study, options: { minSupport?: number; max?: number } = {}): Group[] {
  const label = labeler(subject.examples);
  const byLabel = new Map<string, Example[]>();
  for (const example of subject.examples) byLabel.set(label(example), [...(byLabel.get(label(example)) ?? []), example]);
  const groups: Group[] = [];
  for (const [name, list] of byLabel) {
    const training = list.filter(example => example.split === 'training');
    if (training.length < (options.minSupport ?? 3)) continue;
    const ids = new Set(list.map(example => example.callId));
    groups.push({ id: '', label: name, training, heldOut: list.length - training.length,
      approaches: subject.approaches.filter(approach => approach.calls.some(id => ids.has(id))) });
  }
  groups.sort((a, b) => b.training.length - a.training.length);
  groups.forEach((group, index) => { group.id = `g${index + 1}`; });
  return groups.slice(0, options.max ?? 8);
}

export type Measure = { valid: boolean; error?: string; ofGroup: number; groupSize: number; others: number;
  counterexamples: { call: string; did: string; inputs: Record<string, unknown> }[] };

/** Exactly how a guard does on the training calls: how many of the group's it admits, and which other calls. */
export function measure(subject: Study, group: Group, guard: string): Measure {
  const compiled = compileGuard(guard);
  const groupSize = group.training.length;
  if (!compiled) return { valid: false, error: 'not a JavaScript expression over args', ofGroup: 0, groupSize, others: 0, counterexamples: [] };
  const label = labeler(subject.examples);
  const training = subject.examples.filter(example => example.split === 'training');
  const admitted = training.filter(example => compiled.test(example.args));
  const others = admitted.filter(example => label(example) !== group.label);
  return { valid: true, ofGroup: admitted.length - others.length, groupSize, others: others.length,
    counterexamples: others.slice(0, 5).map(example => ({ call: example.callId, did: label(example), inputs: example.args })) };
}

const json = (value: unknown) => JSON.stringify(value, null, 2);
const fence = (text: string, language = '') => `\`\`\`${language}\n${text}\n\`\`\``;
const short = (value: unknown, limit: number) => { const text = JSON.stringify(value) ?? 'undefined'; return text.length > limit ? `${text.slice(0, limit)}…` : text; };

/** The conditions worth starting from: those admitting most of the group and nothing else, then near misses. */
export function candidateConditions(subject: Study, group: Group, limit = 6): (Measure & { guard: string })[] {
  const measured = candidatePredicates(group.training).map(item => ({ guard: item.text, ...measure(subject, group, item.text) }))
    .filter(item => item.valid && item.ofGroup >= Math.min(3, group.training.length));
  const exact = measured.filter(item => item.others === 0).sort((a, b) => b.ofGroup - a.ofGroup || a.guard.length - b.guard.length);
  const near = measured.filter(item => item.others > 0 && item.others <= 2 && item.ofGroup > (exact[0]?.ofGroup ?? 0))
    .sort((a, b) => b.ofGroup - a.ofGroup || a.others - b.others);
  return [...exact.slice(0, limit), ...near.slice(0, 2)];
}

/** The writer's folder for one group: the function, the group, examples, contrasting calls, the last report. */
export function renderGroup(store: CallStore, subject: Study, group: Group, functionMd: string, extra: { report?: string } = {}): Record<string, string> {
  const files: Record<string, string> = { 'function.md': functionMd };
  const label = labeler(subject.examples);
  const results = new Map<string, number>();
  for (const example of group.training) {
    const record = subject.records.get(example.callId);
    const key = short(record ? store.value(record.output) : null, 200);
    results.set(key, (results.get(key) ?? 0) + 1);
  }
  const conditions = candidateConditions(subject, group);
  files['group.md'] = [`# Group ${group.id}`, '',
    `${group.training.length} recorded calls did this: ${group.label === 'none' ? 'no service call and no function call (an answer only)' : group.label}.`,
    `${group.heldOut} more are held out: the host uses them to check your case.`, '',
    '## Code the executor ran (normalized: $in.<path> is that input, $part.<path> a part of it, v0, v1 ... locals, $h0 ... literals that varied)', '',
    ...group.approaches.slice(0, 6).flatMap(approach => [`- ${approach.id}, ${approach.calls.filter(id => group.training.some(example => example.callId === id)).length} of these calls:`,
      approach.answerOnly ? '  (no code: the executor answered directly)' : fence(approach.template.join('\n;;\n').slice(0, 1200), 'ts')]), '',
    '## Results', '', ...[...results].sort((a, b) => b[1] - a[1]).slice(0, 8).map(([value, n]) => `- ${n}× ${value}`), '',
    '## Conditions to start from', '',
    'Measured on the training calls of all groups. A case\'s condition must admit no call of another group.', '',
    ...(conditions.length ? conditions.map(item => `- \`${item.guard}\`: ${item.ofGroup} of ${item.groupSize} calls of this group, ` +
      `${item.others} of other groups`) : ['No single test on the inputs separates this group. Look at the examples for what does.']), ''].join('\n');
  for (const example of group.training.slice(0, 6)) {
    const record = subject.records.get(example.callId);
    if (record) files[`examples/${example.callId.replace(/[^\w.-]/g, '_')}.json`] = json(exampleOf(store, record));
  }
  // Calls of other groups that look most like this one's: what the condition must not admit.
  const keys = (example: Example) => new Set(Object.entries(example.features).map(([key, value]) => `${key}=${value}`));
  const mine = group.training.map(keys);
  const contrast = subject.examples.filter(example => example.split === 'training' && label(example) !== group.label)
    .map(example => { const theirs = keys(example); return { example, score: Math.max(0, ...mine.map(set => [...set].filter(key => theirs.has(key)).length)) }; })
    .sort((a, b) => b.score - a.score).slice(0, 8);
  files['others.md'] = ['# Calls of other groups', '', 'The calls most like this group\'s that did something else. Your condition must not admit them.', '',
    ...contrast.map(({ example }) => {
      const record = subject.records.get(example.callId);
      return `- inputs ${short(example.args, 300)} → did ${label(example)}, result ${short(record ? store.value(record.output) : null, 160)}`;
    }), ''].join('\n');
  files['report.md'] = extra.report ?? 'No case has been checked yet for this group.\n';
  return files;
}

/**
 * One cases file from per-group case files. Each part has its imports and `export const when = ...; export const run =
 * ...;` (and any helpers it needs); a part becomes one array element, its own statements scoped inside it, so parts
 * cannot collide. Imports from the same module are merged. Returns the parts that do not have that shape.
 */
export function assembleCases(parts: readonly { id: string; text: string }[]): { text: string; included: string[]; errors: { id: string; error: string }[] } {
  const imports = new Map<string, { default?: string; named: Map<string, string>; namespace?: string }>();
  const elements: string[] = [], included: string[] = [], errors: { id: string; error: string }[] = [];
  for (const part of parts) {
    const file = ts.createSourceFile(`${part.id}.ts`, part.text, ts.ScriptTarget.ES2022, true, ts.ScriptKind.TS);
    const body: string[] = [];
    let when = false, run = false, problem: string | undefined;
    const partImports: [string, { default?: string; named: [string, string][]; namespace?: string }][] = [];
    for (const statement of file.statements) {
      if (ts.isImportDeclaration(statement) && ts.isStringLiteral(statement.moduleSpecifier)) {
        const clause = statement.importClause;
        if (clause?.isTypeOnly) continue;
        const entry: { default?: string; named: [string, string][]; namespace?: string } = { named: [] };
        if (clause?.name) entry.default = clause.name.text;
        const bindings = clause?.namedBindings;
        if (bindings && ts.isNamespaceImport(bindings)) entry.namespace = bindings.name.text;
        if (bindings && ts.isNamedImports(bindings))
          for (const element of bindings.elements) if (!element.isTypeOnly) entry.named.push([element.name.text, (element.propertyName ?? element.name).text]);
        partImports.push([statement.moduleSpecifier.text, entry]);
        continue;
      }
      let text = statement.getText(file);
      if (ts.isVariableStatement(statement) && statement.modifiers?.some(item => item.kind === ts.SyntaxKind.ExportKeyword)) {
        for (const declaration of statement.declarationList.declarations)
          if (ts.isIdentifier(declaration.name)) { if (declaration.name.text === 'when') when = true; if (declaration.name.text === 'run') run = true; }
        text = text.replace(/^export\s+/, '');
      } else if (ts.isFunctionDeclaration(statement) && statement.modifiers?.some(item => item.kind === ts.SyntaxKind.ExportKeyword)) {
        if (statement.name?.text === 'when') when = true;
        if (statement.name?.text === 'run') run = true;
        text = text.replace(/^export\s+(default\s+)?/, '');
      } else if (ts.isExportAssignment(statement) || ts.isExportDeclaration(statement)) { problem = 'export only `when` and `run`'; break; }
      body.push(text);
    }
    if (!problem && (!when || !run)) problem = 'case.ts must export `when` and `run`';
    for (const [module, entry] of partImports) {
      const merged = imports.get(module) ?? { named: new Map<string, string>() };
      if (entry.default) { if (merged.default && merged.default !== entry.default) problem ??= `${module} is imported under two default names`; merged.default = entry.default; }
      if (entry.namespace) { if (merged.namespace && merged.namespace !== entry.namespace) problem ??= `${module} is imported under two namespace names`; merged.namespace = entry.namespace; }
      for (const [local, name] of entry.named) {
        if (merged.named.has(local) && merged.named.get(local) !== name) problem ??= `${local} names two different imports`;
        merged.named.set(local, name);
      }
      if (!problem) imports.set(module, merged);
    }
    if (problem) { errors.push({ id: part.id, error: problem }); continue; }
    elements.push(`  // ${part.id}\n  (() => {\n${body.map(line => line.replace(/^/gm, '    ')).join('\n')}\n    return { when, run };\n  })(),`);
    included.push(part.id);
  }
  const head = [...imports].map(([module, entry]) => {
    const pieces = [entry.default, entry.namespace ? `* as ${entry.namespace}` : undefined,
      entry.named.size ? `{ ${[...entry.named].map(([local, name]) => local === name ? local : `${name} as ${local}`).join(', ')} }` : undefined].filter(Boolean);
    return pieces.length ? `import ${pieces.join(', ')} from ${JSON.stringify(module)};` : `import ${JSON.stringify(module)};`;
  });
  return { text: `${head.join('\n')}${head.length ? '\n\n' : ''}export const cases = [\n${elements.join('\n')}\n];\n`, included, errors };
}

export { splitOf };
