/**
 * Crisp mining for the specializer (§5): approach groups with anti-unified templates, candidate guards induced over
 * input features, and their exact precision and coverage. Guards are measured by evaluating the very expression text a
 * cases file would contain, so what the specializer reads in conditions.md is what a guard will do.
 */
import ts from 'typescript';
import { hexDigest } from '../native/hash.js';
import { inputLeaves, normalizeProgram } from './normalize.js';
import { VALUE_CLASSES, tokens } from './snapshot.js';

/** One recorded call as mining sees it. */
export type Example = { callId: string; args: Record<string, unknown>; features: Record<string, string | number | boolean>;
  evals: string[]; approach: string; split: 'training' | 'held-out';
  /** What the call did, whatever code it ran: its service calls and the functions it called, in order (`none` when nothing). */
  behavior?: string };

/** A behavior label from the service calls and child functions of a call, consecutive repeats folded. */
export function behaviorLabel(steps: readonly string[]): string {
  const folded = steps.filter((step, index) => step !== steps[index - 1]);
  return folded.length ? folded.join(' > ') : 'none';
}

/** Calls are split by a hash of their ID: a fifth held out, stably across runs. */
export const splitOf = (callId: string): 'training' | 'held-out' => parseInt(hexDigest(callId).slice(0, 4), 16) % 5 === 0 ? 'held-out' : 'training';

// --- Approaches -----------------------------------------------------------------------------------------------------

export type Approach = { id: string; calls: string[]; answerOnly: boolean;
  /** The normalized programs with differing literals replaced by holes `$h0`, `$h1`, ... */
  template: string[];
  /** Per call, the literal each hole had. */
  holes: Record<string, string[]> };

const LITERALS = new Set([ts.SyntaxKind.StringLiteral, ts.SyntaxKind.NumericLiteral, ts.SyntaxKind.NoSubstitutionTemplateLiteral]);
function scan(text: string): { kind: ts.SyntaxKind; text: string }[] {
  const scanner = ts.createScanner(ts.ScriptTarget.ES2022, true, ts.LanguageVariant.Standard, text);
  const out: { kind: ts.SyntaxKind; text: string }[] = [];
  for (let kind = scanner.scan(); kind !== ts.SyntaxKind.EndOfFileToken; kind = scanner.scan()) out.push({ kind, text: scanner.getTokenText() });
  return out;
}

/**
 * Group calls by approach: identical normalized programs first, then groups whose programs differ only in literals
 * (anti-unified into one template with holes).
 */
export function approaches(examples: readonly Example[]): Approach[] {
  const normalized = new Map(examples.map(example => {
    const leaves = inputLeaves(example.args);
    return [example.callId, example.evals.map(code => normalizeProgram(code, example.args, leaves))];
  }));
  const shapes = new Map<string, Example[]>();
  for (const example of examples) {
    const programs = normalized.get(example.callId)!;
    const shape = programs.length ? programs.map(program => scan(program).map(token => LITERALS.has(token.kind) ? '$lit' : token.text).join(' ')).join('\n;;\n') : '';
    const list = shapes.get(shape) ?? [];
    list.push(example);
    shapes.set(shape, list);
  }
  const result: Approach[] = [];
  for (const [shape, group] of shapes) {
    const programs = group.map(example => normalized.get(example.callId)!.map(scan));
    const template: string[] = [], holes: Record<string, string[]> = Object.fromEntries(group.map(example => [example.callId, []]));
    let hole = 0;
    (programs[0] ?? []).forEach((first, programIndex) => {
      template.push(first.map((token, tokenIndex) => {
        const values = programs.map(program => program[programIndex]?.[tokenIndex]?.text ?? '');
        if (!LITERALS.has(token.kind) || values.every(value => value === token.text)) return token.text;
        const name = `$h${hole++}`;
        group.forEach((example, index) => holes[example.callId]!.push(values[index]!));
        return name;
      }).join(' '));
    });
    result.push({ id: '', calls: group.map(example => example.callId), answerOnly: shape === '', template, holes });
  }
  result.sort((a, b) => b.calls.length - a.calls.length);
  result.forEach((approach, index) => { approach.id = approach.answerOnly ? `answer-only` : `a${index + 1}`; });
  return result;
}

// --- Guards ---------------------------------------------------------------------------------------------------------

/** A TypeScript accessor for a feature path (`items[0].kind` → `args.items?.[0]?.kind`). */
export function accessor(path: string): string {
  const match = /^([A-Za-z_$][\w$]*)(.*)$/s.exec(path);
  if (!match) return `args[${JSON.stringify(path)}]`;
  return `args.${match[1]}${match[2]!.replace(/\.(?=[A-Za-z_$])/g, '?.').replace(/\[/g, '?.[')}`;
}
const regexText = (pattern: RegExp) => `/${pattern.source}/${pattern.flags}`;
const escapeRegex = (text: string) => text.replace(/[.*+?^${}()|[\]\\/]/g, '\\$&');

/** An atomic condition: its TypeScript text and how it is evaluated. */
export type Predicate = { text: string; test: (args: Record<string, unknown>) => boolean };

/** A guard expression over `args`, compiled once; a guard that throws answers false. */
export function compileGuard(text: string): Predicate | undefined { return predicate(text); }
function predicate(text: string): Predicate | undefined {
  try {
    const fn = new Function('args', `"use strict"; return (${text});`) as (args: Record<string, unknown>) => unknown;
    return { text, test: args => { try { return fn(args) === true; } catch { return false; } } };
  } catch { return undefined; }
}

/** Candidate conditions from the features present in the examples. */
export function candidatePredicates(examples: readonly Example[]): Predicate[] {
  const texts = new Set<string>();
  const numbers = new Map<string, Set<number>>();
  for (const example of examples) for (const [key, value] of Object.entries(example.features)) {
    const [family, ...rest] = key.split(':');
    const path = family === 'tok' || family === 'cls' ? rest.slice(0, -1).join(':') : rest.join(':');
    const at = accessor(path);
    if (family === 'str') texts.add(`${at} === ${JSON.stringify(value)}`);
    else if (family === 'tok') {
      const word = rest.at(-1)!;
      texts.add(/^[\w]+$/.test(word) ? `typeof ${at} === 'string' && /\\b${escapeRegex(word)}\\b/i.test(${at})` :
        `typeof ${at} === 'string' && /(^|[^\\p{L}\\p{N}_])${escapeRegex(word)}($|[^\\p{L}\\p{N}_])/iu.test(${at})`);
    } else if (family === 'cls') {
      const pattern = VALUE_CLASSES[rest.at(-1)!];
      if (pattern) texts.add(`typeof ${at} === 'string' && ${regexText(pattern)}.test(${at})`);
    } else if (family === 'bool') texts.add(`${at} === ${value}`);
    else if (family === 'null') texts.add(`${at} == null`);
    else if (family === 'num' || family === 'len' || family === 'arr') {
      const target = family === 'num' ? at : `${at}?.length`;
      const set = numbers.get(target) ?? new Set<number>();
      set.add(Number(value));
      numbers.set(target, set);
      if (family === 'arr') { texts.add(`${at}?.length === 0`); texts.add(`(${at}?.length ?? 0) > 0`); }
    }
  }
  for (const [target, values] of numbers) {
    const sorted = [...values].sort((a, b) => a - b);
    const step = Math.max(1, Math.floor(sorted.length / 12));
    for (let index = 0; index < sorted.length; index += step) {
      const value = sorted[index]!;
      texts.add(`typeof ${target} === 'number' && ${target} <= ${value}`);
      texts.add(`typeof ${target} === 'number' && ${target} >= ${value}`);
    }
  }
  return [...texts].map(predicate).filter((item): item is Predicate => !!item);
}

/** One rule of an induced decision list: a guard that selects one label (a behavior) with no counterexample in training. */
export type Rule = { label: string; guard: string; covers: string[]; heldOut: { covered: number; correct: number } };
const labelOf = (example: Example): string => example.behavior ?? example.approach;

/**
 * Greedy decision-list induction (§5.2): repeatedly take the condition, or pair of conditions, with the widest coverage
 * among the calls not yet covered that selects a single behavior (else approach) without exception, until nothing covers `minSupport`
 * calls. Calls no rule covers are left to the agent. Held-out calls are only measured.
 */
export function induceRules(examples: readonly Example[], options: { minSupport?: number; maxRules?: number; maxPairs?: number } = {}):
  { rules: Rule[]; unclassified: string[] } {
  const minSupport = options.minSupport ?? 3;
  // One behavior for every call says nothing about which inputs select it: tell approaches apart instead.
  if (new Set(examples.map(example => example.behavior)).size < 2) examples = examples.map(({ behavior: _behavior, ...example }) => example);
  const training = examples.filter(example => example.split === 'training');
  const heldOut = examples.filter(example => example.split === 'held-out');
  const predicates = candidatePredicates(training);
  const truth = new Map(predicates.map(item => [item.text, new Set(training.filter(example => item.test(example.args)).map(example => example.callId))]));
  const label = new Map(training.map(example => [example.callId, labelOf(example)]));
  let remaining = new Set(training.map(example => example.callId));
  const rules: Rule[] = [];
  const pure = (covered: string[]): string | undefined => {
    const labels = new Set(covered.map(id => label.get(id)));
    return labels.size === 1 ? [...labels][0] : undefined;
  };
  while (rules.length < (options.maxRules ?? 12)) {
    let best: { guard: string; covered: string[]; approach: string } | undefined;
    const singles = predicates.map(item => ({ item, covered: [...truth.get(item.text)!].filter(id => remaining.has(id)) }))
      .filter(entry => entry.covered.length >= minSupport);
    for (const { item, covered } of singles) {
      const approach = pure(covered);
      if (approach && (!best || covered.length > best.covered.length || (covered.length === best.covered.length && item.text.length < best.guard.length)))
        best = { guard: item.text, covered, approach };
    }
    // Pairs among the widest impure conditions, when no single condition is pure enough.
    const wide = singles.filter(entry => !pure(entry.covered)).sort((a, b) => b.covered.length - a.covered.length).slice(0, options.maxPairs ?? 40);
    for (let i = 0; i < wide.length; i++) for (let j = i + 1; j < wide.length; j++) {
      const second = new Set(wide[j]!.covered);
      const covered = wide[i]!.covered.filter(id => second.has(id));
      if (covered.length < minSupport || (best && covered.length <= best.covered.length)) continue;
      const approach = pure(covered);
      if (approach) best = { guard: `(${wide[i]!.item.text}) && (${wide[j]!.item.text})`, covered, approach };
    }
    if (!best) break;
    const chosen = predicate(best.guard)!;
    const earlier = rules.map(rule => predicate(rule.guard)!);
    const heldCovered = heldOut.filter(example => chosen.test(example.args) && !earlier.some(rule => rule.test(example.args)));
    rules.push({ label: best.approach, guard: best.guard, covers: best.covered,
      heldOut: { covered: heldCovered.length, correct: heldCovered.filter(example => labelOf(example) === best!.approach).length } });
    remaining = new Set([...remaining].filter(id => !best!.covered.includes(id)));
  }
  return { rules, unclassified: [...remaining] };
}

/** Exact precision and coverage of a guard expression over examples (for proposals from the model). */
export function measureGuard(guard: string, label: string, examples: readonly Example[]):
  { valid: boolean; error?: string; covered: number; correct: number; counterexamples: string[] } {
  const item = predicate(guard);
  if (!item) return { valid: false, error: 'the guard is not a JavaScript expression over args', covered: 0, correct: 0, counterexamples: [] };
  const covered = examples.filter(example => item.test(example.args));
  return { valid: true, covered: covered.length, correct: covered.filter(example => labelOf(example) === label).length,
    counterexamples: covered.filter(example => labelOf(example) !== label).map(example => example.callId) };
}

export { tokens };
