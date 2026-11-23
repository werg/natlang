/**
 * Semantic idiom reviewer (WS3 I2, WS6 §2 Tier 0 flagship; plans/semantic-fp/03 §3.1).
 * The crisp floor (`natlang check`, ts-host/src/compiler/idiom.ts) reports what is exact. This reviewer judges the rest:
 * what each instruction step asks for (`stepKind.nl`), whether a function as a whole is a prose interpreter, cue-word
 * rules, and external text typed as trusted. Every decision is a `readout: decision` function, escalated when unsure.
 *
 * `stepKind.nl`, `splitSource` and `stepsOf` are the migrator's (applications/migration/program): the same contract,
 * copied because a package cannot import another package's sources yet. Plan 03 §3.1 wants one implementation; the
 * shared home is `@natlang/std` once both tools move to it.
 */
import { escalate, semanticMap } from '@natlang/node/std';
import stepKind from './stepKind.nl';
import detectCueWords from './detectCueWords.nl';
import detectUntypedExternal from './detectUntypedExternal.nl';
import isProseInterpreter from './isProseInterpreter.nl';
import type { IdiomIssue, IdiomReport, KindedStep, ProgramFile } from './types.js';

export type ReviewOptions = {
  /** Model name to escalate uncertain idiom decisions to (default `'teacher'`). */
  escalateTo?: string;
  /** Minimum confidence to settle locally without escalating (default `0.78`). */
  floor?: number;
  /** Maximum concurrent decision calls (default `8`). */
  concurrency?: number;
};

/** The front matter and body of a `.nl` source (applications/migration/program/host.ts `splitSource`). */
export function splitSource(text: string): { front: string; body: string } {
  const match = /^---\n([\s\S]*?)\n---\n?([\s\S]*)$/.exec(text.replace(/\r\n/g, '\n'));
  if (!match) throw new Error('a .nl file starts with front matter between --- lines');
  return { front: match[1]!, body: match[2]! };
}

/** The body's steps: its numbered items when it has them, else its paragraphs; a fenced block stays in its step
 * (applications/migration/program/host.ts `stepsOf`). */
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

/** The declaration the step classifier reads: the front matter without `readout:` (as the migrator passes it). */
const declarationOf = (front: string) => front.split('\n').filter(line => !/^readout:/.test(line)).join('\n').trim();

/** The `args:` of a front matter, by name (block form `args:\n  name: Type` or flow form `args: { name: Type }`). */
export function argsOf(front: string): Record<string, string> {
  const args: Record<string, string> = {};
  const flow = /^args:\s*\{(.+)\}\s*$/m.exec(front);
  if (flow) {
    for (const part of flow[1]!.split(',')) {
      const kv = /^\s*([\w$]+)\??\s*:\s*(.+?)\s*$/.exec(part);
      if (kv) args[kv[1]!] = kv[2]!;
    }
    return args;
  }
  const lines = front.split('\n');
  const start = lines.findIndex(line => /^args:\s*$/.test(line));
  if (start < 0) return args;
  for (const line of lines.slice(start + 1)) {
    if (/^\S/.test(line)) break;
    const kv = /^\s+([\w$]+)\??\s*:\s*(.+?)\s*$/.exec(line);
    if (kv) args[kv[1]!] = kv[2]!;
  }
  return args;
}

const STEP_ISSUES = {
  mechanism: ['mechanism-in-prose', 'Move this step into a TypeScript helper and keep the `.nl` function for the judgment.'],
  'guard-for-type': ['guard-for-type', 'Delete this step: the signature already guarantees it.'],
  redundant: ['redundant', 'Delete this step: another step already asks for it.'],
} as const;

export async function reviewProgramIdioms(files: ProgramFile[], options: ReviewOptions = {}): Promise<IdiomReport> {
  const to = options.escalateTo ?? 'teacher';
  const floor = options.floor ?? 0.78;
  const concurrency = options.concurrency ?? 8;
  const judgeStep = escalate(stepKind, { to, floor });
  const judgeInterpreter = escalate(isProseInterpreter, { to, floor });
  const judgeCue = escalate(detectCueWords, { to, floor });
  const judgeExternal = escalate(detectUntypedExternal, { to, floor });

  const parsed = files.filter(f => f.path.endsWith('.nl')).map(file => {
    const { front, body } = splitSource(file.content);
    return { file, front, body: body.trim(), signature: declarationOf(front), steps: stepsOf(body) };
  });
  const issues: IdiomIssue[] = [];

  // What each step asks for, then whether the function as a whole is a prose interpreter.
  const kinded = await semanticMap(parsed, async entry => {
    const decisions = await semanticMap(entry.steps, step => judgeStep(step, entry.signature, entry.file.path), { concurrency });
    return { entry, decisions };
  }, { concurrency: 1 });
  for (const { entry, decisions } of kinded) {
    decisions.forEach((decision, index) => {
      const issue = decision.value === 'judgment' ? undefined : STEP_ISSUES[decision.value];
      if (issue) issues.push({ file: entry.file.path, kind: issue[0], stepIndex: index, excerpt: entry.steps[index]!.slice(0, 140),
        confidence: decision.confidence, escalated: decision.escalated, suggestion: issue[1] });
    });
  }
  const multiStep = kinded.filter(({ decisions }) => decisions.length > 1);
  const interpreters = await semanticMap(multiStep, async ({ entry, decisions }) => {
    const steps: KindedStep[] = entry.steps.map((text, index) => ({ text, kind: decisions[index]!.value }));
    return { entry, decision: await judgeInterpreter(entry.file.path, steps, entry.file.hasHelpers ?? false) };
  }, { concurrency });
  for (const { entry, decision } of interpreters) {
    if (decision.value) issues.push({ file: entry.file.path, kind: 'prose-interpreter', excerpt: entry.body.slice(0, 140),
      confidence: decision.confidence, escalated: decision.escalated,
      suggestion: 'Write the algorithm as TypeScript control flow (`semanticMap`, `semanticRepair`, `iterateOn`) calling small typed `.nl` judgments.' });
  }

  const cues = await semanticMap(parsed.filter(p => p.body.length > 0), async entry => ({ entry, decision: await judgeCue(entry.body) }), { concurrency });
  for (const { entry, decision } of cues) {
    if (decision.value) issues.push({ file: entry.file.path, kind: 'cue-word-rule', excerpt: entry.body.slice(0, 140),
      confidence: decision.confidence, escalated: decision.escalated,
      suggestion: 'State what the judgment means instead of trigger words, or make the exact rule a crisp `pluggable` side.' });
  }

  const params = parsed.flatMap(entry => Object.entries(argsOf(entry.front))
    .filter(([, type]) => !/\bUntrusted\s*</.test(type))
    .map(([name, type]) => ({ file: entry.file.path, name, type, body: entry.body })));
  const externals = await semanticMap(params, async param => ({ param, decision: await judgeExternal(param.name, param.type, param.body) }), { concurrency });
  for (const { param, decision } of externals) {
    if (decision.value) issues.push({ file: param.file, kind: 'untyped-external-input', excerpt: `${param.name}: ${param.type}`,
      confidence: decision.confidence, escalated: decision.escalated,
      suggestion: `Type \`${param.name}\` as \`Untrusted<${param.type}>\` so its text reaches the model fenced as data.` });
  }

  return {
    filesReviewed: parsed.length,
    stepsReviewed: parsed.reduce((sum, entry) => sum + entry.steps.length, 0),
    clean: issues.length === 0,
    issues,
  };
}
