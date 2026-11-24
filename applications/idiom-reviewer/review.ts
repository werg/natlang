/**
 * Semantic idiom reviewer (WS3 I2, WS6 §2 Tier 0 flagship; plans/semantic-fp/03 §3.1).
 * The crisp floor (`natlang check`, ts-host/src/compiler/idiom.ts) reports what is exact. This reviewer judges the rest:
 * what each instruction step asks for (`stepKind.nl`), whether a function as a whole is a prose interpreter, cue-word
 * rules, and external text typed as trusted. Every decision is a `readout: decision` function, escalated when unsure.
 *
 * The step judgment (`stepKind`) and the step splitter (`instructionsOf`) are `@natlang/std`'s, shared with the program
 * migrator (applications/migration/program).
 */
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { escalate, instructionsOf, semanticMap, stepKind } from '@natlang/node/std';
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
  /** Optional program root directory used to locate sibling `types.ts` files for alias resolution. */
  rootDir?: string;
  /** Optional in-memory `types.ts` source strings keyed by path for resolving `Untrusted<...>` inside type aliases. */
  typeSources?: Record<string, string>;
};

/** Parse `type Name = ...` and `interface Name { ... }` bodies from TypeScript source text. */
export function parseTypeAliases(source: string): Map<string, string> {
  const aliases = new Map<string, string>();
  const cleaned = source
    .replace(/\/\*[\s\S]*?\*\//g, ' ')
    .replace(/\/\/[^\n]*/g, ' ');

  const declRe = /\b(?:export\s+)?(?:type\s+([A-Za-z_$][\w$]*)\s*(?:<[^>]*>)?\s*=|interface\s+([A-Za-z_$][\w$]*)\s*(?:<[^>]*>)?(?:\s+extends\s+[^{]+)?\s*\{)/g;
  for (const match of cleaned.matchAll(declRe)) {
    const name = match[1] ?? match[2];
    if (!name) continue;
    const isInterface = Boolean(match[2]);
    let pos = (match.index ?? 0) + match[0].length;
    if (isInterface) {
      let depth = 1;
      const start = pos;
      for (; pos < cleaned.length && depth > 0; pos++) {
        if (cleaned[pos] === '{') depth++;
        else if (cleaned[pos] === '}') depth--;
      }
      aliases.set(name, cleaned.slice(start, Math.max(start, pos - 1)));
    } else {
      let brace = 0;
      let paren = 0;
      let angle = 0;
      const start = pos;
      for (; pos < cleaned.length; pos++) {
        const ch = cleaned[pos]!;
        if (ch === '{') brace++;
        else if (ch === '}') brace = Math.max(0, brace - 1);
        else if (ch === '(') paren++;
        else if (ch === ')') paren = Math.max(0, paren - 1);
        else if (ch === '<') angle++;
        else if (ch === '>') angle = Math.max(0, angle - 1);
        else if (ch === ';' && brace === 0 && paren === 0 && angle === 0) break;
      }
      aliases.set(name, cleaned.slice(start, pos));
    }
  }
  return aliases;
}

/** Whether `typeExpr` directly or transitively contains `Untrusted<...>` via parsed `types.ts` aliases. */
export function typeContainsUntrusted(
  typeExpr: string,
  aliases: ReadonlyMap<string, string>,
  seen = new Set<string>(),
): boolean {
  if (/\bUntrusted\s*</.test(typeExpr)) return true;
  const identifiers = typeExpr.match(/\b[A-Za-z_$][\w$]*\b/g) ?? [];
  for (const ident of identifiers) {
    if (seen.has(ident)) continue;
    const body = aliases.get(ident);
    if (!body) continue;
    seen.add(ident);
    if (typeContainsUntrusted(body, aliases, seen)) return true;
  }
  return false;
}

function collectTypeAliases(files: readonly ProgramFile[], options: ReviewOptions): Map<string, string> {
  const merged = new Map<string, string>();
  const ingest = (source: string) => {
    for (const [name, body] of parseTypeAliases(source)) {
      if (!merged.has(name)) merged.set(name, body);
    }
  };

  if (options.typeSources) {
    for (const source of Object.values(options.typeSources)) ingest(source);
  }
  for (const file of files) {
    if (file.path.endsWith('.ts')) ingest(file.content);
    if (file.typesContent) ingest(file.typesContent);
  }
  if (options.rootDir) {
    const root = resolve(options.rootDir);
    const checkedDirs = new Set<string>();
    const dirsToCheck = new Set<string>([root]);
    for (const file of files) {
      let current = resolve(root, dirname(file.path));
      for (let guard = 0; guard < 16; guard++) {
        dirsToCheck.add(current);
        if (current === root) break;
        const parent = dirname(current);
        if (parent === current) break;
        current = parent;
      }
    }
    for (const dir of dirsToCheck) {
      if (checkedDirs.has(dir)) continue;
      checkedDirs.add(dir);
      const typesPath = join(dir, 'types.ts');
      if (existsSync(typesPath)) {
        try {
          ingest(readFileSync(typesPath, 'utf8'));
        } catch {
          // Ignore unreadable sibling types.ts
        }
      }
    }
  }
  return merged;
}

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
    const { front, body, declaration, steps } = instructionsOf(file.content);
    return { file, front, body: body.trim(), signature: declaration, steps };
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

  const typeAliases = collectTypeAliases(files, options);
  const params = parsed.flatMap(entry => {
    const entryAliases = entry.file.typesContent
      ? new Map([...typeAliases, ...parseTypeAliases(entry.file.typesContent)])
      : typeAliases;
    return Object.entries(argsOf(entry.front))
      .filter(([, type]) => !typeContainsUntrusted(type, entryAliases))
      .map(([name, type]) => ({ file: entry.file.path, name, type, body: entry.body }));
  });
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
