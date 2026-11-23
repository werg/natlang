/**
 * Semantic idiom reviewer (WS3 I2, WS6 §2 Tier 0 flagship).
 * Exact splitting and frontmatter parsing stay in TypeScript; semantic classification of sentence groups,
 * cue-word shortcuts, and untyped external inputs runs concurrently through calibrated `readout: decision`
 * functions with `escalate` to a stronger model on uncertain cases.
 */
import { escalate, semanticMap } from '@natlang/node/std';
import classifyGroup from './classifyGroup.nl';
import detectCueWords from './detectCueWords.nl';
import detectUntypedExternal from './detectUntypedExternal.nl';
import type { IdiomIssue, IdiomReport, ProgramFile, SentenceGroup } from './types.js';

export type ReviewOptions = {
  /** Model name to escalate uncertain idiom decisions to (default `'teacher'`). */
  escalateTo?: string;
  /** Minimum confidence to settle locally without escalating (default `0.78`). */
  floor?: number;
  /** Maximum concurrent decision calls (default `8`). */
  concurrency?: number;
};

type ParsedNl = {
  args: Record<string, string>;
  body: string;
};

/** Parse simple YAML frontmatter `args:` and body from a `.nl` file without touching fenced code blocks. */
export function parseNlSource(content: string): ParsedNl {
  const normalized = content.replace(/\r\n/g, '\n');
  const match = /^---\n([\s\S]*?)\n---\n?([\s\S]*)$/.exec(normalized);
  if (!match) return { args: {}, body: normalized.trim() };
  const [, front, body] = match;
  const args: Record<string, string> = {};
  const lines = front!.split('\n');
  let inArgs = false;
  for (const line of lines) {
    if (/^args:\s*\{(.+)\}\s*$/.test(line)) {
      const inner = /^args:\s*\{(.+)\}\s*$/.exec(line)![1]!;
      for (const part of inner.split(',')) {
        const kv = /^\s*([\w$?]+)\s*:\s*(.+?)\s*$/.exec(part);
        if (kv) args[kv[1]!.replace(/\?$/, '')] = kv[2]!;
      }
      inArgs = false;
      continue;
    }
    if (/^args:\s*$/.test(line)) { inArgs = true; continue; }
    if (inArgs) {
      if (/^\S/.test(line)) { inArgs = false; continue; }
      const kv = /^\s+([\w$?]+)\s*:\s*(.+?)\s*$/.exec(line);
      if (kv) args[kv[1]!.replace(/\?$/, '')] = kv[2]!;
    }
  }
  return { args, body: body!.trim() };
}

/** Split a `.nl` body into logical sentence/step groups while stripping fenced code blocks. */
export function splitSentenceGroups(file: string, body: string): SentenceGroup[] {
  const withoutFences = body.replace(/```[\s\S]*?```/g, '').trim();
  if (!withoutFences) return [];
  const raw = withoutFences.split(/\n\s*\n+/).map(s => s.trim()).filter(Boolean);
  return raw.map((text, index) => ({ file, index, text }));
}

export async function reviewProgramIdioms(files: ProgramFile[], options: ReviewOptions = {}): Promise<IdiomReport> {
  const nlFiles = files.filter(f => f.path.endsWith('.nl'));
  const to = options.escalateTo ?? 'teacher';
  const floor = options.floor ?? 0.78;
  const concurrency = options.concurrency ?? 8;

  const judgeGroup = escalate(classifyGroup, { to, floor });
  const judgeCue = escalate(detectCueWords, { to, floor });
  const judgeExternal = escalate(detectUntypedExternal, { to, floor });

  const parsed = nlFiles.map(file => ({ file, ...parseNlSource(file.content) }));
  const groups = parsed.flatMap(entry => splitSentenceGroups(entry.file.path, entry.body));
  const issues: IdiomIssue[] = [];

  const groupResults = await semanticMap(groups, async group => ({
    group,
    decision: await judgeGroup(group),
  }), { concurrency });

  for (const { group, decision } of groupResults) {
    if (decision.value === 'mechanism') {
      issues.push({
        file: group.file,
        kind: 'mechanism-in-prose',
        groupIndex: group.index,
        excerpt: group.text.slice(0, 140),
        confidence: decision.confidence,
        escalated: decision.escalated,
        suggestion: 'Move this deterministic step into a TypeScript helper or host module and call the `.nl` function only for the judgment.',
      });
    } else if (decision.value === 'prose-interpreter') {
      issues.push({
        file: group.file,
        kind: 'prose-interpreter',
        groupIndex: group.index,
        excerpt: group.text.slice(0, 140),
        confidence: decision.confidence,
        escalated: decision.escalated,
        suggestion: 'Replace the imperative English algorithm with TypeScript control flow (`semanticMap`, `semanticRepair`, or `iterateOn`) calling small typed `.nl` leaves.',
      });
    }
  }

  const cueResults = await semanticMap(parsed.filter(p => p.body.length > 0), async entry => ({
    file: entry.file.path,
    body: entry.body,
    decision: await judgeCue(entry.body),
  }), { concurrency });

  for (const { file, body, decision } of cueResults) {
    if (decision.value) {
      issues.push({
        file,
        kind: 'cue-word-rule',
        excerpt: body.slice(0, 140),
        confidence: decision.confidence,
        escalated: decision.escalated,
        suggestion: 'Replace keyword/cue-word rules with either a crisp TypeScript matcher (`pluggable`) or genuine meaning-based criteria.',
      });
    }
  }

  const paramChecks = parsed.flatMap(entry =>
    Object.entries(entry.args)
      .filter(([, type]) => !/\bUntrusted\s*</.test(type))
      .map(([paramName, paramType]) => ({ file: entry.file.path, paramName, paramType, body: entry.body }))
  );

  const extResults = await semanticMap(paramChecks, async item => ({
    item,
    decision: await judgeExternal(item.paramName, item.paramType, item.body),
  }), { concurrency });

  for (const { item, decision } of extResults) {
    if (decision.value) {
      issues.push({
        file: item.file,
        kind: 'untyped-external-input',
        excerpt: `${item.paramName}: ${item.paramType}`,
        confidence: decision.confidence,
        escalated: decision.escalated,
        suggestion: `Type \`${item.paramName}\` as \`Untrusted<${item.paramType}>\` so untrusted external content is fenced and blocked from prompt interpolation.`,
      });
    }
  }

  return {
    filesReviewed: nlFiles.length,
    groupsReviewed: groups.length,
    clean: issues.length === 0,
    issues,
  };
}
