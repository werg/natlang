/**
 * Semantic Merge (WS6 Tier 2 flagship A6):
 * - Exact crisp line-level 3-way merge resolves all non-overlapping hunks with 0 model calls.
 * - Structured JSON/config conflicts merge key-by-key via `semanticMerge`.
 * - Overlapping text/code conflicts check intent compatibility via `intentsCompatible.nl` (`readout: decision`),
 *   then synthesize and verify merged hunks via `mergeHunk.nl` (`ensures:`) inside `semanticRepair` against
 *   the repository's exact verifier callback.
 * - Genuinely incompatible intents (or unverified repairs) emit explicit `<<<<<<< / ======= / >>>>>>>` conflict
 *   markers instead of silently dropping either side.
 */
import { decide } from '@natlang/node';
import { semanticMerge, semanticRepair } from '@natlang/node/std';
import intentsCompatible from './intentsCompatible.nl';
import mergeHunk from './mergeHunk.nl';
import type { ConflictHunk, ConflictResolution, MergeOutcome, ThreeWayFile } from './types.js';

export type MergeOptions = {
  /** Minimum compatibility probability required to attempt semantic hunk synthesis (default `0.7`). */
  compatibilityFloor?: number;
  /** Maximum `semanticRepair` fix attempts per compatible conflict hunk (default `2`). */
  repairAttempts?: number;
  /** Optional repository verifier callback over the candidate merged file content (returns problem strings; empty = pass). */
  verify?: (mergedContent: string, path: string) => Promise<readonly string[]> | readonly string[];
  /** Labels for explicit conflict markers when intents are incompatible (default `'left'` / `'right'`). */
  leftLabel?: string;
  rightLabel?: string;
};

type EditSpan = {
  start: number;
  end: number;
  lines: string[];
};

type Chunk =
  | { kind: 'clean'; lines: string[] }
  | { kind: 'conflict'; startLine: number; endLine: number; base: string[]; left: string[]; right: string[] };

function splitNormalized(text: string): { lines: string[]; trailingNewline: boolean } {
  const norm = text.replace(/\r\n/g, '\n');
  if (norm.length === 0) return { lines: [], trailingNewline: false };
  const trailingNewline = norm.endsWith('\n');
  const body = trailingNewline ? norm.slice(0, -1) : norm;
  return { lines: body.length === 0 ? [] : body.split('\n'), trailingNewline };
}

function joinNormalized(lines: string[], trailingNewline: boolean): string {
  if (lines.length === 0) return '';
  return lines.join('\n') + (trailingNewline ? '\n' : '');
}

/** Exact LCS diff producing minimal ordered edit spans against `base`. */
function diffToEdits(base: readonly string[], target: readonly string[]): EditSpan[] {
  const m = base.length;
  const n = target.length;
  const dp: number[][] = Array.from({ length: m + 1 }, () => new Array<number>(n + 1).fill(0));
  for (let i = m - 1; i >= 0; i--) {
    for (let j = n - 1; j >= 0; j--) {
      dp[i]![j] = base[i] === target[j]
        ? dp[i + 1]![j + 1]! + 1
        : Math.max(dp[i + 1]![j]!, dp[i]![j + 1]!);
    }
  }

  const matches: Array<[number, number]> = [];
  let i = 0;
  let j = 0;
  for (let step = 0; step <= m + n && i < m && j < n; step++) {
    if (base[i] === target[j]) {
      matches.push([i, j]);
      i++;
      j++;
    } else if (dp[i + 1]![j]! >= dp[i]![j + 1]!) {
      i++;
    } else {
      j++;
    }
  }
  matches.push([m, n]);

  const edits: EditSpan[] = [];
  let prevB = 0;
  let prevT = 0;
  for (const [mb, mt] of matches) {
    if (mb > prevB || mt > prevT) {
      edits.push({ start: prevB, end: mb, lines: target.slice(prevT, mt) });
    }
    prevB = mb + 1;
    prevT = mt + 1;
  }
  return edits;
}

function spansOverlap(a: EditSpan, b: EditSpan): boolean {
  if (a.start === a.end && b.start === b.end) return a.start === b.start;
  if (a.start === a.end) return a.start >= b.start && a.start <= b.end;
  if (b.start === b.end) return b.start >= a.start && b.start <= a.end;
  return a.start < b.end && b.start < a.end;
}

function applySpansInRange(base: readonly string[], rangeStart: number, rangeEnd: number, spans: readonly EditSpan[]): string[] {
  const out: string[] = [];
  let cursor = rangeStart;
  for (const span of spans) {
    if (span.start > cursor) out.push(...base.slice(cursor, span.start));
    out.push(...span.lines);
    cursor = span.end;
  }
  if (cursor < rangeEnd) out.push(...base.slice(cursor, rangeEnd));
  return out;
}

function arraysEqual(a: readonly string[], b: readonly string[]): boolean {
  return a.length === b.length && a.every((item, idx) => item === b[idx]);
}

/** Exact deterministic 3-way line merge decomposing a file into clean chunks and overlapping conflicts. */
export function crispThreeWayChunks(file: ThreeWayFile): { chunks: Chunk[]; trailingNewline: boolean } {
  const b = splitNormalized(file.base);
  const l = splitNormalized(file.left);
  const r = splitNormalized(file.right);
  const trailingNewline = l.trailingNewline || r.trailingNewline || b.trailingNewline;

  const leftEdits = diffToEdits(b.lines, l.lines);
  const rightEdits = diffToEdits(b.lines, r.lines);

  const chunks: Chunk[] = [];
  let li = 0;
  let ri = 0;
  let cursor = 0;
  const maxIterations = leftEdits.length + rightEdits.length + 4;

  for (let iter = 0; iter < maxIterations && (li < leftEdits.length || ri < rightEdits.length); iter++) {
    const nextL = leftEdits[li];
    const nextR = rightEdits[ri];
    const nextStart = Math.min(nextL ? nextL.start : b.lines.length + 1, nextR ? nextR.start : b.lines.length + 1);

    if (nextStart > cursor) {
      chunks.push({ kind: 'clean', lines: b.lines.slice(cursor, nextStart) });
      cursor = nextStart;
    }

    const activeL: EditSpan[] = [];
    const activeR: EditSpan[] = [];
    let regionStart = nextStart;
    let regionEnd = nextStart;

    if (nextL && (!nextR || nextL.start <= nextR.start)) {
      activeL.push(nextL);
      li++;
      regionEnd = Math.max(regionEnd, nextL.end);
    } else if (nextR) {
      activeR.push(nextR);
      ri++;
      regionEnd = Math.max(regionEnd, nextR.end);
    }

    // Expand conflict region until no remaining edit on either side overlaps [regionStart, regionEnd].
    const expandBound = leftEdits.length + rightEdits.length + 1;
    for (let step = 0; step < expandBound; step++) {
      let expanded = false;
      const curSpan: EditSpan = { start: regionStart, end: regionEnd, lines: [] };
      if (li < leftEdits.length && spansOverlap(curSpan, leftEdits[li]!)) {
        const e = leftEdits[li++]!;
        activeL.push(e);
        regionStart = Math.min(regionStart, e.start);
        regionEnd = Math.max(regionEnd, e.end);
        expanded = true;
      }
      const curSpan2: EditSpan = { start: regionStart, end: regionEnd, lines: [] };
      if (ri < rightEdits.length && spansOverlap(curSpan2, rightEdits[ri]!)) {
        const e = rightEdits[ri++]!;
        activeR.push(e);
        regionStart = Math.min(regionStart, e.start);
        regionEnd = Math.max(regionEnd, e.end);
        expanded = true;
      }
      if (!expanded) break;
    }

    const leftLines = applySpansInRange(b.lines, regionStart, regionEnd, activeL);
    const rightLines = applySpansInRange(b.lines, regionStart, regionEnd, activeR);
    const baseLines = b.lines.slice(regionStart, regionEnd);

    if (activeR.length === 0) {
      chunks.push({ kind: 'clean', lines: leftLines });
    } else if (activeL.length === 0) {
      chunks.push({ kind: 'clean', lines: rightLines });
    } else if (arraysEqual(leftLines, rightLines)) {
      chunks.push({ kind: 'clean', lines: leftLines });
    } else {
      chunks.push({
        kind: 'conflict',
        startLine: regionStart + 1,
        endLine: Math.max(regionStart + 1, regionEnd),
        base: baseLines,
        left: leftLines,
        right: rightLines,
      });
    }
    cursor = regionEnd;
  }

  if (cursor < b.lines.length) {
    chunks.push({ kind: 'clean', lines: b.lines.slice(cursor) });
  }

  return { chunks, trailingNewline };
}

/** Format an explicit Git-style conflict block so neither intent is silently dropped. */
export function formatConflictMarker(
  left: string,
  right: string,
  leftLabel = 'left',
  rightLabel = 'right',
): string {
  const parts = [`<<<<<<< ${leftLabel}`];
  if (left.length > 0) parts.push(left);
  parts.push('=======');
  if (right.length > 0) parts.push(right);
  parts.push(`>>>>>>> ${rightLabel}`);
  return parts.join('\n');
}

function tryParseJsonObject(text: string): Record<string, unknown> | null {
  try {
    const parsed: unknown = JSON.parse(text);
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
    return null;
  } catch {
    return null;
  }
}

/** Merge a ThreeWayFile: crisp 3-way first, semanticMerge for JSON conflicts, semanticRepair + mergeHunk for text/code. */
export async function mergeFile(file: ThreeWayFile, options: MergeOptions = {}): Promise<MergeOutcome> {
  const leftLabel = options.leftLabel ?? 'left';
  const rightLabel = options.rightLabel ?? 'right';
  const floor = options.compatibilityFloor ?? 0.7;

  // 1. Exact crisp 3-way merge for non-overlapping line edits (0 model calls).
  const { chunks, trailingNewline } = crispThreeWayChunks(file);
  const conflictIndices = chunks
    .map((chunk, idx) => (chunk.kind === 'conflict' ? idx : -1))
    .filter(idx => idx >= 0);

  if (conflictIndices.length === 0) {
    const mergedLines = chunks.flatMap(chunk => (chunk.kind === 'clean' ? chunk.lines : []));
    return {
      path: file.path,
      mode: 'crisp',
      status: 'clean',
      content: joinNormalized(mergedLines, trailingNewline),
      llmCallsUsed: false,
      resolutions: [],
      unresolvedCount: 0,
    };
  }

  // 2. Structured JSON / config merging via `@natlang/node/std` `semanticMerge`.
  if (file.path.endsWith('.json')) {
    const baseObj = tryParseJsonObject(file.base);
    const leftObj = tryParseJsonObject(file.left);
    const rightObj = tryParseJsonObject(file.right);
    if (baseObj && leftObj && rightObj) {
      const merged = await semanticMerge(baseObj, leftObj, rightObj, { floor });
      const resolutions: ConflictResolution[] = [
        ...merged.merged.map(entry => ({
          hunk: {
            path: file.path,
            startLine: 1,
            endLine: 1,
            base: JSON.stringify(entry.base),
            left: JSON.stringify(entry.ours),
            right: JSON.stringify(entry.theirs),
          },
          status: 'merged' as const,
          mergedText: JSON.stringify(entry.value),
          compatible: true,
          confidence: entry.compatibility,
          repairAttempts: 0,
          problems: [],
        })),
        ...merged.unresolved.map(entry => ({
          hunk: {
            path: file.path,
            startLine: 1,
            endLine: 1,
            base: JSON.stringify(entry.base),
            left: JSON.stringify(entry.ours),
            right: JSON.stringify(entry.theirs),
          },
          status: 'unresolved' as const,
          mergedText: formatConflictMarker(
            JSON.stringify(entry.ours, null, 2),
            JSON.stringify(entry.theirs, null, 2),
            leftLabel,
            rightLabel,
          ),
          compatible: entry.reason !== 'incompatible',
          confidence: entry.compatibility,
          repairAttempts: 0,
          problems: entry.problems ?? [`Unresolved JSON conflict at ${entry.path.join('.')}: ${entry.reason}`],
        })),
      ];

      if (merged.unresolved.length === 0) {
        return {
          path: file.path,
          mode: 'json-semantic',
          status: 'clean',
          content: `${JSON.stringify(merged.value, null, 2)}\n`,
          llmCallsUsed: merged.merged.length > 0,
          resolutions,
          unresolvedCount: 0,
        };
      }

      // Emit explicit conflict markers so unresolved JSON keys are never silently dropped.
      const markerContent = formatConflictMarker(file.left.trimEnd(), file.right.trimEnd(), leftLabel, rightLabel) + '\n';
      return {
        path: file.path,
        mode: 'json-semantic',
        status: 'conflict',
        content: markerContent,
        llmCallsUsed: true,
        resolutions,
        unresolvedCount: merged.unresolved.length,
      };
    }
  }

  // 3. Overlapping text/code conflicts: judge compatibility with intentsCompatible.nl, then repair with mergeHunk.nl.
  const assembledChunks: string[][] = chunks.map(chunk =>
    chunk.kind === 'clean' ? chunk.lines : chunk.base,
  );
  const resolutions: ConflictResolution[] = [];

  for (const idx of conflictIndices) {
    const chunk = chunks[idx]!;
    if (chunk.kind !== 'conflict') continue;

    const hunk: ConflictHunk = {
      path: file.path,
      startLine: chunk.startLine,
      endLine: chunk.endLine,
      base: chunk.base.join('\n'),
      left: chunk.left.join('\n'),
      right: chunk.right.join('\n'),
    };

    const decision = await decide(intentsCompatible, hunk);
    const probCompatible =
      decision.probabilities.find(entry => entry.value === true)?.probability ??
      (decision.value ? decision.confidence : 0);

    if (!decision.value || probCompatible < floor) {
      const marker = formatConflictMarker(hunk.left, hunk.right, leftLabel, rightLabel);
      assembledChunks[idx] = marker.split('\n');
      resolutions.push({
        hunk,
        status: 'unresolved',
        mergedText: marker,
        compatible: false,
        confidence: probCompatible,
        repairAttempts: 0,
        problems: ['Overlapping edits have incompatible intents; emitted explicit conflict markers.'],
      });
      continue;
    }

    const renderWithCandidate = (candidate: string): string => {
      const copy = assembledChunks.map((lines, i) => (i === idx ? candidate.split('\n') : lines));
      return joinNormalized(copy.flat(), trailingNewline);
    };

    const checkCandidate = async (candidate: string): Promise<string[]> => {
      const problems: string[] = [];
      if (candidate.trim().length === 0) problems.push('Merged hunk must not be empty.');
      if (candidate.includes('<<<<<<<') || candidate.includes('=======') || candidate.includes('>>>>>>>')) {
        problems.push('Merged hunk must not contain raw conflict markers.');
      }
      if (options.verify) {
        const verifierProblems = await options.verify(renderWithCandidate(candidate), file.path);
        problems.push(...verifierProblems);
      }
      return problems;
    };

    const initial = await mergeHunk(hunk, []);
    const repaired = await semanticRepair(initial, {
      check: checkCandidate,
      fix: (_prev, problems) => mergeHunk(hunk, problems),
      attempts: options.repairAttempts ?? 2,
    });

    if (repaired.ok) {
      assembledChunks[idx] = repaired.value.split('\n');
      resolutions.push({
        hunk,
        status: 'merged',
        mergedText: repaired.value,
        compatible: true,
        confidence: probCompatible,
        repairAttempts: repaired.attempts,
        problems: [],
      });
    } else {
      const marker = formatConflictMarker(hunk.left, hunk.right, leftLabel, rightLabel);
      assembledChunks[idx] = marker.split('\n');
      resolutions.push({
        hunk,
        status: 'unresolved',
        mergedText: marker,
        compatible: true,
        confidence: probCompatible,
        repairAttempts: repaired.attempts,
        problems: repaired.problems,
      });
    }
  }

  const unresolvedCount = resolutions.filter(r => r.status === 'unresolved').length;
  return {
    path: file.path,
    mode: 'text-semantic',
    status: unresolvedCount === 0 ? 'clean' : 'conflict',
    content: joinNormalized(assembledChunks.flat(), trailingNewline),
    llmCallsUsed: true,
    resolutions,
    unresolvedCount,
  };
}
