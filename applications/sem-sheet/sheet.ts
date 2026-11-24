/**
 * Sem-Sheet (Tier 1 Flagship A3, plans/semantic-fp/06_FLAGSHIP_APPLICATIONS.md §3 & §3.5):
 * Browser-compatible (zero Node `fs` or `node:*` imports) analytical grid combining:
 * - Deterministic topological DAG evaluation via `iterateOn(...).withMeasure(remainingCells)` where each step evaluates
 *   the entire ready layer concurrently.
 * - Pure TypeScript formula arithmetic and aggregations (`SUM`, `AVG`, `MIN`, `MAX`, `COUNT`, `+`, `-`, `*`, `/`).
 * - Concurrent semantic columns (`=SEM(...)`, `classifyCell.nl`, `semanticFilter`, tree-mode `semanticReduce`).
 * - `pluggable` column promotion (`crisp | nl | shadow`) emitting `pluggable_shadow` agreement events for stabilized
 *   classification columns.
 * - Collaborative three-way workbook merging via `semanticMerge`.
 */
import { iterateOn, pluggable, untrusted } from '@natlang/node';
import {
  semanticFilter,
  semanticMap,
  semanticMerge,
  semanticReduce,
  type Merge,
  type MergeOptions,
} from '@natlang/node/std';
import classifyCell from './classifyCell.nl';
import combineSummary from './combineSummary.nl';
import evalSemanticCell from './evalSemanticCell.nl';
import type {
  CellCoord,
  CellLabel,
  CellValue,
  FormulaCell,
  SemanticColumnSpec,
  SheetDiff,
  SheetResult,
  SheetState,
} from './types.js';

export type EvaluateSheetOptions = {
  /** Column promotion specs for `classify` cells (`crisp | nl | shadow`). */
  columns?: Record<string, SemanticColumnSpec>;
  /** Maximum concurrent semantic cell evaluations per DAG layer (default `8`). */
  concurrency?: number;
};

const CELL_REF_RE = /\b([A-Z]+)(\d+)\b/g;
const RANGE_RE = /\b([A-Z]+)(\d+):([A-Z]+)(\d+)\b/g;

/** Convert spreadsheet column letters (`A` -> 1, `Z` -> 26, `AA` -> 27) to a 1-based column index. */
export function colLettersToIndex(letters: string): number {
  let acc = 0;
  for (let i = 0; i < letters.length; i++) {
    acc = acc * 26 + (letters.charCodeAt(i) - 64);
  }
  return acc;
}

/** Convert a 1-based column index back to spreadsheet letters (`1` -> `A`, `27` -> `AA`). */
export function indexToColLetters(index: number): string {
  let n = index;
  let out = '';
  for (let guard = 0; guard < 16 && n > 0; guard++) {
    const rem = (n - 1) % 26;
    out = String.fromCharCode(65 + rem) + out;
    n = Math.floor((n - 1) / 26);
  }
  return out;
}

/** Expand an A1:B3 rectangular range into every covered CellCoord in row-major order. */
export function expandRange(startCol: string, startRow: number, endCol: string, endRow: number): CellCoord[] {
  const c0 = Math.min(colLettersToIndex(startCol), colLettersToIndex(endCol));
  const c1 = Math.max(colLettersToIndex(startCol), colLettersToIndex(endCol));
  const r0 = Math.min(startRow, endRow);
  const r1 = Math.max(startRow, endRow);
  const coords: CellCoord[] = [];
  for (let r = r0; r <= r1; r++) {
    for (let c = c0; c <= c1; c++) {
      coords.push(`${indexToColLetters(c)}${r}`);
    }
  }
  return coords;
}

/** Extract exact upstream cell dependencies from any FormulaCell. */
export function cellDependencies(cell: FormulaCell): CellCoord[] {
  if (cell.kind === 'literal') return [];
  if (cell.kind !== 'formula') return [...new Set(cell.deps)];
  const deps = new Set<CellCoord>();
  const body = cell.expr.replace(/^=/, '');
  let withoutRanges = body;
  for (const match of body.matchAll(RANGE_RE)) {
    for (const coord of expandRange(match[1]!, Number(match[2]!), match[3]!, Number(match[4]!))) {
      deps.add(coord);
    }
    withoutRanges = withoutRanges.replace(match[0], ' ');
  }
  for (const match of withoutRanges.matchAll(CELL_REF_RE)) {
    deps.add(`${match[1]!}${match[2]!}`);
  }
  return [...deps];
}

/** Coerce a CellValue to a finite number for spreadsheet arithmetic. */
function toNumber(value: CellValue | undefined): number {
  if (typeof value === 'number') return Number.isFinite(value) ? value : 0;
  if (typeof value === 'boolean') return value ? 1 : 0;
  if (typeof value === 'string') {
    const n = Number(value.trim());
    return Number.isFinite(n) ? n : 0;
  }
  return 0;
}

type Token =
  | { kind: 'num'; value: number }
  | { kind: 'op'; value: '+' | '-' | '*' | '/' }
  | { kind: 'lparen' }
  | { kind: 'rparen' };

/** Tokenize a pure arithmetic expression after cell references and aggregations have been substituted. */
function tokenizeArithmetic(expr: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;
  for (let guard = 0; guard < expr.length + 1 && i < expr.length; guard++) {
    const ch = expr[i]!;
    if (/\s/.test(ch)) { i++; continue; }
    if (ch === '(') { tokens.push({ kind: 'lparen' }); i++; continue; }
    if (ch === ')') { tokens.push({ kind: 'rparen' }); i++; continue; }
    if (ch === '+' || ch === '-' || ch === '*' || ch === '/') {
      tokens.push({ kind: 'op', value: ch });
      i++;
      continue;
    }
    const numMatch = /^\d+(?:\.\d+)?/.exec(expr.slice(i));
    if (numMatch) {
      tokens.push({ kind: 'num', value: Number(numMatch[0]) });
      i += numMatch[0].length;
      continue;
    }
    throw new Error(`Unexpected token in formula expression near "${expr.slice(i, i + 12)}".`);
  }
  return tokens;
}

/** Recursive-descent evaluator for pure numeric arithmetic expressions (zero eval). */
function evaluateArithmetic(expr: string): number {
  const tokens = tokenizeArithmetic(expr);
  let pos = 0;

  const parsePrimary = (): number => {
    const tok = tokens[pos];
    if (!tok) throw new Error('Unexpected end of formula expression.');
    if (tok.kind === 'op' && tok.value === '-') {
      pos++;
      return -parsePrimary();
    }
    if (tok.kind === 'op' && tok.value === '+') {
      pos++;
      return parsePrimary();
    }
    if (tok.kind === 'num') {
      pos++;
      return tok.value;
    }
    if (tok.kind === 'lparen') {
      pos++;
      const val = parseAddSub();
      if (tokens[pos]?.kind !== 'rparen') throw new Error('Unclosed parenthesis in formula.');
      pos++;
      return val;
    }
    throw new Error('Invalid formula syntax.');
  };

  const parseMulDiv = (): number => {
    let left = parsePrimary();
    for (let guard = 0; guard < tokens.length; guard++) {
      const tok = tokens[pos];
      if (!tok || tok.kind !== 'op' || (tok.value !== '*' && tok.value !== '/')) break;
      pos++;
      const right = parsePrimary();
      left = tok.value === '*' ? left * right : (right === 0 ? 0 : left / right);
    }
    return left;
  };

  const parseAddSub = (): number => {
    let left = parseMulDiv();
    for (let guard = 0; guard < tokens.length; guard++) {
      const tok = tokens[pos];
      if (!tok || tok.kind !== 'op' || (tok.value !== '+' && tok.value !== '-')) break;
      pos++;
      const right = parseMulDiv();
      left = tok.value === '+' ? left + right : left - right;
    }
    return left;
  };

  const value = parseAddSub();
  if (pos !== tokens.length) throw new Error('Trailing tokens in formula expression.');
  return Math.round(value * 1e6) / 1e6;
}

/** Evaluate a crisp spreadsheet formula (`SUM`, `AVG`, `MIN`, `MAX`, `COUNT`, and arithmetic) from resolved values. */
export function evaluateCrispFormula(expr: string, values: Readonly<Record<CellCoord, CellValue>>): number {
  let expanded = expr.trim().replace(/^=/, '');
  // Resolve aggregate functions over ranges or comma-separated refs first.
  const aggRe = /\b(SUM|AVG|MIN|MAX|COUNT)\s*\(([^()]+)\)/gi;
  for (let pass = 0; pass < 16 && aggRe.test(expanded); pass++) {
    aggRe.lastIndex = 0;
    expanded = expanded.replace(aggRe, (_whole, fnName: string, argsText: string) => {
      const nums: number[] = [];
      for (const part of argsText.split(',').map(s => s.trim()).filter(Boolean)) {
        const rangeMatch = /^([A-Z]+)(\d+):([A-Z]+)(\d+)$/.exec(part);
        if (rangeMatch) {
          for (const coord of expandRange(rangeMatch[1]!, Number(rangeMatch[2]!), rangeMatch[3]!, Number(rangeMatch[4]!))) {
            if (values[coord] !== null && values[coord] !== undefined && values[coord] !== '') {
              nums.push(toNumber(values[coord]));
            }
          }
        } else if (/^[A-Z]+\d+$/.test(part)) {
          if (values[part] !== null && values[part] !== undefined && values[part] !== '') {
            nums.push(toNumber(values[part]));
          }
        } else {
          const lit = Number(part);
          if (Number.isFinite(lit)) nums.push(lit);
        }
      }
      const fn = fnName.toUpperCase();
      if (fn === 'COUNT') return String(nums.length);
      if (!nums.length) return '0';
      if (fn === 'SUM') return String(nums.reduce((a, b) => a + b, 0));
      if (fn === 'AVG') return String(nums.reduce((a, b) => a + b, 0) / nums.length);
      if (fn === 'MIN') return String(Math.min(...nums));
      if (fn === 'MAX') return String(Math.max(...nums));
      return '0';
    });
  }
  // Substitute remaining individual cell references with their numeric values.
  const numericExpr = expanded.replace(CELL_REF_RE, (_m, col: string, row: string) => {
    const coord = `${col}${row}`;
    return String(toNumber(values[coord]));
  });
  return evaluateArithmetic(numericExpr);
}

/** Build a fenced text representation of a cell's upstream dependencies for semantic leaves. */
function formatInputs(deps: readonly CellCoord[], values: Readonly<Record<CellCoord, CellValue>>): string {
  return deps.map(coord => `${coord}: ${String(values[coord] ?? '')}`).join('\n');
}

/** Well-founded DAG termination measure: the count of cells not yet evaluated. */
export const remainingCells = (state: SheetState): number => state.pending.length;

/**
 * Evaluate a Sem-Sheet workbook in deterministic topological DAG layers via `iterateOn(...).withMeasure(remainingCells)`.
 * Crisp literals/formulas evaluate deterministically in TypeScript; semantic cells (`=SEM`, `classify`, `sem-filter`,
 * `sem-reduce`) execute concurrently per ready layer. Promoted columns use `pluggable` (`crisp | nl | shadow`).
 */
export async function evaluateSheet(
  cells: Readonly<Record<CellCoord, FormulaCell>>,
  options: EvaluateSheetOptions = {},
): Promise<SheetResult> {
  const concurrency = options.concurrency ?? 8;
  const columnSpecs = options.columns ?? {};
  const allCoords = Object.keys(cells).sort();

  const depMap = new Map<CellCoord, CellCoord[]>();
  for (const coord of allCoords) {
    const cell = cells[coord]!;
    depMap.set(coord, cellDependencies(cell).filter(dep => Object.hasOwn(cells, dep)));
  }

  // Build pluggable classifiers once per configured semantic column (`crisp | nl | shadow`).
  type ClassifierFn = (
    prompt: string,
    rowValues: Record<CellCoord, CellValue>,
    inputText: string,
    coord: CellCoord,
    columnName: string,
  ) => Promise<CellLabel>;

  const columnClassifiers = new Map<string, { fn: ClassifierFn; isCrispOnly: boolean }>();
  for (const [colId, spec] of Object.entries(columnSpecs)) {
    const nlSide: ClassifierFn = (prompt, _rowValues, inputText, coord, columnName) =>
      classifyCell(prompt, untrusted(inputText, `sheet cell ${coord}`), columnName);
    if (spec.crisp) {
      const crispSide: ClassifierFn = async (_prompt, rowValues, inputText) => spec.crisp!(rowValues, inputText);
      const wrapped = pluggable<[string, Record<CellCoord, CellValue>, string, CellCoord, string], CellLabel>(
        { crisp: crispSide, nl: nlSide },
        spec.mode,
        { name: `sem-sheet.column.${colId}`, default: 'nl', serve: spec.serve ?? 'crisp' },
      );
      columnClassifiers.set(colId, { fn: wrapped, isCrispOnly: spec.mode === 'crisp' });
    } else {
      columnClassifiers.set(colId, { fn: nlSide, isCrispOnly: false });
    }
  }

  const evaluateLayer = async (state: SheetState): Promise<SheetState> => {
    const ready = state.pending.filter(coord =>
      (depMap.get(coord) ?? []).every(dep => Object.hasOwn(state.values, dep)),
    );
    if (ready.length === 0) {
      throw new Error(`Circular dependency in sheet cells: ${state.pending.join(', ')}.`);
    }

    const nextValues: Record<CellCoord, CellValue> = { ...state.values };
    let crispCount = 0;
    let semanticCount = 0;

    const crispReady: CellCoord[] = [];
    const semanticReady: CellCoord[] = [];
    for (const coord of ready) {
      const cell = cells[coord]!;
      if (cell.kind === 'literal' || cell.kind === 'formula') {
        crispReady.push(coord);
      } else {
        semanticReady.push(coord);
      }
    }

    // Evaluate crisp literals and formulas synchronously with zero LLM calls.
    for (const coord of crispReady) {
      const cell = cells[coord]!;
      if (cell.kind === 'literal') {
        nextValues[coord] = cell.value;
      } else if (cell.kind === 'formula') {
        nextValues[coord] = evaluateCrispFormula(cell.expr, state.values);
      }
      crispCount++;
    }

    // Evaluate all semantic cells in this ready layer concurrently.
    const semanticResults = await semanticMap(
      semanticReady,
      async coord => {
        const cell = cells[coord]!;
        if (cell.kind === 'sem') {
          const inputText = formatInputs(cell.deps, state.values);
          const value = await evalSemanticCell(
            cell.prompt,
            untrusted(inputText, `sheet cell ${coord}`),
            cell.column ?? coord,
          );
          return { coord, value, crisp: false };
        }
        if (cell.kind === 'classify') {
          const rowValues: Record<CellCoord, CellValue> = {};
          for (const dep of cell.deps) rowValues[dep] = state.values[dep] ?? null;
          const inputText = formatInputs(cell.deps, state.values);
          const colEntry = columnClassifiers.get(cell.column);
          if (colEntry) {
            const label = await colEntry.fn(cell.prompt, rowValues, inputText, coord, cell.column);
            return { coord, value: label, crisp: colEntry.isCrispOnly };
          }
          const label = await classifyCell(cell.prompt, untrusted(inputText, `sheet cell ${coord}`), cell.column);
          return { coord, value: label, crisp: false };
        }
        if (cell.kind === 'sem-filter') {
          const items = cell.deps.map(dep => ({
            coord: dep,
            value: untrusted(String(state.values[dep] ?? ''), `sheet cell ${dep}`),
          }));
          const kept = await semanticFilter(items, cell.criterion, { concurrency });
          return { coord, value: kept.map(item => String(item.value)).join(', '), crisp: false };
        }
        if (cell.kind === 'sem-reduce') {
          const items = cell.deps
            .map(dep => ({ dep, text: String(state.values[dep] ?? '') }))
            .filter(entry => entry.text.length > 0)
            .map(entry => untrusted(entry.text, `sheet cell ${entry.dep}`));
          const reduced = await semanticReduce(
            items,
            '',
            (_acc, item) => String(item),
            {
              mode: 'tree',
              concurrency,
              combine: (left, right) =>
                combineSummary(
                  cell.goal,
                  untrusted(String(left), `sheet cell ${coord}:left`),
                  untrusted(String(right), `sheet cell ${coord}:right`),
                ),
            },
          );
          return { coord, value: reduced, crisp: false };
        }
        return { coord, value: null as CellValue, crisp: true };
      },
      { concurrency },
    );

    for (const res of semanticResults) {
      nextValues[res.coord] = res.value;
      if (res.crisp) crispCount++;
      else semanticCount++;
    }

    const readySet = new Set(ready);
    return {
      values: nextValues,
      pending: state.pending.filter(coord => !readySet.has(coord)),
      layers: [...state.layers, ready],
      crispEvaluated: state.crispEvaluated + crispCount,
      semanticEvaluated: state.semanticEvaluated + semanticCount,
    };
  };

  const initial: SheetState = {
    values: {},
    pending: allCoords,
    layers: [],
    crispEvaluated: 0,
    semanticEvaluated: 0,
  };

  const finalState = await iterateOn(evaluateLayer, initial)
    .withMeasure(remainingCells)
    .until(state => state.pending.length === 0);

  const total = finalState.crispEvaluated + finalState.semanticEvaluated;
  return {
    values: finalState.values,
    layers: finalState.layers,
    crispEvaluated: finalState.crispEvaluated,
    semanticEvaluated: finalState.semanticEvaluated,
    crispShare: total > 0 ? finalState.crispEvaluated / total : 1,
  };
}

/** Exact syntax check for a merged cell definition string inside `semanticMerge`. */
export function checkMergedCellDefinition(value: unknown, path: string[]): string[] {
  if (typeof value !== 'string') return [`Cell ${path.join('.')} must be a string definition.`];
  const trimmed = value.trim();
  if (!trimmed) return [`Cell ${path.join('.')} cannot be empty.`];
  if (trimmed.startsWith('=')) {
    let depth = 0;
    for (let i = 0; i < trimmed.length; i++) {
      if (trimmed[i] === '(') depth++;
      else if (trimmed[i] === ')') depth--;
      if (depth < 0) return [`Cell ${path.join('.')} has unbalanced closing parenthesis.`];
    }
    if (depth !== 0) return [`Cell ${path.join('.')} has unclosed parenthesis.`];
  }
  return [];
}

/**
 * Merge two collaborative edits of a workbook (`SheetDiff`) via `semanticMerge`:
 * - Non-overlapping cell edits merge crisply in TypeScript with zero LLM calls.
 * - Same-cell conflicts resolve by intent compatibility and exact formula syntax validation.
 */
export async function mergeSheetDiffs(
  base: SheetDiff,
  ours: SheetDiff,
  theirs: SheetDiff,
  options: MergeOptions = {},
): Promise<Merge<SheetDiff>> {
  return semanticMerge(base, ours, theirs, {
    floor: options.floor ?? 0.6,
    check: options.check ?? checkMergedCellDefinition,
    ...options,
  });
}
