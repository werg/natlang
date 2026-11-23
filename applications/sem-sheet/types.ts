export type PluggableSetting = 'crisp' | 'nl' | 'shadow';

/** Spreadsheet A1-style coordinate such as "A1", "B12", or "D4". */
export type CellCoord = string;

/** Resolved scalar value stored in a spreadsheet cell. */
export type CellValue = string | number | boolean | null;

/** Finite label domain for decision-readout classification columns (`classifyCell.nl`). */
export type CellLabel = 'high' | 'medium' | 'low' | 'none';

/** Definition of one cell in a Sem-Sheet workbook. */
export type FormulaCell =
  | { kind: 'literal'; value: CellValue }
  | { kind: 'formula'; expr: string }
  | { kind: 'sem'; prompt: string; deps: CellCoord[]; column?: string }
  | { kind: 'classify'; prompt: string; deps: CellCoord[]; column: string }
  | { kind: 'sem-filter'; criterion: string; deps: CellCoord[] }
  | { kind: 'sem-reduce'; goal: string; deps: CellCoord[] };

/**
 * Promotion policy for a stabilized semantic classification column (`pluggable` mode: `'crisp' | 'nl' | 'shadow'`).
 * In `'shadow'` mode, both the crisp rule and `classifyCell.nl` run and emit `pluggable_shadow` agreement events.
 */
export type SemanticColumnSpec = {
  id: string;
  label: string;
  prompt: string;
  mode?: PluggableSetting;
  serve?: 'crisp' | 'nl';
  /** Crisp rule for promoted or shadow columns (zero LLM calls when served in `'crisp'` mode). */
  crisp?: (rowValues: Record<CellCoord, CellValue>, inputText: string) => CellLabel;
};

/** Mutable DAG evaluation state carried across `iterateOn(...).withMeasure(remainingCells)` layers. */
export type SheetState = {
  values: Record<CellCoord, CellValue>;
  pending: CellCoord[];
  layers: CellCoord[][];
  crispEvaluated: number;
  semanticEvaluated: number;
};

/** Collaborative workbook snapshot mapping each coordinate to its formula or literal definition text. */
export type SheetDiff = Record<CellCoord, string>;

/** Final evaluated workbook output with DAG layer schedule and crisp/semantic accounting. */
export type SheetResult = {
  values: Record<CellCoord, CellValue>;
  layers: CellCoord[][];
  crispEvaluated: number;
  semanticEvaluated: number;
  crispShare: number;
};
