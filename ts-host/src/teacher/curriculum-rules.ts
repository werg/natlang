/**
 * The per-family rule ladders of the curriculum policy as a data registry (plans/NATLANG_NATIVE_REVIEW.md, P6).
 * `curriculum-policy.ts` used to spell each ladder as a chain of `if`s; here each rung is an entry that says when it
 * applies (`when`), what breaks the contract (`violates`) and the reason it returns. The evaluator is the only code:
 * a new legacy family is a new entry, listed in order, with a test, not a new branch.
 *
 * Order is part of the registry: the first entry that applies returns its reason. A `step` entry marks a place where
 * code that is not a data rule runs (source review, benchmark rules); the policy supplies those.
 */

/** A condition over the context `{ record, curriculum, row }`; paths are dotted and optional-chained. */
export type Condition =
  | { path: string; op: 'truthy' | 'falsy' | 'undefined' }
  | { path: string; op: 'eq' | 'ne'; value: unknown }
  | { path: string; op: 'in'; value: unknown[] }
  /** `(value ?? default) < limit`. */
  | { path: string; op: 'lt'; value: number; default: number }
  /** `!(Number(value ?? default) >= limit)`: also true for a value that is not a number. */
  | { path: string; op: 'below-as-number'; value: number; default: number }
  /** The value has an `includes` that finds `value` (an array of strings, or a string). */
  | { path: string; op: 'includes'; value: unknown }
  /** The JSON text of the value contains `value`. */
  | { path: string; op: 'json-includes'; value: string }
  | { any: Condition[] }
  | { all: Condition[] }
  /** A named crisp predicate over the context, for a rule that data cannot state. */
  | { predicate: string };

export type RuleEntry = {
  /** Stable name of the rung. */
  id: string;
  reason: string;
  /** All must hold for the rung to apply. */
  when: Condition[];
  /** The rung returns its reason when this holds as well. */
  violates: Condition;
};
export type RuleStep = { step: string };
export type RuleRegistry = (RuleEntry | RuleStep)[];
export type RuleContext = { record: unknown; curriculum: unknown; row?: unknown };
export type RuleHooks = { predicates?: Record<string, (context: RuleContext) => boolean>; steps?: Record<string, (context: RuleContext) => string | undefined> };

function read(context: RuleContext, path: string): unknown {
  let value: any = context;
  for (const key of path.split('.')) value = value?.[key];
  return value;
}

export function holds(condition: Condition, context: RuleContext, hooks: RuleHooks = {}): boolean {
  if ('any' in condition) return condition.any.some(item => holds(item, context, hooks));
  if ('all' in condition) return condition.all.every(item => holds(item, context, hooks));
  if ('predicate' in condition) {
    const predicate = hooks.predicates?.[condition.predicate];
    if (!predicate) throw new RangeError(`no predicate named ${JSON.stringify(condition.predicate)} is registered`);
    return predicate(context);
  }
  const value = read(context, condition.path);
  switch (condition.op) {
    case 'truthy': return !!value;
    case 'falsy': return !value;
    case 'undefined': return value === undefined;
    case 'eq': return value === condition.value;
    case 'ne': return value !== condition.value;
    case 'in': return condition.value.includes(value);
    case 'lt': return ((value ?? condition.default) as number) < condition.value;
    case 'below-as-number': return !(Number(value ?? condition.default) >= condition.value);
    case 'includes': return typeof (value as { includes?: unknown } | undefined)?.includes === 'function' &&
      (value as { includes(item: unknown): boolean }).includes(condition.value);
    case 'json-includes': return (JSON.stringify(value) ?? '').includes(condition.value);
  }
}

/** The reason of the first rung that applies and is violated, or undefined. `step` entries run the policy's hook in place. */
export function firstReason(registry: RuleRegistry, context: RuleContext, hooks: RuleHooks = {}): string | undefined {
  for (const entry of registry) {
    if ('step' in entry) {
      const step = hooks.steps?.[entry.step];
      if (!step) throw new RangeError(`no step named ${JSON.stringify(entry.step)} is registered`);
      const reason = step(context);
      if (reason) return reason;
      continue;
    }
    if (entry.when.every(item => holds(item, context, hooks)) && holds(entry.violates, context, hooks)) return entry.reason;
  }
}

const family = (name: string): Condition => ({ path: 'curriculum.family', op: 'eq', value: name });
const source = (name: string): Condition => ({ path: 'record.source', op: 'eq', value: name });
const files = (field: string) => `record.semantics.files_oracle.${field}`;

/** Families whose premise no longer exists in the runtime; their cases are not collected. */
export const RETIRED_FAMILY_NAMES: readonly string[] = ['inline_type_repair'];

/** Source counterfactuals and legacy contracts whose labels cannot be trusted; first match wins. */
export const QUARANTINE_RULES: RuleRegistry = [
  { id: 'treedst-transition', reason: 'unverified_tree_transition_contract', when: [source('treedst')], violates: { predicate: 'no-tree-value-contract' } },
  { step: 'source-review' },
  { id: 'cb-highlighter', reason: 'legacy_highlighter_oracle', when: [{ path: 'record.family', op: 'eq', value: 'cb_highlighter' }],
    violates: { path: 'record.generation.highlighter_quality_version', op: 'ne', value: 2 } },
  { id: 'commaqa-numeric', reason: 'legacy_numeric_reference_contract', when: [family('commaqa_numeric')],
    violates: { path: 'curriculum.family_version', op: 'lt', value: 3, default: 1 } },
  { id: 'commaqa-question', reason: 'unverified_movie_schema_contract', when: [family('commaqa_question')],
    violates: { path: 'curriculum.family_version', op: 'eq', value: 3 } },
  { id: 'entailment-premise-removed', reason: 'unverified_counterfactual', when: [family('entailment_premises')],
    violates: { path: 'curriculum.variant', op: 'eq', value: 'premise_removed' } },
  { id: 'folder-extract', reason: 'legacy_extraction_contract', when: [family('folder_extract')],
    violates: { any: [{ path: files('quote_sources'), op: 'falsy' }, { path: files('return_count'), op: 'falsy' }] } },
  { id: 'folder-edit', reason: 'legacy_rewrite_contract', when: [family('folder_edit')],
    violates: { any: [{ path: files('rubric'), op: 'falsy' }, { path: files('return_count'), op: 'falsy' }] } },
  { id: 'folder-index', reason: 'legacy_counts_contract', when: [family('folder_index')],
    violates: { any: [{ path: files('return_count'), op: 'falsy' }, { path: files('total'), op: 'undefined' }] } },
  { id: 'folder-triage', reason: 'legacy_move_contract', when: [family('folder_triage')],
    violates: { path: files('compare'), op: 'ne', value: 'moves' } },
  { id: 'folder-find', reason: 'legacy_article_evidence_policy', when: [family('folder_find')],
    violates: { path: 'curriculum.answer_evidence.length', op: 'falsy' } },
  { id: 'folder-mixed-payment', reason: 'legacy_payment_scope', when: [family('folder_mixed'), { path: 'record.dataset', op: 'eq', value: 'banking77' }],
    violates: { path: 'curriculum.payment_scope_version', op: 'ne', value: 2 } },
];

/** Exercises whose generation requests are held (unfair answer-span contracts). */
export const GENERATION_HOLD_RULES: RuleRegistry = [
  { id: 'qasper-extractive', reason: 'awaiting_extractive_equivalence_oracle', when: [source('qasper')], violates: { all: [] } },
];

/**
 * Failures of old rows that say nothing about the model's decisions, in the order `runtimeFailureReason` checks them.
 * Rows that were accepted are never listed. The benchmark rules (registered by source name) and the markdown
 * and named-tree rungs are code (steps and predicates): they need row-level reads that data does not state.
 */
export const RUNTIME_FAILURE_RULES: RuleRegistry = [
  { id: 'markdown-terminal-newline', reason: 'legacy_markdown_terminal_newline_oracle', when: [], violates: { predicate: 'legacy-markdown-terminal-newline' } },
  { id: 'treedst-named-tree', reason: 'obsolete_named_tree_oracle', when: [source('treedst')], violates: { predicate: 'treedst-oracle-not-named-tree' } },
  { step: 'benchmark-rules' },
  // Extractive annotations do not enumerate every semantically equivalent span boundary: wrong-answer traces are kept for review.
  { id: 'extractive-answer-equivalence', reason: 'unreviewed_extractive_answer_equivalence',
    when: [{ path: 'record.source', op: 'in', value: ['qasper', 'musique'] }],
    violates: { path: 'row.outcome.rejection_reasons', op: 'includes', value: 'answer' } },
  // Rows from before runtime contract 17 that depend on behaviour the runtime has since changed.
  { id: 'obsolete-runtime-contract', reason: 'obsolete_runtime_contract',
    when: [{ path: 'row.provenance.runtime_contract_version', op: 'below-as-number', value: 17, default: 0 }],
    violates: { any: [
      { predicate: 'effective-family-inline-late-binding' },
      { all: [{ path: 'record.family', op: 'eq', value: 'cb_reconciliation' }, { path: 'record.semantics.expected', op: 'json-includes', value: '__proto__' }] },
      { all: [{ predicate: 'effective-family-logic-proof-verifier' }, { path: 'row.trajectory', op: 'json-includes', value: 'bad character at' }] },
    ] } },
];
