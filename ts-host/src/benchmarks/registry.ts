/**
 * Registry for benchmark-specific policy. Core code (teacher/, evaluation/) asks the registry by name and never
 * imports a dataset module; each `benchmarks/<name>/` package registers its pieces, and `benchmarks/builtin.ts`
 * lists the packages that ship with the host. This module imports nothing from the rest of src.
 */

/** A reviewed variant of a source row that is exempt from an otherwise pending source-review hold. */
export type SourceContract = {
  /** Dataset name as it appears in `ProgramRecord.source` / the source review table. */
  dataset: string;
  /** The held source id this contract's reviewed variant is allowed to use. */
  sourceId: string;
  /** Exact, immutable-record check for the reviewed variant. */
  isReviewedVariant(record: unknown): boolean;
};

/** Minimal row shape the runtime-failure rules need (mirrors teacher/curriculum-policy `runtimeFailureReason`). */
export type FailureRow = { task: Record<string, unknown>; provenance?: Record<string, unknown>;
  outcome?: Record<string, unknown>; trajectory?: unknown[] };
/** Returns a reason when a rejected outcome of this dataset must not be used as negative evidence. */
export type RuntimeFailureRule = (row: FailureRow, record: Record<string, any>) => string | undefined;

const contracts: SourceContract[] = [];
const failureRules = new Map<string, RuntimeFailureRule[]>();

export function registerSourceContract(contract: SourceContract): void {
  if (contracts.some(entry => entry.dataset === contract.dataset && entry.sourceId === contract.sourceId)) return;
  contracts.push(contract);
}
export function sourceContractsFor(dataset: string): readonly SourceContract[] {
  return contracts.filter(contract => contract.dataset === dataset);
}

export function registerRuntimeFailureRule(dataset: string, rule: RuntimeFailureRule): void {
  const rules = failureRules.get(dataset) ?? [];
  if (!rules.includes(rule)) failureRules.set(dataset, [...rules, rule]);
}
export function runtimeFailureRulesFor(dataset: string): readonly RuntimeFailureRule[] {
  return failureRules.get(dataset) ?? [];
}
