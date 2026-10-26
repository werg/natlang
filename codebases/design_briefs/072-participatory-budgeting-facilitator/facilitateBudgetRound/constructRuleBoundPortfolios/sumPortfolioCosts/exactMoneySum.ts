import type { ExactPortfolioSum, Money, PortfolioCostLine } from "../../../types.js";

function integerMinorUnits(value: string): bigint {
  if (!/^(?:0|[1-9][0-9]*)$/.test(value)) throw new Error("invalid_nonnegative_minor_units");
  return BigInt(value);
}

function validScale(scale: number): boolean {
  return Number.isSafeInteger(scale) && scale >= 0;
}

export default function exactMoneySum(lines: PortfolioCostLine[], budget: Money): ExactPortfolioSum {
  const unresolved: string[] = [];
  const duplicatedLineIds = new Set<string>();
  const byId = new Map<string, PortfolioCostLine>();
  const conflictingIds = new Set<string>();

  if (!validScale(budget.scale)) unresolved.push("invalid_budget_scale");
  let budgetMinor = 0n;
  try { budgetMinor = integerMinorUnits(budget.minorUnits); }
  catch { unresolved.push("invalid_budget_minor_units"); }

  for (const line of lines) {
    const previous = byId.get(line.lineId);
    if (!previous) byId.set(line.lineId, line);
    else {
      duplicatedLineIds.add(line.lineId);
      if (previous.cost.minorUnits !== line.cost.minorUnits || previous.cost.currency !== line.cost.currency ||
          previous.cost.scale !== line.cost.scale || previous.status !== line.status) conflictingIds.add(line.lineId);
    }
  }
  for (const id of conflictingIds) unresolved.push(`conflicting_duplicate_line:${id}`);

  let total = 0n;
  for (const [id, line] of byId) {
    if (conflictingIds.has(id)) continue;
    if (line.status !== "supported") { unresolved.push(`unresolved_cost:${id}`); continue; }
    if (!validScale(line.cost.scale)) { unresolved.push(`invalid_scale:${id}`); continue; }
    if (line.cost.currency !== budget.currency || line.cost.scale !== budget.scale) {
      unresolved.push(`currency_or_scale_mismatch:${id}`);
      continue;
    }
    if (line.sourceRefs.length === 0) { unresolved.push(`cost_without_source:${id}`); continue; }
    try { total += integerMinorUnits(line.cost.minorUnits); }
    catch { unresolved.push(`invalid_minor_units:${id}`); }
  }

  const withinBudget = unresolved.length === 0 && total <= budgetMinor;
  if (unresolved.length === 0 && total > budgetMinor) unresolved.push("portfolio_exceeds_budget");
  return {
    total: { minorUnits: total.toString(), currency: budget.currency, scale: budget.scale },
    withinBudget,
    unresolved,
    duplicatedLineIds: [...duplicatedLineIds].sort(),
  };
}
