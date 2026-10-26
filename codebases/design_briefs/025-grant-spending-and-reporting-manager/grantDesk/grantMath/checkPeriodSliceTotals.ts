import type { GrantPeriodSlice, PeriodConservationResult } from "../../types.js";

function parseMinorUnits(value: string): bigint | null {
  return /^(?:0|[1-9][0-9]*)$/.test(value) ? BigInt(value) : null;
}

/** Exact conservation check: no floating point conversion or implicit rounding. */
export default function verifyPeriodConservation(slices: GrantPeriodSlice[]): PeriodConservationResult {
  const blockers: string[] = [];
  const withheld = new Set<string>();
  const firstById = new Map<string, GrantPeriodSlice>();
  const invalidExpenses = new Set<string>();

  for (const slice of slices) {
    const prior = firstById.get(slice.sliceId);
    if (prior) {
      invalidExpenses.add(slice.expenseId);
      blockers.push(`duplicate_slice:${slice.sliceId}`);
    } else firstById.set(slice.sliceId, slice);
    if (!slice.sliceId || !slice.expenseId || !slice.periodId || !slice.basisRef ||
        !Number.isInteger(slice.amount.scale) || slice.amount.scale < 0 ||
        !Number.isInteger(slice.originalAmount.scale) || slice.originalAmount.scale < 0) {
      invalidExpenses.add(slice.expenseId);
      blockers.push(`invalid_slice_metadata:${slice.sliceId}`);
    }
    if (parseMinorUnits(slice.amount.minorUnits) === null || parseMinorUnits(slice.originalAmount.minorUnits) === null ||
        slice.amount.currency !== slice.originalAmount.currency || slice.amount.scale !== slice.originalAmount.scale) {
      invalidExpenses.add(slice.expenseId);
      blockers.push(`invalid_or_incompatible_money:${slice.expenseId}`);
    }
  }

  const byExpense = new Map<string, GrantPeriodSlice[]>();
  for (const slice of firstById.values()) {
    const group = byExpense.get(slice.expenseId) ?? [];
    group.push(slice);
    byExpense.set(slice.expenseId, group);
  }
  for (const [expenseId, group] of byExpense) {
    const first = group[0];
    let total = 0n;
    for (const slice of group) {
      const amount = parseMinorUnits(slice.amount.minorUnits);
      if (amount !== null) total += amount;
      if (slice.originalAmount.minorUnits !== first.originalAmount.minorUnits ||
          slice.originalAmount.currency !== first.originalAmount.currency ||
          slice.originalAmount.scale !== first.originalAmount.scale ||
          slice.amount.currency !== first.amount.currency || slice.amount.scale !== first.amount.scale) {
        invalidExpenses.add(expenseId);
        blockers.push(`inconsistent_expense_basis:${expenseId}`);
      }
    }
    const original = parseMinorUnits(first.originalAmount.minorUnits);
    if (original === null || total !== original) {
      invalidExpenses.add(expenseId);
      blockers.push(`period_slices_do_not_conserve:${expenseId}`);
    }
  }
  for (const expenseId of invalidExpenses) withheld.add(expenseId);
  const verifiedSlices = [...firstById.values()].filter(slice => !withheld.has(slice.expenseId));
  return { verifiedSlices, withheldExpenseIds: [...withheld].sort(), blockers: [...new Set(blockers)].sort() };
}
