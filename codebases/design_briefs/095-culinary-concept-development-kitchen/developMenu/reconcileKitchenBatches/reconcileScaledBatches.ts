export type RecipeLineExact = { ingredientId: string; quantityMinor: string; unit: string; yieldServings: string; };
export type ScaledBatch = { dishId: string; ingredientId: string; quantityMinor: string; unit: string; };

export type BatchReconciliation = {
  expected: ScaledBatch[];
  supplied: ScaledBatch[];
  discrepancies: string[];
  blockers: string[];
};

/** Exact integer scaling only; domain decisions and blocked-state handling remain in Natlang. */
export default function reconcileScaledBatches(
  covers: number,
  dishes: Array<{ id: string; recipe: RecipeLineExact[] }>,
  supplied: ScaledBatch[],
): BatchReconciliation {
  const blockers: string[] = [];
  if (!Number.isSafeInteger(covers) || covers <= 0) {
    return { expected: [], supplied, discrepancies: [], blockers: ["covers_must_be_a_positive_safe_integer"] };
  }
  const expected: ScaledBatch[] = [];
  for (const dish of dishes) for (const line of dish.recipe) {
    if (!/^\d+$/.test(line.quantityMinor) || !/^\d+$/.test(line.yieldServings)) {
      blockers.push(`malformed_nonnegative_integer:${dish.id}:${line.ingredientId}`);
      continue;
    }
    const quantity = BigInt(line.quantityMinor);
    const yieldServings = BigInt(line.yieldServings);
    if (quantity <= 0n) {
      blockers.push(`recipe_quantity_must_be_positive:${dish.id}:${line.ingredientId}`);
      continue;
    }
    if (yieldServings <= 0n) {
      blockers.push(`recipe_yield_must_be_positive:${dish.id}:${line.ingredientId}`);
      continue;
    }
    const numerator = quantity * BigInt(covers);
    if (numerator % yieldServings !== 0n) {
      blockers.push(`nonintegral_minor_unit_scaling:${dish.id}:${line.ingredientId}`);
      continue;
    }
    expected.push({ dishId: dish.id, ingredientId: line.ingredientId, quantityMinor: String(numerator / yieldServings), unit: line.unit });
  }
  const key = (batch: ScaledBatch) => `${batch.dishId}\u0000${batch.ingredientId}\u0000${batch.unit}`;
  const expectedByKey = new Map<string, ScaledBatch>();
  const suppliedByKey = new Map<string, ScaledBatch>();
  const discrepancies: string[] = [];
  for (const batch of expected) {
    const k = key(batch);
    if (expectedByKey.has(k)) discrepancies.push(`duplicate recipe line ${batch.dishId}/${batch.ingredientId}/${batch.unit}`);
    else expectedByKey.set(k, batch);
  }
  for (const batch of supplied) {
    const k = key(batch);
    if (suppliedByKey.has(k)) discrepancies.push(`duplicate supplied batch ${batch.dishId}/${batch.ingredientId}/${batch.unit}`);
    else suppliedByKey.set(k, batch);
  }
  for (const [k, batch] of expectedByKey) {
    const actual = suppliedByKey.get(k);
    if (!actual) discrepancies.push(`missing supplied batch ${batch.dishId}/${batch.ingredientId}/${batch.unit}`);
    else if (actual.quantityMinor !== batch.quantityMinor) discrepancies.push(`quantity mismatch ${batch.dishId}/${batch.ingredientId}/${batch.unit}`);
  }
  for (const [k, batch] of suppliedByKey) if (!expectedByKey.has(k)) discrepancies.push(`unmatched supplied batch ${batch.dishId}/${batch.ingredientId}/${batch.unit}`);
  return { expected, supplied, discrepancies, blockers: [...new Set(blockers)].sort() };
}
