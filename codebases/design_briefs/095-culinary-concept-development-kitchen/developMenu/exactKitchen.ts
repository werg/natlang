export type RecipeLineExact = { dishId: string; ingredientId: string; quantityMinor: string; unit: string; yieldServings: string; };
export type ScaledBatch = { dishId: string; ingredientId: string; quantityMinor: string; unit: string; };
export function scaleRecipe(lines: RecipeLineExact[], covers: string): ScaledBatch[] {
  const target = BigInt(covers);
  if (target <= 0n) throw new Error("covers must be positive");
  return lines.map((line) => {
    const base = BigInt(line.yieldServings);
    if (base <= 0n) throw new Error("recipe yield must be positive");
    const numerator = BigInt(line.quantityMinor) * target;
    if (numerator % base !== 0n) throw new Error("exact quantity needs an approved rounding rule");
    return { dishId: line.dishId, ingredientId: line.ingredientId, quantityMinor: String(numerator / base), unit: line.unit };
  });
}
