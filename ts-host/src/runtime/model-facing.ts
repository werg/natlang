/**
 * Model-facing text variants (plans/MODEL_FACING_CHANGES.md, scripts/model_facing_ab.py).
 * Adopted variants are on by default; a new variant stays off until a live measurement shows its target behaviour moving.
 * NATLANG_MODEL_FACING overrides for a measurement arm: a comma list of variant names, `combined` (all variants) or
 * `none` (the texts as they were before any variant).
 */
export const MODEL_FACING_VARIANTS = ['effect-receipt', 'read-file-binding'] as const;
export type ModelFacingVariant = typeof MODEL_FACING_VARIANTS[number];
/** Measured 2026-11-10 (plans/MODEL_FACING_CHANGES.md): neutral task success, target behaviour moved. */
const ADOPTED: readonly ModelFacingVariant[] = ['effect-receipt', 'read-file-binding'];

function enabled(): ReadonlySet<string> {
  // The browser bundle has no process.
  const raw = (typeof process === 'undefined' ? '' : process.env.NATLANG_MODEL_FACING ?? '').trim();
  if (!raw) return new Set(ADOPTED);
  if (raw === 'none') return new Set();
  if (raw === 'combined') return new Set(MODEL_FACING_VARIANTS);
  return new Set(raw.split(',').map(name => name.trim()).filter(Boolean));
}

const active = enabled();
export const modelFacing = (name: ModelFacingVariant): boolean => active.has(name);
