/**
 * Model-facing text variants (plans/MODEL_FACING_CHANGES.md).
 * Adopted variants are on by default; a new variant stays off until a live measurement shows its target behaviour moving.
 * NATLANG_MODEL_FACING overrides for a measurement arm: a comma list of variant names, `combined` (all variants) or
 * `none` (the texts as they were before any variant).
 */
/**
 * `learned-compaction`: the runtime's compaction sites use the learned operators (`compaction: nl`;
 * plans/neuralese/OPERATORS_RESTORE_20261117.md): a value over 8,000 characters is cut and its operator state shown
 * beside the cut. Adopted 2026-11-25 (owner: long values are summarized and encoded as Neuralese); the next held eval
 * shows any regression. Opt-outs: NATLANG_NEURALESE_COMPACTION=crisp, or NATLANG_MODEL_FACING without it.
 */
export const MODEL_FACING_VARIANTS = ['effect-receipt', 'learned-compaction', 'compact-callable-scope'] as const;
export type ModelFacingVariant = typeof MODEL_FACING_VARIANTS[number];
/** effect-receipt, compact-callable-scope: measured 2026-11-10 (plans/MODEL_FACING_CHANGES.md), neutral task success,
 * target behaviour moved. learned-compaction: adopted 2026-11-25 on first principles. */
const ADOPTED: readonly ModelFacingVariant[] = ['effect-receipt', 'compact-callable-scope', 'learned-compaction'];

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

/**
 * The model-facing prompt revision (plans/semantic-fp/02 R1): `current`, or `2026-11-22` for the texts and tool order
 * before the 2026-11-23 sentence pass, byte for byte (native/prompt-2026-11-22.ts). Set by NATLANG_PROMPT_REVISION, or
 * by a `prompt-revision-2026-11-22` entry in NATLANG_MODEL_FACING (held_eval.series() passes it on that way, so it is
 * part of the series key and reaches every collector).
 */
export const PROMPT_REVISIONS = ['current', '2026-11-22'] as const;
export type PromptRevision = typeof PROMPT_REVISIONS[number];
const revision: PromptRevision = (() => {
  if (typeof process === 'undefined') return 'current';
  const raw = (process.env.NATLANG_PROMPT_REVISION ?? '').trim();
  const named = raw || ((process.env.NATLANG_MODEL_FACING ?? '').split(',').map(name => name.trim())
    .find(name => name.startsWith('prompt-revision-'))?.slice('prompt-revision-'.length) ?? '');
  if (!named) return 'current';
  if (!(PROMPT_REVISIONS as readonly string[]).includes(named))
    throw new Error(`NATLANG_PROMPT_REVISION=${named} is not a prompt revision; use one of ${PROMPT_REVISIONS.join(', ')}`);
  return named as PromptRevision;
})();
export const promptRevision = (): PromptRevision => revision;
