/** The `fusion` block of natlang.json (plans/FUSED_PIPELINES.md), parsed and validated. */
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { FUSION_CERTIFICATE_SCHEMA, type FusionCertificate, type FusionMode } from '../runtime/fusion.js';
import type { FusionPlannerMode } from './planner.js';

export type FusionSettings = {
  /** `off` (default): every hand-off is text. `shadow`: text is served, the fused path is measured. `on`: fused where certified. */
  mode: FusionMode;
  /** Who plans: `crisp` (default), `nl` (applications/fusion-planner, verified) or `shadow` (both, crisp served). */
  planner: FusionPlannerMode;
  /** Path (from the package root) of the certificate the training pipeline issued for the profile's weights. */
  certificate?: string;
  /** Digest of the weights the profile serves; the certificate must name the same. */
  weights?: string;
};

export function parseFusionSettings(raw: unknown): FusionSettings {
  if (raw === undefined) return { mode: 'off', planner: 'crisp' };
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) throw new TypeError('fusion must be an object');
  const fields = raw as Record<string, unknown>;
  for (const key of Object.keys(fields)) if (!['mode', 'planner', 'certificate', 'weights'].includes(key)) throw new TypeError(`unknown fusion field: ${key}`);
  const mode = fields.mode ?? 'off', planner = fields.planner ?? 'crisp';
  if (mode !== 'off' && mode !== 'shadow' && mode !== 'on') throw new TypeError('fusion.mode must be "off", "shadow" or "on"');
  if (planner !== 'crisp' && planner !== 'nl' && planner !== 'shadow') throw new TypeError('fusion.planner must be "crisp", "nl" or "shadow"');
  for (const key of ['certificate', 'weights'] as const)
    if (fields[key] !== undefined && (typeof fields[key] !== 'string' || !fields[key])) throw new TypeError(`fusion.${key} must be a nonempty string`);
  return { mode, planner, ...(fields.certificate ? { certificate: fields.certificate as string } : {}),
    ...(fields.weights ? { weights: fields.weights as string } : {}) };
}

/** Read the certificate named by the settings; `undefined` when none is named or the file is absent or malformed. */
export function readFusionCertificate(root: string, settings: FusionSettings): FusionCertificate | undefined {
  if (!settings.certificate) return undefined;
  const path = resolve(root, settings.certificate);
  if (!existsSync(path)) return undefined;
  try {
    const value = JSON.parse(readFileSync(path, 'utf8')) as FusionCertificate;
    return value?.schema === FUSION_CERTIFICATE_SCHEMA ? value : undefined;
  } catch { return undefined; }
}
