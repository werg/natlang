/** Exact application checks live outside the model-visible natlang project. */
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { resolve, sep } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import type { OracleVerdict } from '../evaluation/oracles.js';

export type TrustedExactVerifier = { path: string; sha256: string; oracle: unknown };
export type ExactVerifierInput = { value: unknown; oracle: unknown; inputs: Record<string, unknown>;
  fixture: Record<string, string> };
export type ExactVerifierResult = { ok: boolean; issues: string[]; semantic_review_required?: boolean };

const root = resolve(fileURLToPath(new URL('../../../', import.meta.url)));

export async function checkTrustedExactVerifier(spec: TrustedExactVerifier, input: ExactVerifierInput): Promise<OracleVerdict> {
  if (!spec.path.startsWith('training/semantic_applications/') || !spec.path.endsWith('.mjs') ||
      spec.path.split('/').some(part => !part || part === '.' || part === '..') ||
      !/^[a-f0-9]{64}$/.test(spec.sha256)) throw new Error('invalid trusted exact verifier binding');
  const path = resolve(root, spec.path);
  if (!path.startsWith(root + sep)) throw new Error('trusted exact verifier escapes the repository');
  const source = await readFile(path);
  if (createHash('sha256').update(source).digest('hex') !== spec.sha256)
    throw new Error(`trusted exact verifier source changed: ${spec.path}`);
  const module = await import(pathToFileURL(path).href) as { verifyCandidate?: (input: ExactVerifierInput) =>
    ExactVerifierResult | Promise<ExactVerifierResult> };
  if (typeof module.verifyCandidate !== 'function')
    throw new Error(`trusted exact verifier lacks verifyCandidate: ${spec.path}`);
  const result = await module.verifyCandidate(input);
  if (typeof result?.ok !== 'boolean' || !Array.isArray(result.issues) ||
      result.issues.some(issue => typeof issue !== 'string') ||
      (result.semantic_review_required !== undefined && typeof result.semantic_review_required !== 'boolean'))
    throw new Error(`invalid trusted exact verifier result: ${spec.path}`);
  return { accepted: result.ok, level: 'exact', comparison_version: 'trusted-application-invariants/1',
    ...(result.issues.length ? { verdict: result.issues.join('; ') } : {}),
    ...(result.semantic_review_required ? { needs_review: true } : {}) };
}
