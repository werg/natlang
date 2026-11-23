/** Exact application checks live outside the model-visible natlang project. */
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { resolve, sep } from 'node:path';
import { pathToFileURL } from 'node:url';
import type { OracleVerdict } from '../evaluation/oracles.js';

export type TrustedExactVerifier = { path: string; sha256: string; oracle: unknown };
/** fixture is the world folder the run started from; files, when the task has a folder, is the folder it left. */
export type ExactVerifierInput = { value: unknown; oracle: unknown; inputs: Record<string, unknown>;
  fixture: Record<string, string>; files?: Record<string, string> };
/** score is graded credit in [0, 1] (`--reward graded` reads it); ok alone decides acceptance and held scoring. */
export type ExactVerifierResult = { ok: boolean; issues: string[]; score?: number; semantic_review_required?: boolean };
/** Repository roots a trusted verifier may live under: application invariants and authored semantic-FP tasks. */
export const TRUSTED_VERIFIER_ROOTS = ['training/semantic_applications/', 'training/semantic_fp_tasks/'] as const;

// The teacher executable may be frozen outside the checkout. Its working
// directory (or an explicit override) is the canonical repository checkout.
const root = resolve(process.env.NATLANG_REPO_ROOT ?? process.cwd());

export async function checkTrustedExactVerifier(spec: TrustedExactVerifier, input: ExactVerifierInput): Promise<OracleVerdict> {
  if (!TRUSTED_VERIFIER_ROOTS.some(prefix => spec.path.startsWith(prefix)) || !spec.path.endsWith('.mjs') ||
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
      (result.semantic_review_required !== undefined && typeof result.semantic_review_required !== 'boolean') ||
      (result.score !== undefined && !(typeof result.score === 'number' && result.score >= 0 && result.score <= 1)) ||
      (result.ok && result.score !== undefined && result.score !== 1))
    throw new Error(`invalid trusted exact verifier result: ${spec.path}`);
  return { accepted: result.ok, level: 'exact', comparison_version: 'trusted-application-invariants/1',
    ...(result.score !== undefined ? { score: result.score } : {}),
    ...(result.issues.length ? { verdict: result.issues.join('; ') } : {}),
    ...(result.semantic_review_required ? { needs_review: true } : {}) };
}
