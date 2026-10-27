/**
 * The one root score of program tuning: teacher collection, evaluation and student rollouts all score a root here.
 *
 * A root with `expected` and no verifier scores 1 for an equal result and 0 otherwise (exact oracle). A program may
 * instead declare a crisp verifier, `tune.reward.mjs` beside its `natlang.json`: its default export
 * `(value, example) => number | boolean` grades the root's result in [0, 1] (true/false = 1/0) and may use any reference
 * data the example carries (`expected`, `reference`). Graded scores are the RL rewards (group-relative advantages
 * use their spread); a score of exactly 1 is a success (a usable teacher demonstration, a correct evaluation root, an
 * RFT episode at the default threshold). A failed execution scores 0. The verifier's content digest binds every
 * score it produced, so a changed verifier never reuses old scores.
 */
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

export const VERIFIER_FILE = 'tune.reward.mjs';
export type RootScore = { readonly quality: number; readonly source: 'root-oracle' | 'root-verifier'; readonly verifier?: string };
type Verifier = (value: unknown, example: Readonly<Record<string, unknown>>) => unknown;

const equal = (left: unknown, right: unknown) => JSON.stringify(left) === JSON.stringify(right);
const loaded = new Map<string, Promise<{ verify: Verifier; digest: string } | null>>();

/** The verifier's content digest, or null when the program declares none. */
export function verifierDigest(root: string): string | null {
  const path = join(root, VERIFIER_FILE);
  return existsSync(path) ? createHash('sha256').update(readFileSync(path)).digest('hex') : null;
}

async function verifierOf(root: string) {
  const digest = verifierDigest(root);
  const key = `${root}:${digest}`;
  if (!loaded.has(key)) loaded.set(key, (async () => {
    if (digest === null) return null;
    const module = await import(`${pathToFileURL(join(root, VERIFIER_FILE)).href}?sha256=${digest}`) as { default?: unknown };
    if (typeof module.default !== 'function') throw new Error(`${VERIFIER_FILE} must default-export (value, example) => number | boolean`);
    return { verify: module.default as Verifier, digest };
  })());
  return loaded.get(key)!;
}

/** Whether a root can be scored at all (an exact oracle or the program's verifier). */
export async function scorable(root: string, example: object): Promise<boolean> {
  return 'expected' in example || (await verifierOf(root)) !== null;
}

/** The score of one root's result; null when the root has neither oracle nor verifier. */
export async function scoreRoot(root: string, example: Readonly<Record<string, unknown>>, value: unknown, failed: boolean): Promise<RootScore | null> {
  const verifier = await verifierOf(root);
  if (verifier) {
    if (failed) return { quality: 0, source: 'root-verifier', verifier: verifier.digest };
    const raw = verifier.verify(value, example);
    const quality = typeof raw === 'boolean' ? (raw ? 1 : 0) : raw;
    if (typeof quality !== 'number' || !Number.isFinite(quality) || quality < 0 || quality > 1)
      throw new Error(`${VERIFIER_FILE} returned ${JSON.stringify(raw)}; a root score is a number in [0, 1] or a boolean`);
    return { quality, source: 'root-verifier', verifier: verifier.digest };
  }
  if (!('expected' in example)) return null;
  return { quality: !failed && equal(value, example.expected) ? 1 : 0, source: 'root-oracle' };
}
