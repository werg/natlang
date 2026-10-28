/**
 * Loads the guidance-writer package (applications/guidance-writer) as a callable folder, the way the runtime loads any
 * natlang package: its two `.nl` definitions (with the sha256 of each file's bytes, the policy identity the engine
 * records) and its crisp helpers (visibility.ts, crispSelection.ts) run by the runtime's module loader. Node only.
 */
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { callableMeta } from '../runtime/callable.js';
import { loadCallables } from '../runtime/node-files.js';
import { TrainingGuidance, type GuidanceRun, type GuidanceWriterPackage } from '../native/training-guidance.js';

const sha256 = (bytes: string | Uint8Array) => createHash('sha256').update(bytes).digest('hex');

/** The package in this checkout (`NATLANG_GUIDANCE_WRITER` overrides). */
export function guidanceWriterDirectory(): string {
  const candidates = [process.env.NATLANG_GUIDANCE_WRITER, fileURLToPath(new URL('../../../applications/guidance-writer', import.meta.url))];
  const found = candidates.find(item => item && existsSync(join(item, 'writeGuidance.nl')));
  if (!found) throw Object.assign(new Error('guidance-writer-missing: the guidance-writer package was not found; ' +
    'set NATLANG_GUIDANCE_WRITER to its directory'), { code: 'guidance-writer-missing' });
  return resolve(found);
}

const cache = new Map<string, GuidanceWriterPackage>();
/** The package's definitions and crisp parts, loaded once per directory. */
export function loadGuidanceWriter(directory = guidanceWriterDirectory()): GuidanceWriterPackage {
  const cached = cache.get(directory);
  if (cached) return cached;
  const tree = loadCallables(directory, directory) as Record<string, any>;
  const definition = (name: string) => {
    const found = callableMeta(tree[name])?.definition;
    if (!found) throw new Error(`guidance-writer-missing: ${name}.nl did not load as a natural-language function`);
    return { definition: found, sha256: sha256(readFileSync(join(directory, `${name}.nl`))) };
  };
  const { visibleFacts } = tree.visibility as Pick<GuidanceWriterPackage, 'visibleFacts'>;
  const { crispSelection, CRISP_SELECTION_IDENTITY } = tree.crispSelection as
    Pick<GuidanceWriterPackage, 'crispSelection'> & { CRISP_SELECTION_IDENTITY: string };
  if (typeof visibleFacts !== 'function' || typeof crispSelection !== 'function' || typeof CRISP_SELECTION_IDENTITY !== 'string')
    throw new Error('guidance-writer-missing: visibility.ts or crispSelection.ts lacks its exports');
  const loaded: GuidanceWriterPackage = { selection: definition('selectGuidanceFacts'), writer: definition('writeGuidance'),
    visibleFacts, crispSelection, crispSha256: sha256(CRISP_SELECTION_IDENTITY) };
  cache.set(directory, loaded);
  return loaded;
}

/** One run's training guidance over the package in this checkout. */
export function trainingGuidance(run: GuidanceRun, directory?: string): TrainingGuidance {
  return new TrainingGuidance(run, loadGuidanceWriter(directory));
}
