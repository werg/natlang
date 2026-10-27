/**
 * The one resolution of a tuning root's host services: teacher collection, evaluation, RL rollouts and repairs all
 * create their runtime's services here.
 *
 * An SDK host may pass services for each root (`TuneOptions.services`: a test or sandbox world). A program may instead
 * provide its own, `tune.services.mjs` beside its `natlang.json`: its default export `(example) => services` (or a
 * promise of them) builds a root's services from the example (its `input` and `reference`), and an optional
 * `declarations` export (`{ name: declaration text }`) is what the model is shown of each service
 * (`serviceDeclarations`). The program's declarations apply to host-passed services too, so a test world and the real
 * services look the same to the model. `natlang tune` itself passes none: a program's services are its own file's.
 */
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import type { Services } from '../runtime/runtime.js';
import type { SplitExample } from './collect.js';

export const SERVICES_FILE = 'tune.services.mjs';
type Provided = { build: (example: SplitExample) => Services | Promise<Services>; declarations?: Record<string, string> };

const loaded = new Map<string, Promise<Provided | null>>();

async function providedBy(root: string): Promise<Provided | null> {
  const path = join(root, SERVICES_FILE);
  if (!existsSync(path)) return null;
  const digest = createHash('sha256').update(readFileSync(path)).digest('hex'), key = `${root}:${digest}`;
  if (!loaded.has(key)) loaded.set(key, (async () => {
    const module = await import(`${pathToFileURL(path).href}?sha256=${digest}`) as { default?: unknown; declarations?: unknown };
    if (typeof module.default !== 'function') throw new Error(`${SERVICES_FILE} must default-export (example) => services`);
    const declarations = module.declarations;
    if (declarations !== undefined && (typeof declarations !== 'object' || declarations === null
      || Object.values(declarations).some(text => typeof text !== 'string')))
      throw new Error(`${SERVICES_FILE} declarations must map each service name to its declaration text`);
    return { build: module.default as Provided['build'], ...(declarations ? { declarations: declarations as Record<string, string> } : {}) };
  })());
  return loaded.get(key)!;
}

/** A root's services and their declarations (both absent when neither the host nor the program provides any). */
export async function rootServices(root: string, given: ((example: SplitExample) => Services) | undefined, example: SplitExample):
  Promise<{ services?: Services; serviceDeclarations?: Record<string, string> }> {
  const provided = await providedBy(root);
  const services = given ? given(example) : provided ? await provided.build(example) : undefined;
  if (!services) return {};
  return { services, ...(provided?.declarations ? { serviceDeclarations: provided.declarations } : {}) };
}
