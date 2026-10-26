/**
 * Automatic tuned-weight discovery for `natlang run`/`call` (plans/HIERARCHICAL_LORA_TUNING.md §1.1, §6): the
 * program's active release is found beside it and served on the selected backend without flags or manifests.
 * A release the backend cannot serve runs the base model, marked degraded with the reason; never as tuned execution.
 */
import { createReadStream } from 'node:fs';
import { createHash } from 'node:crypto';
import { join } from 'node:path';
import type { ProgramDescriptor } from '../adaptation/types.js';
import { WeightBinding } from '../adaptation/weights/binding.js';
import { currentRelease } from '../adaptation/weights/release.js';
import { managedModelSha256 } from '../model/local-server.js';
import type { ResolvedModelChoice } from '../model/config.js';

export type TunedServing = {
  /** For `createNatlangRuntime({ weights })`. */
  weights?: readonly WeightBinding[];
  /** For the managed llama.cpp session: the GGUF adapters to load. */
  loraAdapters?: () => readonly { id: string; path: string }[];
  /** One author-facing line when the program is tuned but not served as tuned, else undefined. */
  notice?: string;
};

async function fileSha256(path: string): Promise<string> {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
}

export async function tunedServing(root: string, program: ProgramDescriptor | undefined, choice: ResolvedModelChoice,
  environment: NodeJS.ProcessEnv = process.env): Promise<TunedServing> {
  const release = currentRelease(root);
  if (!release) return {};
  const { manifest, directory } = release;
  let incompatible: string | undefined;
  const loaded: { id: string; path: string }[] = [];
  if (choice.kind !== 'managed-local') incompatible = `the ${choice.kind} backend does not serve tuned weights; use the managed local model or the native server`;
  else {
    const base = await managedModelSha256(environment);
    const exported = manifest.artifacts.filter(item => item.exports?.gguf);
    const families = manifest.families.map(family => manifest.artifacts.find(item => item.artifact === family.artifact)).filter(Boolean);
    if (!exported.length && !families.length) incompatible = 'this release has no llama.cpp export; rerun natlang tune for the managed backend';
    else if (exported.some(item => item.exports!.gguf!.baseGgufSha256 !== base))
      incompatible = 'the tuned adapters were exported for another base model file; rerun natlang tune';
    else for (const item of exported) {
      const path = join(directory, item.exports!.gguf!.file);
      if (await fileSha256(path).catch(() => '') !== item.exports!.gguf!.sha256)
        throw new Error(`tuned adapter ${item.exports!.gguf!.file} is missing or differs from its release receipt; rerun natlang tune`);
      loaded.push({ id: item.artifact, path });
    }
  }
  const binding = new WeightBinding(manifest, { ...(program ? { program } : {}), available: new Set(loaded.map(item => item.id)),
    ...(incompatible ? { incompatible } : {}) });
  return { weights: [binding], ...(loaded.length ? { loraAdapters: () => loaded } : {}),
    ...(binding.status !== 'ready' ? { notice: `natlang: tuned weights ${binding.status}: ${binding.reason}` } : {}) };
}
