/**
 * The digest operator (plans/neuralese/DECISIONS.md 43): a short Neuralese digest of a large value, shown in a call's
 * opening listing in place of the cut-off preview while the variable keeps the whole value for exact reading.
 *
 * A digest is a write, not an encoding: the model reads the full value at a write site conditioned on the receiving
 * call's instructions, and the stop head decides the digest's length. The server owns the plan
 * (training/neuralese/natlang_neuralese/digest.py, `POST /v1/neuralese/digest`): a value that fits the write site's
 * window (by default the model's whole context) is digested in one write; a longer one in token chunks whose digests a
 * final write combines. The trainer uses the same plan, so what is trained is what runs.
 *
 * The digest instructions are the prompt piece `digest`; under a system-prompt bank their soft form is sent instead.
 */
import { DIGEST_PROMPT } from '../builtin/index.js';
import { neuraleseRef, textToParts, type NeuraleseRef } from '../native/neuralese.js';
import type { NeuraleseStore } from '../native/neuralese-store.js';
import { activeSystemPrompts, softenText, type SystemPromptBank } from '../native/system-prompts.js';
import { fetchModel } from '../model/chat-completion.js';
import { HttpNeuraleseStore } from '../model/neuralese-server.js';

export const DIGEST_TYPE = 'Neuralese<Digest>';

/** What a digest is written from. */
export type DigestSite = { name: string; type: string; value: string; instructions: string };
/** Writes the digest of one value, or undefined to keep the preview. */
export type Digester = (site: DigestSite) => Promise<NeuraleseRef | undefined>;

/** A digester on a Neuralese server, with the soft digest instructions from `bank` when it has them. `window` caps
 * the value tokens of one write site (default: the server's, from the model's context). */
export function serverDigester(options: { endpoint: string; headers?: Record<string, string>; store?: NeuraleseStore;
  bank?: SystemPromptBank; window?: number }): Digester {
  const remote = new HttpNeuraleseStore(options.endpoint, options.headers);
  const base = options.endpoint.replace(/\/$/, '');
  return async site => {
    const bank = activeSystemPrompts(options.bank);
    const softened = bank?.size ? softenText(DIGEST_PROMPT, bank) : DIGEST_PROMPT;
    const system = softened === DIGEST_PROMPT ? DIGEST_PROMPT : textToParts(softened);
    if (Array.isArray(system)) for (const part of system) if (part.type === 'neuralese' && !(await remote.has(part.id))) {
      const block = await options.store?.get(part.id);
      if (!block) throw new Error(`neuralese-unknown-block: ${part.id} is neither on the server nor in the store`);
      const { id: _, ...rest } = block.meta;
      await remote.put({ ...rest, data: block.data });
    }
    const response = await fetchModel(`${base}/v1/neuralese/digest`, { method: 'POST',
      headers: { 'content-type': 'application/json', ...options.headers },
      body: JSON.stringify({ ...site, system, ...(options.window ? { window: options.window } : {}) }) });
    if (!response.ok) throw new Error(`digest failed (${response.status}): ${(await response.text()).slice(0, 300)}`);
    const meta = await response.json() as { id: string };
    if (options.store && !(await options.store.has(meta.id))) {
      const block = await remote.get(meta.id);
      if (!block) throw new Error(`neuralese-unknown-block: the server wrote ${meta.id} but does not have it`);
      const { id: _, ...rest } = block.meta;
      await options.store.put({ ...rest, data: block.data });
    }
    return neuraleseRef(DIGEST_TYPE, meta.id);
  };
}
