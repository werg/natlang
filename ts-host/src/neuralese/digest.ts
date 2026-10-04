/**
 * The digest operator (plans/neuralese/DECISIONS.md 43): a short Neuralese digest of a large value, shown in a call's
 * opening listing in place of the cut-off preview while the variable keeps the whole value for exact reading.
 *
 * A digest is a write, not an encoding: the model reads the full value at a write site and the stop head decides how
 * long the digest is (a length cost trains it to be short, S3 phase E). The write site is purpose-conditioned by the
 * receiving call's instructions (decision 13: purpose lives at the write site, not in an argument):
 *
 *   system  the digest instructions (prompt piece `digest`, soft under a bank like every runtime prompt)
 *   user    the receiving call's instructions, then the value's name, type and full text
 *   reply   `const digest: Neuralese<Digest> = ` and the write
 *
 * `digestSite` builds that site; the training data converter and trainers build the same one from recorded digest
 * sites, so what is trained is what runs.
 */
import { DIGEST_PROMPT } from '../native/prompt.js';
import { encodeMessages, neuraleseRef, type NeuraleseRef } from '../native/neuralese.js';
import type { NeuraleseStore } from '../native/neuralese-store.js';
import { activeSystemPrompts, softenMessages, type SystemPromptBank } from '../native/system-prompts.js';
import { fetchModel } from '../model/chat-completion.js';
import { HttpNeuraleseStore, requestBlockIds } from '../model/neuralese-server.js';

export const DIGEST_TYPE = 'Neuralese<Digest>';
export const DIGEST_PREFIX = 'const digest: Neuralese<Digest> = ';
/** Values longer than this are cut for the write site (a chunked digest is a later step). */
export const DIGEST_SOURCE_CHARS = 48_000;

/** What a digest is written from. */
export type DigestSite = { name: string; type: string; value: string; instructions: string };
/** Writes the digest of one value, or undefined to keep the preview. */
export type Digester = (site: DigestSite) => Promise<NeuraleseRef | undefined>;

/** The write site's messages. */
export function digestSite(site: DigestSite): { role: string; content: string }[] {
  const value = site.value.length > DIGEST_SOURCE_CHARS ?
    site.value.slice(0, DIGEST_SOURCE_CHARS) + ` <<cut off: ${site.value.length - DIGEST_SOURCE_CHARS} of ${site.value.length} characters not shown>>` : site.value;
  return [{ role: 'system', content: DIGEST_PROMPT },
    { role: 'user', content: `The call that receives the value has these instructions:\n${site.instructions}\n\n` +
      `The value of ${site.name} (${site.type}):\n${value}` }];
}

/** A digester that writes on a Neuralese server (`/v1/neuralese/write`), with soft prompt pieces from `bank`. */
export function serverDigester(options: { endpoint: string; headers?: Record<string, string>; store?: NeuraleseStore;
  bank?: SystemPromptBank }): Digester {
  const remote = new HttpNeuraleseStore(options.endpoint, options.headers);
  return async site => {
    const encoded = encodeMessages(softenMessages(digestSite(site), activeSystemPrompts(options.bank)));
    for (const id of requestBlockIds(encoded.messages)) {
      if (await remote.has(id)) continue;
      const block = await options.store?.get(id);
      if (!block) throw new Error(`neuralese-unknown-block: ${id} is neither on the server nor in the store`);
      const { id: _, ...rest } = block.meta;
      await remote.put({ ...rest, data: block.data });
    }
    const response = await fetchModel(options.endpoint.replace(/\/$/, '') + '/v1/neuralese/write', { method: 'POST',
      headers: { 'content-type': 'application/json', ...options.headers },
      body: JSON.stringify({ messages: encoded.messages, prefix: DIGEST_PREFIX }) });
    if (!response.ok) throw new Error(`digest write failed (${response.status}): ${(await response.text()).slice(0, 300)}`);
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
