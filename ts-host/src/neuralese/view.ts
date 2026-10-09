/**
 * The Neuralese instance of the builtin `view(value, instructions?)` on a server (plans/neuralese/DECISIONS.md
 * 2026-10-09: "one summarizer family" and "representation chosen by use"). A view is a template write of view's body:
 * the model reads the value at view's write site, with the body as its system text, and its reply is forced to
 * `return_result` with the value written as a block; the stop head decides the length. Without instructions the view
 * is faithful compression, with them it keeps what their purpose needs.
 *
 * The server owns the plan (training/neuralese/natlang_neuralese/view.py, `POST /v1/neuralese/view`): a value that fits
 * the write site's window (by default the model's whole context) is viewed in one write; a longer one in token chunks
 * whose views a final write combines. The trainer writes through the same site and plan, so what is trained is what
 * runs. The runtime's opening listing shows such a view for each argument it would cut off, with the receiving call's
 * instructions as the view's (prompt.ts `listingViewInstructions`).
 *
 * The body is the prompt piece `view`; under a system-prompt bank its soft form is sent instead.
 */
import { VIEW_PROMPT } from '../builtin/index.js';
import { neuraleseRef, textToParts, type NeuraleseRef } from '../native/neuralese.js';
import type { NeuraleseStore } from '../native/neuralese-store.js';
import { activeSystemPrompts, softenText, type SystemPromptBank } from '../native/system-prompts.js';
import { fetchModel } from '../model/chat-completion.js';
import { HttpNeuraleseStore, withRestoredBlocks } from '../model/neuralese-server.js';

/** The type of a written view: the Neuralese instance of view's result. */
export const VIEW_TYPE = 'Neuralese<string>';

/** The arguments of one view call: the value, and what the view is for (absent: faithful). */
export type ViewSite = { value: string; instructions?: string };
/** Runs view's Neuralese instance on one value, or answers undefined to keep the crisp preview. */
export type Viewer = (site: ViewSite) => Promise<NeuraleseRef | undefined>;

/** View's Neuralese instance on a Neuralese server, with the soft body from `bank` when it has it. `window` caps the
 * value tokens of one write site (default: the server's, from the model's context). */
export function serverViewer(options: { endpoint: string; headers?: Record<string, string>; store?: NeuraleseStore;
  bank?: SystemPromptBank; window?: number }): Viewer {
  const remote = new HttpNeuraleseStore(options.endpoint, options.headers);
  const base = options.endpoint.replace(/\/$/, '');
  return async site => {
    const bank = activeSystemPrompts(options.bank);
    const softened = bank?.size ? softenText(VIEW_PROMPT, bank) : VIEW_PROMPT;
    const system = softened === VIEW_PROMPT ? VIEW_PROMPT : textToParts(softened);
    const ids = Array.isArray(system) ? system.flatMap(part => part.type === 'neuralese' ? [part.id] : []) : [];
    const meta = await withRestoredBlocks(remote, options.store, ids, async () => {
      const response = await fetchModel(`${base}/v1/neuralese/view`, { method: 'POST',
        headers: { 'content-type': 'application/json', ...options.headers },
        body: JSON.stringify({ value: site.value, ...(site.instructions ? { instructions: site.instructions } : {}), system,
          ...(options.window ? { window: options.window } : {}) }) });
      if (!response.ok) throw new Error(`view failed (${response.status}): ${(await response.text()).slice(0, 300)}`);
      return await response.json() as { id: string };
    });
    if (options.store && !(await options.store.has(meta.id))) {
      const block = await remote.get(meta.id);
      if (!block) throw new Error(`neuralese-unknown-block: the server wrote ${meta.id} but does not have it`);
      const { id: _, ...rest } = block.meta;
      await options.store.put({ ...rest, data: block.data });
    }
    return neuraleseRef(VIEW_TYPE, meta.id);
  };
}
