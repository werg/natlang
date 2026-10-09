/**
 * The model context of the task's conversation through entry `at` (default: the newest entry), as a ContextView
 * (§2.1). Pluggable (PORT.md, "Pluggable hot paths"), selected by the host setting context: `crisp` (default)
 * pi-durable's derivation with its incremental cache (`durable.view`); `nl` the raw active range (`durable.scan`)
 * derived by `deriveContext`, the natural-language statement of the same rules; `shadow` both, compared.
 */
import { pluggable } from '@natlang/node';
import { ai, durable } from 'natlang:services';
import deriveContext from './context/deriveContext.nl';
import type { ContextView } from '../types.js';

/** The derivation from the raw active range. */
async function derive(at?: number): Promise<ContextView> {
  const range = await durable.scan(at);
  return deriveContext(range.head, range.entries);
}

/** Whether a message holds a Neuralese block, or a stored call (a text part with a stored field), which a Neuralese reader reads as one. */
const holdsBlock = (message: { content?: unknown }) => Array.isArray(message.content) && message.content.some(part => {
  const found = part as { type?: unknown; stored?: unknown } | null;
  return found?.type === 'neuralese' || (found?.type === 'text' && found.stored !== undefined);
});

export default async function context(at?: number): Promise<ContextView> {
  const view = await pluggable({ crisp: () => durable.view(at), nl: () => derive(at) }, await durable.implementation('context'),
    { default: 'crisp', name: 'pi.context' })();
  // Every block of the context has its metadata in the runtime's store, and every stored call is forced for the agent's
  // reader, before anything estimates it (ai.estimateTokens reads lengths synchronously); a block or call that stays
  // unknown fails that estimate, naming it.
  if (view.messages.some(holdsBlock)) await ai.blockMeta(view.messages as never);
  return view;
}
