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

/** Whether a message holds a Neuralese block. */
const holdsBlock = (message: { content?: unknown }) =>
  Array.isArray(message.content) && message.content.some(part => (part as { type?: unknown } | null)?.type === 'neuralese');

export default async function context(at?: number): Promise<ContextView> {
  const view = await pluggable({ crisp: () => durable.view(at), nl: () => derive(at) }, await durable.implementation('context'),
    { default: 'crisp', name: 'pi.context' })();
  // Every block of the context has its metadata in the runtime's store before anything estimates it (ai.estimateTokens
  // reads lengths synchronously); a block that stays unknown fails that estimate, naming it.
  if (view.messages.some(holdsBlock)) await ai.blockMeta(view.messages as never);
  return view;
}
