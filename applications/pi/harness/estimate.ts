/**
 * The tokens a request over `view` followed by `extra` holds (§8.3). Pluggable hot path, selected by the host setting
 * planning: `crisp` (default) pi-durable's estimateContext (through durable.estimate, over ai.estimateTokens); `nl` `estimate/rules.nl`;
 * `shadow` both, compared.
 */
import { pluggable } from '@natlang/node';
import { ai, durable } from 'natlang:services';
import rules from './estimate/rules.nl';
import type { ContextView, Message } from '../types.js';

/** Whether a message holds a Neuralese block or a stored call (as in harness/context.ts). */
const holdsBlock = (message: { content?: unknown }) => Array.isArray(message.content) && message.content.some(part => {
  const found = part as { type?: unknown; stored?: unknown } | null;
  return found?.type === 'neuralese' || (found?.type === 'text' && found.stored !== undefined);
});

export default async function estimate(view: ContextView, extra: Message[]): Promise<number> {
  // Both sides estimate through ai.estimateTokens, for the agent's reader, which reads block and forced-view lengths
  // synchronously: make every block and stored call of view and extra known first (a no-op for a view from
  // harness/context, which already did).
  const messages = [...view.messages, ...extra];
  if (messages.some(holdsBlock)) await ai.blockMeta(messages as never);
  return pluggable({ crisp: () => durable.estimate(view, extra), nl: () => rules(view, extra) }, await durable.implementation('planning'),
    { default: 'crisp', name: 'pi.estimate' })();
}
