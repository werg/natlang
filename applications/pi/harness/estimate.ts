/**
 * The tokens a request over `view` followed by `extra` holds (§8.3). Pluggable hot path, selected by the host setting
 * planning: `crisp` (default) pi-durable's estimateContext (through durable.estimate, over ai.estimateTokens); `nl` `estimate/rules.nl`;
 * `shadow` both, compared.
 */
import { pluggable } from '@natlang/node';
import { ai, durable } from 'natlang:services';
import rules from './estimate/rules.nl';
import type { ContextView, Message } from '../types.js';

/** Whether a message holds a Neuralese block (as in harness/context.ts). */
const holdsBlock = (message: { content?: unknown }) =>
  Array.isArray(message.content) && message.content.some(part => (part as { type?: unknown } | null)?.type === 'neuralese');

export default async function estimate(view: ContextView, extra: Message[]): Promise<number> {
  // Both sides estimate through ai.estimateTokens, which reads block lengths synchronously from the runtime's store:
  // make every block of view and extra known first (a no-op for a view from harness/context, which already did).
  const messages = [...view.messages, ...extra];
  if (messages.some(holdsBlock)) await ai.blockMeta(messages as never);
  return pluggable({ crisp: () => durable.estimate(view, extra), nl: () => rules(view, extra) }, await durable.implementation('planning'),
    { default: 'crisp', name: 'pi.estimate' })();
}
