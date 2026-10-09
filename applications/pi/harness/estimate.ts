/**
 * The tokens a request over `view` followed by `extra` holds (§8.3). Pluggable hot path, selected by the host setting
 * planning: `crisp` (default) pi-durable's estimateContext (through durable.estimate); `nl` `estimate/rules.nl`;
 * `shadow` both, compared.
 */
import { pluggable } from '@natlang/node';
import { durable } from 'natlang:services';
import rules from './estimate/rules.nl';
import type { ContextView, Message } from '../types.js';

export default async function estimate(view: ContextView, extra: Message[]): Promise<number> {
  return pluggable({ crisp: () => durable.estimate(view, extra), nl: () => rules(view, extra) }, await durable.implementation('planning'),
    { default: 'crisp', name: 'pi.estimate' })();
}
