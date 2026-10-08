/**
 * The tokens a request over `view` followed by `extra` holds (§8.3). Pluggable hot path ("planning"): pi-durable's
 * crisp estimateContext (through durable.estimate) by default; with planning "natural-language", `estimate/rules.nl`.
 */
import { durable } from 'natlang:services';
import rules from './estimate/rules.nl';
import type { ContextView, Message } from '../types.js';

export default async function estimate(view: ContextView, extra: Message[]): Promise<number> {
  if (durable.implementation('planning') === 'natural-language') return rules(view, extra);
  return durable.estimate(view, extra);
}
