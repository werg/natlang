/**
 * The model context of the task's conversation through entry `at` (default: the newest entry), as a ContextView
 * (§2.1). Pluggable (PORT.md, "Pluggable hot paths"): by default pi-durable's crisp derivation with its incremental
 * cache (`durable.view`); with the host setting context "natural-language", the raw active range (`durable.scan`)
 * derived by `deriveContext`, the natural-language statement of the same rules.
 */
import { durable } from 'natlang:services';
import deriveContext from './context/deriveContext.nl';
import type { ContextView } from '../types.js';

export default async function context(at?: number): Promise<ContextView> {
  if (durable.implementation('context') === 'crisp') return durable.view(at);
  const range = await durable.scan(at);
  return deriveContext(range.head, range.entries);
}
