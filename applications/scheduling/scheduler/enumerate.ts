/**
 * Complete feasible schedules, up to `limit`. Pluggable hot path ("enumeration"): the exact depth-first search of the
 * workspace (`calendar.exact`) by default; with the setting "natural-language", `enumerate/construct.nl`, which builds
 * the same schedules from the domains.
 */
import { calendar } from 'natlang:services';
import construct from './enumerate/construct.nl';
import type { Domain, Hard, Offered } from '../types.js';

export default async function enumerate(domains: Domain[], order: string[], slot: number, hard: Hard, limit: number): Promise<Offered> {
  if (calendar.implementation('enumeration') === 'natural-language') return construct(domains, order, slot, limit);
  return calendar.exact(hard, limit);
}
