/**
 * Complete feasible schedules, up to `limit`. Pluggable hot path ("enumeration"): the exact depth-first search of the
 * workspace (`calendar.exact`) by default; with the setting "nl" (the older "natural-language" is accepted),
 * `enumerate/construct.nl`, which builds the same schedules from the domains.
 */
import { pluggable } from '@natlang/node';
import { calendar } from 'natlang:services';
import construct from './enumerate/construct.nl';
import type { Domain, Hard, Offered } from '../types.js';

export default async function enumerate(domains: Domain[], order: string[], slot: number, hard: Hard, limit: number): Promise<Offered> {
  return pluggable({ crisp: () => calendar.exact(hard, limit), nl: () => construct(domains, order, slot, limit) },
    calendar.implementation('enumeration'), { name: 'scheduling.enumerate', default: 'crisp' })();
}
