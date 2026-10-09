/**
 * Which recorded cell results survive an edit. Pluggable hot path (it runs after every merge): with
 * `settings.staleness` "crisp" the exact rule in `staleness/exact.ts`, with "nl" (or the older "natlang") `staleness/judge.nl`,
 * and "shadow" runs both.
 */
import { pluggable } from '@natlang/node';
import exact from './staleness/exact.ts';
import judge from './staleness/judge.nl';
import type { CellRecord, CellVerdict, Delta, WikiPage, WikiSettings } from '../types.js';

export default async function staleness(page: WikiPage, delta: Delta, records: CellRecord[], settings: WikiSettings): Promise<CellVerdict[]> {
  if (!records.length) return [];
  return pluggable({ crisp: () => exact(page, delta, records), nl: () => judge(page, delta, records) }, settings.staleness,
    { name: 'wiki.staleness' })();
}
