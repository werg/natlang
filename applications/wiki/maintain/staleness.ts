/**
 * Which recorded cell results survive an edit. Pluggable hot path (it runs after every merge): with
 * `settings.staleness` "crisp" the exact rule in `staleness/exact.ts`, with "natlang" `staleness/judge.nl`.
 */
import exact from './staleness/exact.ts';
import judge from './staleness/judge.nl';
import type { CellRecord, CellVerdict, Delta, WikiPage, WikiSettings } from '../types.js';

export default async function staleness(page: WikiPage, delta: Delta, records: CellRecord[], settings: WikiSettings): Promise<CellVerdict[]> {
  if (!records.length) return [];
  return settings.staleness === 'crisp' ? exact(page, delta, records) : judge(page, delta, records);
}
