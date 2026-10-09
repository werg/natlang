/** The crisp staleness rule: any difference between the two pages makes every recorded result stale. */
import type { CellRecord, CellVerdict, Delta, WikiPage } from '../../types.js';

export default function exact(_page: WikiPage, delta: Delta, records: CellRecord[]): CellVerdict[] {
  const edited = delta.changed.length + delta.added.length + delta.removed.length > 0;
  return records.map(record => edited
    ? { block_id: record.block_id, status: 'stale', reason: 'the page changed' }
    : { block_id: record.block_id, status: 'fresh', reason: 'the page is unchanged' });
}
