/**
 * Summarize one update against the block it edits. Pluggable hot path (it runs once per update): with
 * `settings.changes` "crisp" the exact comparison in `changes/exact.ts`, with "natlang" `changes/summarize.nl`.
 */
import exact from './changes/exact.ts';
import summarize from './changes/summarize.nl';
import type { Change, WikiBlock, WikiSettings, WikiUpdate } from '../../types.js';

export default async function changes(block: WikiBlock, update: WikiUpdate, settings: WikiSettings): Promise<Change> {
  return settings.changes === 'crisp' ? exact(block, update) : summarize(block, update);
}
