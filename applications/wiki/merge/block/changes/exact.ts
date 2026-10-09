/** The crisp summary of an update: text comparison only. It reads no meaning, so its intent is the new text itself. */
import type { Change, WikiBlock, WikiUpdate } from '../../../types.js';

const squash = (text: string) => text.replace(/\s+/g, ' ').trim();
const lines = (text: string) => text.split('\n').map(line => line.trim()).filter(Boolean);

export default function exact(block: WikiBlock, update: WikiUpdate): Change {
  const old = block.text, text = update.text;
  const base = { update_id: update.id, block_id: update.block_id, author: update.author, text };
  if (text === old) return { ...base, kind: 'noop', intent: 'Keep the block as it is.', adds: [], removes: [] };
  if (squash(text) === squash(old)) return { ...base, kind: 'format', intent: 'Respace the block.', adds: [], removes: [] };
  const before = lines(old), after = lines(text);
  const adds = after.filter(line => !before.includes(line)), removes = before.filter(line => !after.includes(line));
  const kind = squash(text).startsWith(squash(old)) ? 'extend' : squash(old).startsWith(squash(text)) ? 'trim' : 'rewrite';
  return { ...base, kind, intent: `Make the block read: ${squash(text).slice(0, 200)}`, adds, removes };
}
