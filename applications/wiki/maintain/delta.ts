/** The blocks whose source differs between two pages: text, kind, language or return type. Plumbing. */
import type { Delta, WikiBlock, WikiPage } from '../types.js';

const source = (block: WikiBlock) => JSON.stringify([block.kind, block.language ?? null, block.returns ?? null, block.text]);

export default function delta(before: WikiPage, after: WikiPage): Delta {
  const old = new Map(before.blocks.map(block => [block.id, source(block)]));
  const now = new Map(after.blocks.map(block => [block.id, source(block)]));
  return {
    changed: after.blocks.filter(block => old.has(block.id) && old.get(block.id) !== now.get(block.id)).map(block => block.id),
    added: after.blocks.filter(block => !old.has(block.id)).map(block => block.id),
    removed: before.blocks.filter(block => !now.has(block.id)).map(block => block.id),
  };
}
