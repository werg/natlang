/** Lex a page for its structure: the heading blocks and the [[links]] in prose. Plumbing: no decision is made here. */
import type { Heading, LinkRef, Scan, WikiPage } from '../types.js';

const HEADING = /^(#{1,6})\s+(.+?)\s*#*\s*$/;
const LINK = /\[\[([^\]|]+)(?:\|([^\]]*))?\]\]/g;

export default function scan(page: WikiPage): Scan {
  const headings: Heading[] = [], links: LinkRef[] = [];
  for (const block of page.blocks) {
    if (block.kind !== 'prose') continue;
    const first = block.text.split('\n').find(line => line.trim())?.trim() ?? '';
    const match = HEADING.exec(first);
    if (match) headings.push({ block_id: block.id, level: match[1]!.length, title: match[2]! });
    for (const link of [...block.text.matchAll(LINK)]) links.push({ block_id: block.id, target: link[1]!.trim(), text: (link[2] ?? link[1]!).trim() });
  }
  return { headings, links };
}
