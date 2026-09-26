#!/usr/bin/env node
// Recipe discovery reads the family registry, not previously generated data files.
import { FAMILIES } from './families.mjs';

const tracks = new Map();
for (const [name, family] of Object.entries(FAMILIES)) {
  const track = family.track ?? 'interpreter';
  if (!tracks.has(track)) tracks.set(track, { id: track, generated_families: [], source_families: [] });
  tracks.get(track)[family.source ? 'source_families' : 'generated_families'].push(name);
}
console.log(JSON.stringify([...tracks.values()].sort((a, b) => a.id.localeCompare(b.id))));
