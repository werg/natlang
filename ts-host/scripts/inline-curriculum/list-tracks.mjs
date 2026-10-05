#!/usr/bin/env node
// Recipe discovery reads the family registry, not previously generated data files.
import { FAMILIES } from './families.mjs';

const tracks = new Map();
// Demonstration-only families (static reference replays, no teacher oracle) are never collected by a recipe.
for (const [name, family] of Object.entries(FAMILIES)) {
  if (family.demonstration) continue;
  const track = family.track ?? 'interpreter';
  if (!tracks.has(track)) tracks.set(track, { id: track, generated_families: [], source_families: [] });
  tracks.get(track)[family.source ? 'source_families' : 'generated_families'].push(name);
}
console.log(JSON.stringify([...tracks.values()].sort((a, b) => a.id.localeCompare(b.id))));
