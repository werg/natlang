/**
 * The exact verifiers of the stages' outputs. Each returns the problems as sentences (empty when the value is fine); a
 * stage that fails is asked once more with them, and the publisher rejects the brief when the second try fails too.
 */
import type { Outline, Passage, Table } from './types.js';

const numberTokens = /\d+(?:[.,]\d+)*/g;

/** The numbers a text states, with thousands separators removed ("1,200" is 1200) so that spellings compare equal. */
export function numbersIn(text: string): string[] {
  return (String(text).match(numberTokens) ?? []).map(token => /^\d{1,3}(?:,\d{3})+(?:\.\d+)?$/.test(token) ? token.replace(/,/g, '') : token);
}

/** Numbers of `body` that occur neither in a passage nor in a cell of `table`. */
export function unsourcedNumbers(body: string, passages: Passage[], table: Table | null): string[] {
  const known = new Set<string>();
  for (const passage of passages) for (const number of numbersIn(passage.text)) known.add(number);
  if (table) for (const cell of [...table.columns, ...table.rows.flat()]) for (const number of numbersIn(String(cell))) known.add(number);
  return [...new Set(numbersIn(body).filter(number => !known.has(number)))];
}

export type Offered = { passage_ids: string[], table_ids: string[], asset_ids: string[] };

/** Whether an outline uses only offered ids, gives each table and asset to one section, and has a section. */
export function outlineProblems(outline: Outline, offered: Offered): string[] {
  const problems: string[] = [];
  if (!outline || typeof outline.title !== 'string' || !outline.title.trim()) problems.push('the outline needs a title');
  if (!outline || !Array.isArray(outline.sections) || !outline.sections.length) return [...problems, 'the outline needs at least one section'];
  const passages = new Set(offered.passage_ids), tables = new Set(offered.table_ids), assets = new Set(offered.asset_ids);
  const tableUse = new Map<string, number>(), assetUse = new Map<string, number>();
  outline.sections.forEach((section, index) => {
    const label = `section ${index + 1} (${JSON.stringify(section.heading)})`;
    if (typeof section.heading !== 'string' || !section.heading.trim()) problems.push(`${label} needs a heading`);
    const stray = (section.passage_ids ?? []).filter(id => !passages.has(id));
    if (stray.length) problems.push(`${label} names passage ids that were not offered: ${stray.join(', ')}; the offered ids are ${offered.passage_ids.join(', ')}`);
    if (section.table_id !== null && section.table_id !== undefined) {
      if (!tables.has(section.table_id)) problems.push(`${label} names table ${JSON.stringify(section.table_id)}, which was not offered; the offered tables are ${offered.table_ids.join(', ') || '(none)'}`);
      tableUse.set(section.table_id, (tableUse.get(section.table_id) ?? 0) + 1);
    }
    for (const id of section.asset_ids ?? []) {
      if (!assets.has(id)) problems.push(`${label} names asset ${JSON.stringify(id)}, which was not offered; the offered assets are ${offered.asset_ids.join(', ') || '(none)'}`);
      assetUse.set(id, (assetUse.get(id) ?? 0) + 1);
    }
  });
  for (const [id, count] of tableUse) if (count > 1) problems.push(`table ${id} is assigned to ${count} sections; give it to one`);
  for (const [id, count] of assetUse) if (count > 1) problems.push(`asset ${id} is assigned to ${count} sections; give it to one`);
  return problems;
}
