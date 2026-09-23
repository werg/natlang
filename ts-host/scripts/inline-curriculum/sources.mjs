// Adapters from pinned external sources to curriculum cases. Source answers and formal annotations stay
// in the oracle block; the model sees only the natural-language premises through a paged store.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { Random, curriculumCase, evalCall, returnCall } from './lib.mjs';
import { factStore } from './logic.mjs';
import { SOURCES, cachePath } from './acquire.mjs';

const CACHE = process.env.NATLANG_DATASETS ?? fileURLToPath(new URL('../../../vendor/datasets', import.meta.url));
const loaded = new Map();
function folioRows() {
  if (loaded.has('folio')) return loaded.get('folio');
  const source = SOURCES.folio;
  const rows = source.files.flatMap(file => {
    let text;
    try { text = readFileSync(cachePath(CACHE, 'folio', source.revision, file.path), 'utf8'); }
    catch { throw new Error('FOLIO is not in the dataset cache; run node scripts/inline-curriculum/acquire.mjs --source folio'); }
    return text.split('\n').filter(line => line.trim()).map(line => ({ ...JSON.parse(line), split: file.split }));
  });
  loaded.set('folio', rows);
  return rows;
}

// v0.0 spells the third label both Uncertain and Unknown.
const VERDICT = { True: 'entailed', False: 'contradicted', Uncertain: 'unknown', Unknown: 'unknown' };
const names = text => new Set((text.match(/\b[A-Z][a-z]+\b/g) ?? []).filter(word => !['All', 'No', 'Some', 'If', 'Every', 'Each', 'There', 'The', 'People', 'Either', 'Any', 'A', 'An'].includes(word)));

/**
 * FOLIO: expert-written premises and a conclusion labelled True, False, or Uncertain. Premises are mixed
 * with those of an unrelated story in a paged store; validation stories are held out as test cases.
 */
export function folioEntailment(seed, index) {
  const rows = folioRows();
  const row = rows[index % rows.length];
  const rng = new Random(seed, `folio:${row.example_id}`);
  const own = names([...row.premises, row.conclusion].join(' '));
  const unrelated = rng.shuffle(rows.filter(other => other.story_id !== row.story_id && other.split === row.split))
    .find(other => ![...names(other.premises.join(' '))].some(word => own.has(word)));
  const premises = row.premises.map(text => text.trim());
  const facts = rng.shuffle([...premises, ...(unrelated?.premises ?? []).map(text => text.trim())])
    .map((text, i) => ({ id: `P${i + 1}`, text }));
  if (!VERDICT[row.label]) throw new Error(`FOLIO example ${row.example_id}: unexpected label ${row.label}`);
  const expected = { verdict: VERDICT[row.label] };
  return [curriculumCase({ family: 'folio_entailment', shape: `story${row.story_id}`, variant: `ex${row.example_id}`,
    splitGroup: `folio:story:${row.story_id}`, split: row.split === 'train' ? 'train' : 'test',
    slice: 'observation_followup', domain: 'logic', mode: 'followup', worldSemantics: 'open_world',
    evidence: { world: premises, retrieved: premises, background: [`FOL: ${row['premises-FOL'].join(' ; ')}`, `label: ${row.label}`, `source: FOLIO ${SOURCES.folio.revision}`] },
    assumptions: ['Only the stored premises hold; a statement they neither establish nor refute is unknown.'],
    decisive: premises.map(text => ({ marker: text, source: 'eval', note: 'a premise of the story' })),
    plausibleActions: ['answer entailed', 'answer contradicted', 'answer unknown'],
    minimumSequence: ['read the premise store', 'reason from the relevant premises alone'],
    reference: { root: [...Array.from({ length: Math.ceil(facts.length / 8) }, (_, page) =>
      evalCall(`for (const f of facts.page(${page + 1})) console.log(f.id + ': ' + f.text);`)), returnCall(expected)] },
    root: { name: 'evaluate_conclusion', args: { conclusion: 'string' }, returns: 'Evaluation',
      instructions: `Decide whether conclusion follows from the premises in the store (facts), using only those premises and no outside knowledge.
The store also holds premises about unrelated matters.
verdict is "entailed" if the premises make the conclusion true, "contradicted" if they make it false, and "unknown" if they settle neither.` },
    files: { 'evaluate_conclusion/facts.ts': factStore(facts, 'The premise store.'),
      'types.ts': 'export type Evaluation = { verdict: "entailed" | "contradicted" | "unknown" };\n' },
    inputs: { conclusion: row.conclusion.trim() }, expected })];
}
export const folioCount = () => folioRows().length;
