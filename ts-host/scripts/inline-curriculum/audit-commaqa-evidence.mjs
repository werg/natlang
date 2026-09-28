// Offline audit of the adapter's evidence against the original symbolic KB and language configuration.
import { readFileSync, writeFileSync } from 'node:fs';
import { numericNationalityFact } from './sources-ai2.mjs';

export function auditEvidence(worlds, split) {
  const failures = [], rawInversions = [];
  let tableFacts = 0, throwFacts = 0, questions = 0;
  worlds.forEach((world, index) => {
    const fail = (kind, fact) => failures.push({ split, world: index, kind, fact });
    const kb = new Set(Object.values(world.kb).flat());
    for (const sport of ['d', 'j']) {
      const name = sport === 'd' ? 'discus' : 'javelin';
      const mapping = world.pred_lang_config.table.find(item => item.predicate === `nation${sport}_p($1, ?)`);
      if (!mapping?.questions.every(q => q.toLowerCase().includes(name)) ||
          mapping.steps[0]?.question !== `table_nation${sport}(?, $1)`) fail('language_mapping', sport);
    }
    for (const [fact, sentence] of Object.entries(world.per_fact_context)) {
      if (!kb.has(fact)) fail('rendered_fact_absent_from_kb', fact);
      if (fact.startsWith('table_')) {
        tableFacts++;
        const [, sport, athlete, country] = /^table_nation([dj])\(([^,]+), (.+)\)$/.exec(fact) ?? [];
        if (!sport) { fail('unsupported_table_fact', fact); continue; }
        const corrected = numericNationalityFact(fact);
        const expected = `athlete: ${athlete} ; country: ${country} ; sport: ${sport === 'd' ? 'Discus' : 'Javelin'} Throw.`;
        if (corrected !== expected) fail('corrected_rendering', fact);
        const opposite = sport === 'd' ? 'javelin' : 'discus';
        if (sentence.toLowerCase().includes(opposite)) rawInversions.push({ split, world: index, fact });
      } else if (fact.startsWith('text_')) {
        throwFacts++;
        const [, sport, athlete, length] = /^text_([dj])throw\(([^,]+), (.+)\)$/.exec(fact) ?? [];
        if (!sport || !sentence.includes(athlete) || !sentence.includes(length) ||
            !sentence.toLowerCase().includes(sport === 'd' ? 'discus' : 'javelin')) fail('throw_rendering', fact);
      }
    }
    for (const fact of kb) if (!(fact in world.per_fact_context)) fail('kb_fact_without_evidence', fact);
    for (const qa of world.qa_pairs) {
      questions++;
      for (const fact of qa.facts_used) if (!kb.has(fact)) fail('question_fact_absent_from_kb', fact);
    }
  });
  return { split, worlds: worlds.length, table_facts: tableFacts, throw_facts: throwFacts,
    questions, raw_inversions: rawInversions.length, inversion_examples: rawInversions.slice(0, 4), failures };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const [directory, output] = process.argv.slice(2);
  if (!directory || !output) throw new Error('usage: audit-commaqa-evidence.mjs NUMERIC_DIR REPORT.json');
  const splits = ['train', 'dev', 'test'].map(split => auditEvidence(JSON.parse(readFileSync(`${directory}/${split}.json`, 'utf8')), split));
  const report = { splits, failures: splits.flatMap(s => s.failures),
    table_facts: splits.reduce((n, s) => n + s.table_facts, 0),
    raw_inversions: splits.reduce((n, s) => n + s.raw_inversions, 0) };
  writeFileSync(output, JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify({ table_facts: report.table_facts, raw_inversions: report.raw_inversions, failures: report.failures.length }));
  if (report.failures.length) throw new Error('evidence audit failed');
}
