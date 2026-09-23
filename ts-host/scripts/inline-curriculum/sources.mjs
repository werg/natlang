// Adapters from pinned external sources to curriculum cases. Source answers and formal annotations stay
// in the oracle block; the model sees only the natural-language premises through a paged store.
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { Random, curriculumCase, evalCall, literal, returnCall } from './lib.mjs';
import { READ_ALL, factStore } from './logic.mjs';
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

// PrOntoQA-OOD proof-only examples: fictional ontologies ("Every lempus is a gorpus."), one entity, and a goal.
const PRONTO_FILES = ['2hop_ProofsOnly_random_noadj.json', '3hop_ProofsOnly_random_noadj.json', '4hop_ProofsOnly_random_noadj.json',
  '1hop_ProofsOnly_5testhops_random_noadj.json', '2hop_ProofsOnly_4shot_5testhops_random_noadj.json'];
function prontoRows() {
  if (loaded.has('prontoqa')) return loaded.get('prontoqa');
  const source = SOURCES.prontoqa;
  const dir = cachePath(CACHE, 'prontoqa', source.revision, 'generated_ood_data');
  const seen = new Set(), rows = [];
  for (const file of PRONTO_FILES) {
    let data;
    try { data = JSON.parse(readFileSync(`${dir}/${file}`, 'utf8')); }
    catch { throw new Error('PrOntoQA is not in the dataset cache; run node scripts/inline-curriculum/acquire.mjs --source prontoqa'); }
    for (const [key, entry] of Object.entries(data)) for (const [slot, example] of Object.entries(entry)) {
      if (seen.has(example.question)) continue;
      seen.add(example.question);
      rows.push({ id: `${file.replace(/\.json$/, '')}:${key}:${slot}`, file, split: slot === 'test_example' ? 'test' : 'train', ...example });
    }
  }
  loaded.set('prontoqa', rows);
  return rows;
}

/** One PrOntoQA sentence as a literal or a rule: `{ subject, predicate, negated, isClass, rule }`. */
function prontoSentence(text) {
  const singular = word => word.toLowerCase().replace(/uses$/, 'us');
  let m = /^(?:Every|Each) (\w+) is (not )?(an? )?(\w+)\.$/.exec(text);
  if (m) return { rule: true, subject: m[1], negated: !!m[2], isClass: !!m[3], predicate: m[4] };
  m = /^(\w+) are (not )?(\w+)\.$/.exec(text);
  if (m) return { rule: true, subject: singular(m[1]), negated: !!m[2], isClass: /uses$/.test(m[3]), predicate: singular(m[3]) };
  m = /^([A-Z]\w+) is (not )?(an? )?(\w+)\.$/.exec(text);
  if (m) return { rule: false, subject: m[1], negated: !!m[2], isClass: !!m[3], predicate: m[4] };
  throw new Error(`unparsed PrOntoQA sentence: ${text}`);
}

/** Every literal about entity derivable from the facts, with the fact ids of one chain to each. */
function prontoClosure(facts, entity) {
  const derived = new Map();
  const key = literal => `${literal.negated ? 'not ' : ''}${literal.isClass ? 'a ' : ''}${literal.predicate}`;
  const queue = [];
  for (const fact of facts) if (!fact.parsed.rule && fact.parsed.subject === entity) {
    derived.set(key(fact.parsed), [fact.id]);
    if (fact.parsed.isClass && !fact.parsed.negated) queue.push({ cls: fact.parsed.predicate, chain: [fact.id] });
  }
  for (let i = 0; i < queue.length; i++) {
    const { cls, chain } = queue[i];
    for (const fact of facts) if (fact.parsed.rule && fact.parsed.subject === cls) {
      const literal = key(fact.parsed);
      if (derived.has(literal)) continue;
      derived.set(literal, [...chain, fact.id]);
      if (fact.parsed.isClass && !fact.parsed.negated) queue.push({ cls: fact.parsed.predicate, chain: [...chain, fact.id] });
    }
  }
  return { derived, key };
}

/**
 * PrOntoQA: prove a goal about an entity by submitting a chain of fact ids to a verifier. The counterpart
 * removes one rule the proof needs (and every alternative), so the same goal is not provable from the store.
 */
export function prontoProof(seed, index, mode = 'proof') {
  const rows = prontoRows();
  const row = rows[index % rows.length];
  const rng = new Random(seed, `pronto:${row.id}`);
  const sentences = row.question.split(/(?<=\.) /).map(text => text.trim()).filter(Boolean);
  const goalText = row.query.replace(/^Prove: /, '');
  const goal = prontoSentence(goalText);
  const entity = goal.subject;
  const build = removed => {
    const kept = sentences.filter(text => text !== removed);
    return rng.shuffle(kept).map((text, i) => ({ id: `F${i + 1}`, text, parsed: prontoSentence(text) }));
  };
  const provable = build(null);
  const { derived, key } = prontoClosure(provable, entity);
  const chain = derived.get(key(goal));
  if (!chain) throw new Error(`PrOntoQA ${row.id}: the goal is not derivable from its own theory`);
  // Remove a rule of the proof such that nothing else derives the goal.
  const rules = chain.slice(1).map(id => provable.find(fact => fact.id === id).text);
  const removed = rng.shuffle(rules).find(text => !prontoClosure(build(text), entity).derived.has(key(goal)));
  if (!removed) throw new Error(`PrOntoQA ${row.id}: every rule of the proof has an alternative`);
  const certificate = `cert-${createHash('sha256').update(`${row.id}:${goalText}`).digest('hex').slice(0, 10)}`;
  const shape = row.id.replace(/[^A-Za-z0-9]+/g, '_');
  return [['provable', provable, { status: 'proved', certificate }], ['rule_removed', build(removed), { status: 'unprovable', certificate: null }]]
    .map(([variant, facts, expected]) => {
      const ids = facts.map(({ id, text }) => ({ id, text }));
      const structured = Object.fromEntries(facts.map(fact => [fact.id, fact.parsed]));
      const proofChain = variant === 'provable' ? prontoClosure(facts, entity).derived.get(key(goal)) : null;
      const verifier = `type Parsed = { rule: boolean, subject: string, negated: boolean, isClass: boolean, predicate: string };
const FACTS: Record<string, Parsed> = ${literal(structured)};
const GOAL: Parsed = ${literal(goal)};
const say = (p: Parsed) => (p.rule ? 'every ' : '') + p.subject + ' is ' + (p.negated ? 'not ' : '') + (p.isClass ? 'a ' : '') + p.predicate;
/**
 * Check a proof of the goal. chain lists fact ids in order: first a fact about the entity ("X is a C"), then
 * each rule applied to the class reached so far ("Every C is a D", "Every D is E"). Returns a certificate when
 * the chain ends in the goal, and otherwise the first problem.
 */
export function verify(chain: string[]): { ok: boolean, certificate: string | null, problem: string | null } {
  const fail = (problem: string) => ({ ok: false, certificate: null, problem });
  if (!chain.length) return fail('the chain is empty');
  const first = FACTS[chain[0]];
  if (!first) return fail('unknown fact id ' + chain[0]);
  if (first.rule || first.subject !== GOAL.subject) return fail(chain[0] + ' (' + say(first) + ') is not a fact about ' + GOAL.subject);
  let reached: Parsed = first;
  for (const id of chain.slice(1)) {
    const rule = FACTS[id];
    if (!rule) return fail('unknown fact id ' + id);
    if (!rule.rule) return fail(id + ' (' + say(rule) + ') is not a rule');
    if (!reached.isClass || reached.negated) return fail('the chain already ended with "' + say(reached) + '"; no rule applies after it');
    if (rule.subject !== reached.predicate) return fail(id + ' is about ' + rule.subject + ', but the chain so far shows ' + GOAL.subject + ' is a ' + reached.predicate);
    reached = { rule: false, subject: GOAL.subject, negated: rule.negated, isClass: rule.isClass, predicate: rule.predicate };
  }
  const ok = reached.predicate === GOAL.predicate && reached.negated === GOAL.negated && reached.isClass === GOAL.isClass;
  return ok ? { ok, certificate: ${JSON.stringify(certificate)}, problem: null } : fail('the chain shows "' + say(reached) + '", not the goal "' + say(GOAL) + '"');
}
`;
      const reference = mode === 'search' ? [evalCall(PRONTO_SEARCH), returnCall(expected)] : variant === 'provable' ?
        [evalCall(READ_ALL), evalCall(`const checked = proof.verify(${JSON.stringify(proofChain)});\nchecked`), returnCall(expected)] :
        [evalCall(READ_ALL), returnCall(expected)];
      const search = mode === 'search';
      return curriculumCase({ family: search ? 'prontoqa_search' : 'prontoqa_proof', shape, variant, pairGroup: `pronto${search ? '-search' : ''}:${shape}`,
        splitGroup: `prontoqa:${row.file}:${row.id.split(':')[1]}`, split: row.split,
        slice: search ? 'iterate' : 'observation_followup', domain: 'logic', mode: 'single_call', worldSemantics: 'open_world', inline: 'avoid',
        ...(search ? { iterate: 'required' } : {}),
        evidence: { world: variant === 'provable' ? proofChain.map(id => facts.find(f => f.id === id).text) : [`removed: ${removed}`],
          retrieved: [], background: [`source: PrOntoQA-OOD ${SOURCES.prontoqa.revision} ${row.id}`, `gold chain of thought: ${row.chain_of_thought.join(' ')}`] },
        minimumSequence: ['read the fact store', 'find a chain from a fact about the entity through rules to the goal', 'verify it, or conclude no chain exists'],
        reference: { root: reference },
        root: { name: 'prove_goal', args: { goal: 'string' }, returns: 'ProofResult',
          instructions: `Prove goal from the facts in the store (facts), using only those facts. A proof is a chain of fact ids: a fact about the entity, then each rule you apply in turn. Check it with proof.verify(chain); on success return status "proved" with the verifier's certificate. If the facts do not prove goal, return status "unprovable" with a null certificate.` +
            (search ? ' Find the chain by searching forward with iterateOn: each step applies the rules to the classes the entity has reached so far, until the goal is reached or a step reaches nothing new.' : '') },
        files: { 'prove_goal/facts.ts': factStore(ids, 'The fact store.'), 'prove_goal/proof.ts': verifier,
          'types.ts': 'export type ProofResult = { status: "proved" | "unprovable", certificate: string | null };\n' },
        inputs: { goal: goalText }, expected });
    });
}
export const prontoCount = () => prontoRows().length;
/** The same proofs found by forward search with iterateOn (logic in the iteration slice). */
export const prontoSearch = (seed, index) => prontoProof(seed, index, 'search');

const PRONTO_SEARCH = `const all: { id: string, text: string }[] = [];
const count = facts.pages();
for (let p = 1; p <= count; p++) all.push(...facts.page(p));
const entity = goal.split(' ')[0];
const singular = (word: string) => word.toLowerCase().replace(/uses$/, 'us');
type Rule = { id: string, from: string, to: string, isClass: boolean, negated: boolean };
const rules: Rule[] = [];
const reached: Record<string, string[]> = {};
for (const fact of all) {
  let m = /^(?:Every|Each) (\\w+) is (not )?(an? )?(\\w+)\\.$/.exec(fact.text);
  if (m) { rules.push({ id: fact.id, from: m[1], to: m[4], isClass: !!m[3], negated: !!m[2] }); continue; }
  m = /^(\\w+) are (not )?(\\w+)\\.$/.exec(fact.text);
  if (m) { rules.push({ id: fact.id, from: singular(m[1]), to: singular(m[3]), isClass: /uses$/.test(m[3]), negated: !!m[2] }); continue; }
  m = /^(\\w+) is an? (\\w+)\\.$/.exec(fact.text);
  if (m && m[1] === entity) reached[m[2]] = [fact.id];
}
const target = goal.replace(/\\.$/, '').replace(' is an ', ' is a ');
const say = (rule: Rule) => entity + ' is ' + (rule.negated ? 'not ' : '') + (rule.isClass ? 'a ' : '') + rule.to;
type Search = { reached: Record<string, string[]>, frontier: string[], found: string[] | null };
const expand = (state: Search): Search => {
  const next = { ...state.reached };
  const frontier: string[] = [];
  let found = state.found;
  for (const cls of state.frontier) for (const rule of rules.filter(r => r.from === cls)) {
    const chain = [...state.reached[cls], rule.id];
    if (found === null && say(rule) === target) found = chain;
    if (rule.isClass && !rule.negated && !next[rule.to]) { next[rule.to] = chain; frontier.push(rule.to); }
  }
  return { reached: next, frontier, found };
};
const direct = Object.keys(reached).find(cls => entity + ' is a ' + cls === target);
const final = await iterateOn(expand, { reached, frontier: Object.keys(reached), found: direct ? reached[direct] : null })
  .until(state => state.found !== null || state.frontier.length === 0);
if (final.found === null) return { status: 'unprovable', certificate: null };
const checked = proof.verify(final.found);
return { status: 'proved', certificate: checked.certificate };`;

/**
 * FOLIO, batched: every conclusion drawn from one story, each judged in its own inline child against the
 * premises the parent read (logic in the inline slice). Stories keep their original split.
 */
export function folioBatch(seed, index) {
  const rows = folioRows();
  const stories = [...new Set(rows.map(row => `${row.split}:${row.story_id}`))];
  const [split, story] = stories[index % stories.length].split(':');
  const examples = rows.filter(row => row.split === split && String(row.story_id) === story);
  const rng = new Random(seed, `folio-batch:${story}`);
  const premises = examples[0].premises.map(text => text.trim());
  const facts = rng.shuffle(premises).map((text, i) => ({ id: `P${i + 1}`, text }));
  const conclusions = examples.map((row, i) => ({ id: `C${i + 1}`, text: row.conclusion.trim(), verdict: VERDICT[row.label] }));
  if (conclusions.some(c => !c.verdict)) throw new Error(`FOLIO story ${story}: unexpected label`);
  const expected = Object.fromEntries(conclusions.map(c => [c.id, c.verdict]));
  const plain = conclusions.map(({ id, text }) => ({ id, text }));
  return [curriculumCase({ family: 'folio_batch', shape: `story${story}`, variant: split, splitGroup: `folio:story:${story}`,
    split: split === 'train' ? 'train' : 'test', slice: 'inline_placement', domain: 'logic', mode: 'single_call', inline: 'required',
    worldSemantics: 'open_world',
    evidence: { world: premises, retrieved: [], background: examples.map(row => `${row.conclusion.trim()} => ${row.label}`).concat(`source: FOLIO ${SOURCES.folio.revision}`) },
    assumptions: ['Only the stored premises hold; a statement they neither establish nor refute is unknown.'],
    minimumSequence: ['read the premises', 'judge each conclusion against them in its own child'],
    reference: { root: [evalCall(`${READ_ALL}
const premises = all.map(f => f.text);
const judged = await Promise.all(conclusions.map(conclusion => nl<Verdict>\`Using only premises and no outside knowledge, is conclusion entailed, contradicted, or unknown?\`(conclusion)));
return Object.fromEntries(conclusions.map((conclusion, i) => [conclusion.id, judged[i]]));`), returnCall(expected)],
      children: plain.map(c => ({ match: JSON.stringify(c.text), value: expected[c.id] })) },
    root: { name: 'evaluate_conclusions', args: { conclusions: '{ id: string, text: string }[]' }, returns: 'Record<string, Verdict>',
      instructions: `For each of conclusions, decide whether it follows from the premises in the store (facts), using only those premises and no outside knowledge: "entailed" if the premises make it true, "contradicted" if they make it false, and "unknown" if they settle neither. Judge each conclusion separately. Return a record from conclusion id to verdict.` },
    files: { 'evaluate_conclusions/facts.ts': factStore(facts, 'The premise store.'),
      'types.ts': 'export type Verdict = "entailed" | "contradicted" | "unknown";\n' },
    inputs: { conclusions: plain }, expected })];
}
