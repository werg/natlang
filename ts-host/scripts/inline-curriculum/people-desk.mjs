// Schnitzeljagd's synthetic people world as directory reducers (owner 2026-10-06: Schnitzeljagd retrieval tasks as
// NatLang programs). Records are short passages (bios, rosters, directories, company pages) about generated people;
// each record's provenance lists the facts it states ("<person>:<attribute>"), so whether a record answers a question is
// gold, not guessed. One nl lambda reads one record for one question; the program votes over the records that state an
// answer. Data only: the world generator and its banks stay in Schnitzeljagd.
//
// people_lookup: one question about one person.
// people_chain:  two hops through a mentor ("In which city was the mentor of X born?"): look up the mentor, then ask the
//   second question about the mentor by name.
import { readFileSync } from 'node:fs';
import { Folder } from '../../dist/index.js';
import { Random, curriculumCase, evalCall, returnCall } from './lib.mjs';

const CORPUS = process.env.NATLANG_SYNTH_PEOPLE ?? '/mnt/external/sdkb-archive/corpora/tasks-synth-people-r4-20260928';
const LICENSE = 'LicenseRef-generated (Schnitzeljagd schnitz.synth_world, seed 0)';
const PROMPT = 'Does record state the answer to question? If it does, give that short answer exactly as the record states it; otherwise answer is empty.';

let world;
function load() {
  if (world) return world;
  const jsonl = path => readFileSync(path, 'utf8').split('\n').filter(Boolean).map(line => JSON.parse(line));
  const records = jsonl(`${CORPUS}/sources.jsonl`).map(record => ({ id: record.record_id, text: record.text,
    facts: new Set(record.provenance?.facts ?? []) }));
  const byId = new Map(records.map(record => [record.id, record]));
  const episodes = [];
  for (const [file, split] of [['episodes-train.jsonl', 'train'], ['episodes-validation.jsonl', 'test']]) for (const episode of jsonl(`${CORPUS}/${file}`)) {
    const attribute = episode.episode_id.split('-').slice(2).join('-');
    const question = episode.query.split('Question: ').pop().trim();
    const supports = episode.required_ids.map(id => byId.get(id)).filter(Boolean);
    // The asked fact is the one every supporting record states.
    const fact = [...(supports[0]?.facts ?? [])].find(f => f.endsWith(`:${attribute}`) && supports.every(s => s.facts.has(f)));
    if (!fact || typeof episode.answer !== 'string') continue;
    const stated = supports.filter(s => s.text.includes(episode.answer));
    if (stated.length < 2) continue;
    episodes.push({ id: episode.episode_id, split, attribute, question, answer: episode.answer, fact, person: fact.split(':')[0],
      name: question.match(/(?:of|was|did|does) ([A-Z][a-z]+(?: [A-Z][a-z]+){1,3})/)?.[1], supports: stated });
  }
  world = { records, episodes: episodes.filter(e => e.name && e.question.includes(e.name)) };
  return world;
}

/** Records for the folder: the chosen supports plus distractors that name none of the people asked about. */
function library(rng, supports, names, size) {
  const { records } = load();
  const chosen = new Map(supports.map(record => [record.id, record]));
  const free = records.filter(record => !chosen.has(record.id) && !names.some(name => record.text.includes(name) ||
    record.text.includes(name.split(' ').pop())));
  for (const record of rng.sample(free, size - chosen.size)) chosen.set(record.id, record);
  return rng.shuffle([...chosen.values()]);
}

const fileId = record => record.id.slice(0, 12);
const states = (record, hop) => record.facts.has(hop.fact);

function caseFor(rng, index, split, hops, family) {
  const supports = hops.flatMap(hop => rng.sample(hop.supports, Math.min(hop.supports.length, rng.int(2, 3))));
  const names = [...new Set(hops.map(hop => hop.name))];
  const records = library(rng, [...new Map(supports.map(r => [r.id, r])).values()], names, rng.int(10, 14));
  // Every record must be gold for every hop: a record outside a hop's supports that still states the fact (a copy
  // the episode did not list) would make the vote disagree with the reference.
  for (const hop of hops) for (const record of records) if (states(record, hop) && !record.text.includes(hop.answer)) return null;
  const folder = Folder.fromData(records.map(record => ({ id: fileId(record), body: record.text })),
    { id: 'id', path: 'records/{id}.md', body: 'body', format: 'frontmatter' });
  const files = Object.fromEntries(folder.listFiles().map(file => [file.path, new TextDecoder().decode(folder.readBytesSync(file.path))]));
  const lookup = `const records = await folder.files('records/*.md');
const read = nl<(record: File, question: string) => Promise<{ states: boolean, answer: string }>>\`${PROMPT}\`;
async function lookup(question: string): Promise<string> {
  const votes = new Map<string, number>();
  for (const record of records) {
    const found = await read(record, question);
    if (found.states) votes.set(found.answer, (votes.get(found.answer) ?? 0) + 1);
  }
  return [...votes].sort((a, b) => b[1] - a[1])[0]?.[0] ?? '';
}
`;
  const final = hops.at(-1).answer;
  let code, ask;
  if (hops.length === 1) {
    ask = hops[0].question;
    code = `${lookup}return await lookup(${JSON.stringify(ask)});`;
  } else {
    const [first, second] = hops;
    ask = second.question.replace(second.name, `the mentor of ${first.name}`);
    const template = second.question.replace(second.name, '${mentor}');
    code = `${lookup}const mentor = await lookup(${JSON.stringify(first.question)});\nreturn await lookup(\`${template}\`);`;
  }
  const children = hops.flatMap(hop => records.map(record => ({ match: [fileId(record), hop.question.replace(/\?$/, '')],
    calls: [['read_file', { path: `${fileId(record)}.md` }],
      returnCall(states(record, hop) ? { states: true, answer: hop.answer } : { states: false, answer: '' })] })));
  const record = curriculumCase({ family, shape: `people${index}`, variant: 'v0', slice: 'inline_placement', domain: 'relational',
    splitGroup: `synth-people:${hops.map(hop => hop.person).join('>')}`, mode: 'single_call', inline: 'required',
    evidence: { world: hops.map(hop => `${hop.question} ${hop.answer}`), retrieved: [], background: [] }, decisive: [],
    minimumSequence: hops.length === 1 ? ['read every record for the question with an inline nl lambda', 'return the answer most records state'] :
      ['look up the mentor: read every record with an inline nl lambda and vote', 'ask the second question about the mentor by name the same way'],
    reference: { root: [evalCall(code), returnCall(final)], children },
    root: { name: 'answer_from_records', kind: 'directory-reducer', args: {}, returns: 'string',
      instructions: `Question: ${ask}\nThe files in records/ are short records about people and companies; several may state the same fact. Read each record for the question with an inline nl lambda that says whether the record states the answer and what it states${hops.length > 1 ? '; find the mentor first, then ask about the mentor by name' : ''}. Return the answer most records state, as a short phrase.` },
    folderFiles: files, expectedFiles: files, inputs: {}, expected: final, split });
  record.semantics.oracle = { level: 'span', threshold: 0.8, normalization: 'qa',
    rubric: 'Answer the question from the records. Accept the same entity, date or value in another phrasing.',
    context: { question: ask, supporting_text: supports.map(s => s.text) } };
  record.license = LICENSE;
  record.gold_sources = ['schnitzeljagd-synth-world-facts'];
  record.dataset = 'synth_people';
  record.dataset_records = hops.map(hop => hop.id);
  return record;
}

export function peopleLookup(seed, index, split = 'train') {
  const rng = new Random(seed, `people_lookup:${index}`);
  const pool = load().episodes.filter(e => e.split === split);
  for (let attempt = 0; attempt < 20; attempt++) {
    const record = caseFor(rng, index, split, [rng.pick(pool)], 'people_lookup');
    if (record) return [record];
  }
  return [];
}

export function peopleChain(seed, index, split = 'train') {
  const rng = new Random(seed, `people_chain:${index}`);
  const { episodes } = load();
  const pool = episodes.filter(e => e.split === split);
  const byName = new Map();
  for (const e of pool) if (e.attribute !== 'mentor' && e.attribute !== 'sibling') (byName.get(e.name) ?? byName.set(e.name, []).get(e.name)).push(e);
  const mentors = pool.filter(e => e.attribute === 'mentor' && byName.has(e.answer));
  for (let attempt = 0; attempt < 20; attempt++) {
    const first = rng.pick(mentors), second = rng.pick(byName.get(first.answer));
    const record = caseFor(rng, index, split, [first, second], 'people_chain');
    if (record) return [record];
  }
  return [];
}
