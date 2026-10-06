// Knowledge desk: retrieval and multi-hop QA as NatLang programs (owner 2026-10-06: bring the bgkit/Schnitzeljagd
// retrieval and knowledge tasks in as proper programs; a compression prompt becomes the instructions of an inline
// lambda that reads one source for the consumer's need). HotpotQA supplies the question, its ten articles, the two
// supporting titles and their supporting sentences, so per-source judgments and notes are gold, not guessed.
//
// knowledge_evidence: a directory reducer. Every article in library/ goes through one nl lambda that decides whether
//   it states facts needed for the question and quotes them; supporting articles move to evidence/; a last lambda
//   answers from the collected facts.
// knowledge_research: iterateOn research. A library service searches titles and reads articles; each step an nl
//   lambda reads the latest article against the question and the notes so far and either names the next article to
//   look up (the bridge entity) or answers.
import { Folder } from '../../dist/index.js';
import { hotpotRows, sourceRecordId } from './folder-data.mjs';
import { SOURCES } from './acquire.mjs';
import { Random, curriculumCase, evalCall, literal, returnCall } from './lib.mjs';

const answerHint = answer =>
  /^(?:(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2},? \d{4}|\d{4}-\d{2}-\d{2})$/i.test(answer) ?
    ' Give the full calendar date.' : /^[+-]?\d+(?:\.\d+)?$/.test(answer) ? ' Give the number alone.' : '';

function spanOracle(question, support) {
  return { level: 'span', threshold: 0.8, normalization: 'qa',
    rubric: 'Answer the supplied question from the supplied supporting text. Accept equivalent phrasings and more precise answers the text supports; reject wrong entities, numbers or negations.',
    context: { question: question.question, supporting_text: support.map(item => item.body) } };
}

function articles(rng, question, pool, extra) {
  const others = rng.sample(pool.filter(row => row.id !== question.id), extra);
  const documents = new Map();
  for (const row of [question, ...others]) for (const item of row.context)
    if (!documents.has(item.title)) documents.set(item.title, item.text);
  const records = rng.shuffle([...documents].map(([title, body]) => ({
    id: sourceRecordId('hotpot-document', title, body).slice(0, 16), title, body })));
  return { records, others };
}

export function knowledgeEvidence(seed, index, split = 'train') {
  const rng = new Random(seed, `knowledge_evidence:${index}`), pool = hotpotRows(split);
  const question = rng.pick(pool);
  const { records, others } = articles(rng, question, pool, rng.int(1, 3));
  const folder = Folder.fromData(records.map(({ id, title, body }) => ({ id, body: `# ${title}\n\n${body}` })),
    { id: 'id', path: 'library/{id}.md', body: 'body', format: 'frontmatter' });
  const files = Object.fromEntries(folder.listFiles().map(file => [file.path, new TextDecoder().decode(folder.readBytesSync(file.path))]));
  const supportIds = new Set(question.supports.map(title => records.find(item => item.title === title).id));
  const facts = Object.fromEntries(question.supports.map((title, i) => [records.find(item => item.title === title).id, question.supportSentences[i].join(' ')]));
  const ask = question.question + answerHint(question.answer);
  const code = `const question = ${JSON.stringify(ask)};
const judge = nl<(file: File) => Promise<{ relevant: boolean, facts: string }>>\`Does this article state facts needed to answer question? If it does, quote exactly those facts; otherwise facts is empty.\`;
const notes: string[] = [];
for (const file of await folder.files('library/*.md')) {
  const found = await judge(file);
  if (found.relevant) { await file.moveTo('evidence/'); notes.push(found.facts); }
}
return await nl<string>\`Answer question from notes alone, as a short phrase.\`(notes);`;
  const expectedFiles = Object.fromEntries(Object.entries(files).map(([path, text]) => {
    const id = path.slice('library/'.length, -3);
    return [supportIds.has(id) ? `evidence/${id}.md` : path, text];
  }));
  const notes = [...supportIds].map(id => facts[id]);
  const children = records.map(item => ({ match: [item.id, 'Does this article state facts needed'],
    calls: [['read_file', { path: `${item.id}.md` }],
      returnCall(supportIds.has(item.id) ? { relevant: true, facts: facts[item.id] } : { relevant: false, facts: '' })] }));
  children.push({ match: ['Answer question from notes alone'], value: question.answer });
  const record = curriculumCase({ family: 'knowledge_evidence', shape: `hotpot-${index}`, variant: 'v0',
    splitGroup: `knowledge:${question.id}`, slice: 'inline_placement', domain: 'other', mode: 'single_call', inline: 'required',
    evidence: { world: [question.question], retrieved: question.supports, background: [] }, decisive: [],
    minimumSequence: ['judge every article in library/ for the question with an inline nl lambda',
      'move the articles that support the answer into evidence/', 'answer from the quoted facts'],
    reference: { root: [evalCall(code), returnCall(question.answer)], children },
    root: { name: 'research_question', kind: 'directory-reducer', args: {}, returns: 'string',
      instructions: `Question: ${ask}\nThe articles in library/ include the ones that answer it among many that do not. Judge each article with an inline nl lambda that reads it for this question and quotes the facts it contributes; move each supporting article into evidence/, keeping its filename. Then answer from the quoted facts alone. Return only the answer, as a short phrase.` },
    folderFiles: files, expectedFiles, inputs: {}, expected: question.answer, split });
  record.semantics.oracle = spanOracle(question, [...supportIds].map(id => records.find(item => item.id === id)));
  record.semantics.files_oracle = { compare: 'moves', threshold: 1 };
  record.curriculum.answer_evidence = question.supportSentences.flat();
  record.license = SOURCES.hotpotqa.license;
  record.gold_sources = ['hotpotqa-answer', 'hotpotqa-supporting-facts'];
  record.source_support = { titles: question.supports, sentences: question.supportSentences, notes };
  record.dataset = 'hotpotqa';
  record.dataset_records = [question.id, ...others.map(row => row.id)];
  record.generation.layout = { id: 'id', path: 'library/{id}.md', body: 'body', format: 'frontmatter' };
  return [record];
}

const words = text => new Set(text.toLowerCase().match(/[a-z0-9]+/g) ?? []);
/** The service's search, run here to check the reference's lookups (the TS in the service is the same procedure). */
function searchTitles(articles, query, limit = 5) {
  const q = words(query);
  return Object.keys(articles).map(title => {
    const t = words(title), b = words(articles[title]);
    let score = 0; for (const w of q) score += (t.has(w) ? 3 : 0) + (b.has(w) ? 1 : 0);
    return { title, score };
  }).filter(hit => hit.score > 0).sort((a, b) => b.score - a.score || (a.title < b.title ? -1 : a.title > b.title ? 1 : 0)).slice(0, limit).map(hit => hit.title);
}

export function knowledgeResearch(seed, index, split = 'train') {
  const rng = new Random(seed, `knowledge_research:${index}`), pool = hotpotRows(split);
  const question = rng.pick(pool);
  const { records, others } = articles(rng, question, pool, rng.int(2, 5));
  const ask = question.question + answerHint(question.answer);
  const [first, second] = question.supports;
  const byTitle = Object.fromEntries(records.map(item => [item.title, item.body]));
  // The service: exact-title lookup plus a word-overlap search over titles and leads, both deterministic.
  const service = `const ARTICLES: Record<string, string> = ${literal(Object.fromEntries(records.map(item => [item.title, item.body])))};
const words = (text: string) => new Set(text.toLowerCase().match(/[a-z0-9]+/g) ?? []);
/** Titles of the articles best matching query (by shared words in title and text), best first. */
export function search(query: string, limit: number = 5): string[] {
  const q = words(query);
  return Object.keys(ARTICLES).map(title => {
    const t = words(title), b = words(ARTICLES[title]);
    let score = 0; for (const w of q) score += (t.has(w) ? 3 : 0) + (b.has(w) ? 1 : 0);
    return { title, score };
  }).filter(hit => hit.score > 0).sort((a, b) => b.score - a.score || (a.title < b.title ? -1 : a.title > b.title ? 1 : 0)).slice(0, limit).map(hit => hit.title);
}
/** The text of the article with this exact title. */
export function read(title: string): string { return ARTICLES[title] ?? ''; }
`;
  const code = `const question = ${JSON.stringify(ask)};
type State = { notes: string[], next: string | null, answer: string | null, steps: number };
const step = nl<(article: string, notes: string[]) => Promise<{ facts: string, next: string | null, answer: string | null }>>\`Read article for question. Quote the facts in it that bear on question. If notes and these facts answer question, give answer; otherwise name the exact title of the article to look up next in next.\`;
const start = library.search(question, 1)[0];
const final = await iterateOn(async (s: State) => {
  const found = await step(library.read(s.next!), s.notes);
  const notes = [...s.notes, found.facts];
  const next = found.answer ? null : library.search(found.next ?? '', 1)[0] ?? null;
  return { notes, next, answer: found.answer, steps: s.steps + 1 };
}, { notes: [], next: start, answer: null, steps: 0 }).withLimit({ maxSteps: 6 }).until(s => s.answer !== null || s.next === null);
return final.answer ?? '';`;
  // The reference reaches the first supporting article from the question itself only when search ranks it first, and
  // the second through searching the title the first step names.
  const top = searchTitles(byTitle, ask, 1)[0];
  const ordered = top === second ? [second, first] : [first, second];
  if (top !== ordered[0] || searchTitles(byTitle, ordered[1], 1)[0] !== ordered[1]) return [];
  const sentences = Object.fromEntries(question.supports.map((title, i) => [title, question.supportSentences[i].join(' ')]));
  // A step is known by its article: a plain-text run of its opening. The later step also sees the earlier facts in its
  // notes, so it is listed first (the first matching child answers).
  const marker = title => /^[A-Za-z0-9 ,.'()-]*/.exec(byTitle[title].slice(0, 60))[0].trim();
  if (ordered.some(title => marker(title).length < 16) || marker(ordered[0]) === marker(ordered[1])) return [];
  const children = [
    { match: ['Read article for question', marker(ordered[1])],
      value: { facts: sentences[ordered[1]], next: null, answer: question.answer } },
    { match: ['Read article for question', marker(ordered[0])],
      value: { facts: sentences[ordered[0]], next: ordered[1], answer: null } },
    { match: 'An iterative process', value: { verdict: 'continue', reason: 'Each step reads an article not read before and adds its facts to the notes.' } },
  ];
  const record = curriculumCase({ family: 'knowledge_research', shape: `hotpot-${index}`, variant: 'v0',
    splitGroup: `knowledge:${question.id}`, slice: 'inline_placement', domain: 'other', mode: 'single_call',
    inline: 'required', iterate: 'required',
    evidence: { world: [question.question], retrieved: question.supports, background: [] }, decisive: [],
    minimumSequence: ['search the library for the question', 'iterate: read an article with an nl step that quotes facts and names the next article or answers',
      'stop when answered'],
    reference: { root: [evalCall(code), returnCall(question.answer)], children },
    root: { name: 'research', args: {}, returns: 'string',
      instructions: `Question: ${ask}\nResearch it in the library service (library.search finds article titles, library.read returns an article). Use iterateOn: each step reads one article with an inline nl lambda that quotes the facts bearing on the question and either answers or names the next article to look up. Return only the answer, as a short phrase.` },
    files: { 'research/library.ts': service }, inputs: {}, expected: question.answer, split });
  record.semantics.oracle = spanOracle(question, question.supports.map(title => ({ body: byTitle[title] })));
  record.curriculum.answer_evidence = question.supportSentences.flat();
  record.license = SOURCES.hotpotqa.license;
  record.gold_sources = ['hotpotqa-answer', 'hotpotqa-supporting-facts'];
  record.source_support = { titles: question.supports, sentences: question.supportSentences, order: ordered };
  record.dataset = 'hotpotqa';
  record.dataset_records = [question.id, ...others.map(row => row.id)];
  return [record];
}
