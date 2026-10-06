// Retrieval and long-context tasks from bgkit and Schnitzeljagd whose intermediate answers are unknown, so they are
// collected from the teacher rather than replayed (owner 2026-10-06). Each case has a minimal reference that verifies
// the case (one lambda over the sources, answered with the gold), but the teacher writes its own program and is graded
// on the outcome: per-item lambdas over sessions, pages, chunks or citations are what the instructions ask for.
// Records carry generation.collection = 'teacher'; static demonstration builds leave these families out.
//
// web_research:     a question over a recorded web corpus (bgkit web_search_r1 / web_sds tool slots); a search/open
//                   service over the recorded pages, iterateOn research.
// memory_answer:    dated conversation sessions as files; a question asked later (bgkit memory_qa_v2), including
//                   knowledge updates, temporal questions and abstention.
// story_choice:     a long story and a multiple-choice question (QuALITY via bgkit qa_quality_mcq).
// story_answer:     a story summary and a free-form question (NarrativeQA via bgkit qa_narrativeqa).
// citance_summary:  citation contexts from papers citing one paper; summarize what it does (Schnitzeljagd citances).
import { closeSync, openSync, readFileSync, readSync } from 'node:fs';
import { Folder } from '../../dist/index.js';
import { Random, curriculumCase, evalCall, literal, returnCall } from './lib.mjs';

const EXPORT = process.env.NATLANG_BGKIT_EXPORT ?? '/home/werg/data/bgkit-export';
const CITANCES = process.env.NATLANG_CITANCES ?? '/mnt/external/sdkb-archive/corpora/tasks-citance-recall-20260928';
const cache = new Map();
const jsonl = path => readFileSync(path, 'utf8').split('\n').filter(Boolean).map(line => JSON.parse(line));
/** A JSONL export too large for one string (memory sessions run to ~180 KB a row): an index of line offsets, rows
 * parsed on demand, so a build holds only the rows it draws. */
class Rows {
  constructor(path) {
    this.fd = openSync(path, 'r');
    this.offsets = [];
    const chunk = Buffer.allocUnsafe(1 << 24);
    let position = 0, start = 0, read;
    while ((read = readSync(this.fd, chunk, 0, chunk.length, position)) > 0) {
      for (let i = chunk.indexOf(10, 0); i !== -1 && i < read; i = chunk.indexOf(10, i + 1)) {
        if (position + i > start) this.offsets.push([start, position + i]);
        start = position + i + 1;
      }
      position += read;
    }
    if (position > start) this.offsets.push([start, position]);
  }
  get length() { return this.offsets.length; }
  at(index) {
    const [start, end] = this.offsets[index], buffer = Buffer.allocUnsafe(end - start);
    readSync(this.fd, buffer, 0, buffer.length, start);
    return JSON.parse(buffer.toString('utf8'));
  }
  close() { closeSync(this.fd); }
}
function rows(name) {
  if (!cache.has(name)) {
    try { cache.set(name, new Rows(`${EXPORT}/${name}.jsonl`)); }
    catch { throw new Error(`${name} is not exported; run .venv-neuralese/bin/python -m scripts.neuralese_data.export_bgkit_jsonl ${name}`); }
  }
  return cache.get(name);
}
// Small stores are read whole.
const store = name => { const all = rows(name); return Array.from({ length: all.length }, (_, i) => all.at(i)); };
const bgkitSplit = row => row.split === 'eval' ? 'test' : 'train';
const BGKIT_LICENSE = 'bgkit task stores (data-only reuse); upstream licenses in dataset_license';

function teacherCase(fields, extra) {
  const record = curriculumCase(fields);
  record.generation.collection = 'teacher';
  Object.assign(record, extra);
  return record;
}
const folderOf = (rows, layout) => {
  const folder = Folder.fromData(rows, layout);
  return Object.fromEntries(folder.listFiles().map(file => [file.path, new TextDecoder().decode(folder.readBytesSync(file.path))]));
};

// ---- web_research ---------------------------------------------------------------------------------------------------

/** Pages from recorded tool slots: blocks "[id] Title\ntext"; chunks of one page are joined in order of appearance. */
function pagesOf(slots) {
  const pages = new Map();
  for (const slot of slots) for (const block of slot.split(/\n\s*\n(?=\[[^\]\n]+\] )/)) {
    const match = /^\[([^\]\n]+)\] ([^\n]*)\n?([\s\S]*)$/.exec(block.trim());
    if (!match) continue;
    const [, id, title, text] = match;
    const page = pages.get(id) ?? pages.set(id, { id, title, text: '' }).get(id);
    if (!page.text.includes(text.trim())) page.text = (page.text + '\n' + text.trim()).trim();
  }
  return [...pages.values()].map(page => ({ ...page, text: page.text.slice(0, 6000) }));
}

let webPool;
function webRows(split) {
  webPool ??= ['web_search_r1_v3', 'web_sds_v3'].flatMap(store).filter(row => row.meta?.final && Array.isArray(row.meta.gold) &&
    row.meta.gold.length && typeof row.meta.question === 'string').map(row => {
    let slots, queries;
    try { slots = JSON.parse(row.context); queries = JSON.parse(row.prompt); } catch { return null; }
    const pages = pagesOf(slots);
    // Recorded queries: "web search <query>: <question>" (other recorded tools are not searches).
    const searches = queries.map(q => /^web search (.*?): /.exec(q)?.[1]).filter(Boolean);
    return pages.length >= 2 && searches.length ? { row, pages, searches } : null;
  }).filter(Boolean);
  return webPool.filter(item => bgkitSplit(item.row) === split);
}

const WEB_SERVICE = pages => `const PAGES: { id: string, title: string, text: string }[] = ${literal(pages)};
const words = (text: string) => new Set(text.toLowerCase().match(/[a-z0-9]+/g) ?? []);
/** Pages best matching query (shared words, title words count triple), best first: id, title and a snippet. */
export function search(query: string, limit: number = 5): { id: string, title: string, snippet: string }[] {
  const q = words(query);
  return PAGES.map(page => {
    const t = words(page.title), b = words(page.text);
    let score = 0; for (const w of q) score += (t.has(w) ? 3 : 0) + (b.has(w) ? 1 : 0);
    return { page, score };
  }).filter(hit => hit.score > 0).sort((a, b) => b.score - a.score).slice(0, limit)
    .map(hit => ({ id: hit.page.id, title: hit.page.title, snippet: hit.page.text.slice(0, 240) }));
}
/** The page with this id: its title and text ('' when there is no such page). */
export function open(id: string): string {
  const page = PAGES.find(p => p.id === id);
  return page ? page.title + '\\n' + page.text : '';
}
`;

export function webResearch(seed, index, split = 'train') {
  const rng = new Random(seed, `web_research:${index}`);
  const { row, pages, searches } = rng.pick(webRows(split));
  const question = row.meta.question.trim(), gold = row.meta.gold.map(String);
  const code = `const QUERIES = ${literal(searches)};
type Research = { next: number, pages: string[] };
const final = await iterateOn((s: Research): Research => {
  const pages = [...s.pages];
  for (const hit of web.search(QUERIES[s.next])) if (!pages.includes(hit.id)) pages.push(hit.id);
  return { next: s.next + 1, pages };
}, { next: 0, pages: [] }).withLimit({ maxSteps: 16 }).until(s => s.next >= QUERIES.length);
return await nl<string>\`Answer question from these pages, as a short phrase.\`(final.pages.map(id => web.open(id)));`;
  return [teacherCase({ family: 'web_research', shape: `web${index}`, variant: 'v0', slice: 'iterate', domain: 'other',
    splitGroup: `bgkit-web:${question}`, mode: 'single_call', inline: 'required', iterate: 'required',
    evidence: { world: [question], retrieved: [], background: [] }, decisive: [],
    minimumSequence: ['search the recorded web corpus', 'iterate: read pages with an nl step that notes facts and picks the next search or answers',
      'answer as a short phrase'],
    reference: { root: [evalCall(code), returnCall(gold[0])],
      children: [{ match: 'Answer question from these pages', value: gold[0] },
        { match: 'An iterative process', value: { verdict: 'continue', reason: 'Each step runs a search not run before.' } }] },
    root: { name: 'research', args: { question: 'string' }, returns: 'string',
      instructions: 'Answer question from the web corpus: web.search(query) returns matching pages (id, title, snippet) and web.open(id) returns a page. You do not know the answer in advance; a question may need several pages, each found by searching with names learned from earlier pages. Use iterateOn: each step reads a page or result with an inline nl lambda that notes the facts bearing on question and either answers or says what to search next. Return only the answer, as a short phrase.' },
    files: { 'research/web.ts': WEB_SERVICE(pages) }, inputs: { question }, expected: gold[0], split }, {
    license: BGKIT_LICENSE, dataset_license: row.meta.license ?? null, gold_sources: [`bgkit-${row.store}-gold`], dataset: 'bgkit',
    dataset_records: [`${row.store}:${row.index}`] }) ].map(record => {
    record.semantics.oracle = { level: 'span', threshold: 0.8, normalization: 'qa', alternates: gold.slice(1),
      rubric: 'Answer the research question. Accept the same entity, date or number in another phrasing; reject other entities.',
      context: { question } };
    return record;
  });
}

// ---- memory_answer --------------------------------------------------------------------------------------------------

export function memoryAnswer(seed, index, split = 'train') {
  const rng = new Random(seed, `memory_answer:${index}`);
  const all = rows('memory_qa_v2_pairs');
  for (let attempt = 0; attempt < 40; attempt++) {
    const row = all.at(rng.int(0, all.length - 1));
    if (bgkitSplit(row) !== split) continue;
    let sessions, asked;
    try { sessions = JSON.parse(row.context); asked = JSON.parse(row.prompt)[0]; } catch { continue; }
    if (!sessions.length || sessions.length > 80 || typeof row.target !== 'string') continue;
    const total = sessions.reduce((n, s) => n + s.length, 0);
    if (total > 240000) continue;
    const rows = sessions.map((text, i) => {
      const date = /\[session on (\d{4})\/(\d{2})\/(\d{2})/.exec(text);
      return { id: `${String(i + 1).padStart(3, '0')}${date ? `-${date[1]}-${date[2]}-${date[3]}` : ''}`, body: text };
    });
    const files = folderOf(rows, { id: 'id', path: 'memory/{id}.md', body: 'body', format: 'frontmatter' });
    const question = asked.trim();
    const code = `return await nl<string>\`Answer question from the memory sessions alone; if they do not contain the answer, say it was not mentioned.\`(await folder.files('memory/*.md'));`;
    const record = teacherCase({ family: 'memory_answer', shape: `memory${index}`, variant: 'v0', slice: 'inline_placement', domain: 'other',
      splitGroup: `bgkit-memory:${row.meta.episode ?? row.index}`, mode: 'single_call', inline: 'required',
      evidence: { world: [question], retrieved: [], background: [] }, decisive: [],
      minimumSequence: ['read the sessions for the question with an inline nl lambda per session', 'combine the notes, latest facts first', 'answer or say it was not mentioned'],
      reference: { root: [evalCall(code), returnCall(row.target)], children: [{ match: 'Answer question from the memory sessions', value: row.target }] },
      root: { name: 'recall', kind: 'directory-reducer', args: { question: 'string' }, returns: 'string',
        instructions: 'memory/ holds dated sessions of past conversations, one file per session, in order. Answer question (it says when it is asked) from them. Read each session for the question with an inline nl lambda that notes what it says about it; when facts change over time the latest one counts. If the sessions never mention the answer, say it was not mentioned in our conversations. Return a short answer.' },
      folderFiles: files, expectedFiles: files, inputs: { question }, expected: row.target, split }, {
      license: 'LicenseRef-mixed (bgkit memory_qa_v2: dialogue corpora plus teacher-written sessions)', gold_sources: ['bgkit-memory-qa-target'],
      dataset: 'bgkit', dataset_records: [`${row.store}:${row.index}`], question_type: row.meta.qtype ?? null });
    record.semantics.oracle = { level: 'span', threshold: 0.8, normalization: 'qa',
      rubric: 'Answer the question from the dated sessions. The latest stated fact counts. When the reference says it was not mentioned, accept only an answer that says so.',
      context: { question } };
    return [record];
  }
  return [];
}

// ---- story_choice / story_answer ------------------------------------------------------------------------------------

export function storyChoice(seed, index, split = 'train') {
  const rng = new Random(seed, `story_choice:${index}`);
  const pool = store('qa_quality_mcq').filter(row => bgkitSplit(row) === split && Array.isArray(row.meta?.options) && /^[A-D]$/.test(row.target));
  const row = rng.pick(pool);
  const question = row.instruction.split('\nOptions:')[0].trim();
  const options = row.meta.options.map(String);
  const files = { 'story.md': row.context };
  const code = `return await nl<'A' | 'B' | 'C' | 'D'>\`Read the story and choose the option that answers question; give its letter.\`(folder.file('story.md'), question, options);`;
  const record = teacherCase({ family: 'story_choice', shape: `quality${index}`, variant: 'v0', slice: 'inline_placement', domain: 'other',
    splitGroup: `quality:${row.meta.doc ?? row.meta.title}`, mode: 'single_call', inline: 'optional',
    evidence: { world: [question], retrieved: [], background: [] }, decisive: [],
    minimumSequence: ['read the story in parts with inline nl lambdas, noting what bears on the question and options', 'choose the option'],
    reference: { root: [evalCall(code), returnCall(row.target)], children: [{ match: 'choose the option that answers question', value: row.target }] },
    root: { name: 'choose', kind: 'directory-reducer', args: { question: 'string', options: 'string[]' }, returns: '"A" | "B" | "C" | "D"',
      instructions: 'story.md is a long story. Answer question by choosing among options (A, B, C, D in order). The story is too long to take in at once: read it in parts with inline nl lambdas that note what each part says about the question and the options, then decide. Return the letter.' },
    folderFiles: files, expectedFiles: files, inputs: { question, options }, expected: row.target, split }, {
    license: 'CC-BY-4.0 (QuALITY via bgkit qa_quality_mcq)', gold_sources: ['quality-gold-label'], dataset: 'quality',
    dataset_records: [`${row.store}:${row.index}`] });
  record.semantics.oracle = 'exact';
  return [record];
}

export function storyAnswer(seed, index, split = 'train') {
  const rng = new Random(seed, `story_answer:${index}`);
  const pool = store('qa_narrativeqa').filter(row => bgkitSplit(row) === split && row.context && Array.isArray(row.meta?.answers));
  const row = rng.pick(pool);
  const question = row.instruction.trim(), answers = row.meta.answers.map(String);
  const files = { 'story.md': `# ${row.meta.title ?? 'Story'}\n\n${row.context}` };
  const code = `return await nl<string>\`Answer question from the story, briefly.\`(folder.file('story.md'), question);`;
  const record = teacherCase({ family: 'story_answer', shape: `narrative${index}`, variant: 'v0', slice: 'inline_placement', domain: 'other',
    splitGroup: `narrativeqa:${row.meta.doc}`, mode: 'single_call', inline: 'optional',
    evidence: { world: [question], retrieved: [], background: [] }, decisive: [],
    minimumSequence: ['read the story for the question with an inline nl lambda', 'answer briefly'],
    reference: { root: [evalCall(code), returnCall(answers[0])], children: [{ match: 'Answer question from the story', value: answers[0] }] },
    root: { name: 'answer', kind: 'directory-reducer', args: { question: 'string' }, returns: 'string',
      instructions: 'story.md tells a story. Answer question from it with an inline nl lambda that reads the story; return a brief answer (a phrase or one sentence).' },
    folderFiles: files, expectedFiles: files, inputs: { question }, expected: answers[0], split }, {
    license: 'Apache-2.0 (NarrativeQA via bgkit qa_narrativeqa); stories under their own terms', gold_sources: ['narrativeqa-reference-answers'],
    dataset: 'narrativeqa', dataset_records: [`${row.store}:${row.index}`] });
  record.semantics.oracle = { level: 'span', threshold: 0.6, normalization: 'qa', alternates: answers.slice(1),
    rubric: 'Answer the question about the story. Accept an answer that names the same person, thing or event as a reference answer, in any wording.',
    context: { question, reference_answers: answers } };
  return [record];
}

// ---- citance_summary ------------------------------------------------------------------------------------------------

let citancePool;
function citanceRows(split) {
  citancePool ??= [['episodes-train.jsonl', 'train'], ['episodes-validation.jsonl', 'test']].flatMap(([file, s]) =>
    jsonl(`${CITANCES}/${file}`).map(episode => ({ ...episode, split: s })))
    .filter(episode => typeof episode.answer === 'string' && episode.supports?.length >= 3);
  return citancePool.filter(episode => episode.split === split);
}

export function citanceSummary(seed, index, split = 'train') {
  const rng = new Random(seed, `citance_summary:${index}`);
  const episode = rng.pick(citanceRows(split));
  const request = episode.query.replace(/^Use the stored citation contexts\.\s*/, '').trim();
  const supports = rng.shuffle(episode.supports).slice(0, 24);
  const files = folderOf(supports.map(record => ({ id: record.record_id.slice(0, 12), body: record.text })),
    { id: 'id', path: 'citations/{id}.md', body: 'body', format: 'frontmatter' });
  const code = `return await nl<string>\`Do what request asks from the citation contexts.\`(await folder.files('citations/*.md'), request);`;
  const record = teacherCase({ family: 'citance_summary', shape: `citance${index}`, variant: 'v0', slice: 'inline_placement', domain: 'writing',
    splitGroup: `citance:${episode.episode_id.split('-').slice(0, -1).join('-') || episode.episode_id}`, mode: 'single_call', inline: 'required',
    evidence: { world: [request], retrieved: [], background: [] }, decisive: [],
    minimumSequence: ['read each citation context with an inline nl lambda, noting what it says the cited paper does', 'write the summary from the notes'],
    reference: { root: [evalCall(code), returnCall(episode.answer)], children: [{ match: 'Do what request asks from the citation contexts', value: episode.answer }] },
    root: { name: 'summarize_cited', kind: 'directory-reducer', args: { request: 'string' }, returns: 'string',
      instructions: 'citations/ holds passages from papers that cite one paper. Do what request asks from them: read each passage with an inline nl lambda that notes what it says the cited paper does or finds, then write the answer from those notes alone, without inventing details. Return the text.' },
    folderFiles: files, expectedFiles: files, inputs: { request }, expected: episode.answer, split }, {
    license: 'CC-BY-SA-4.0 (unarXive citrec via Schnitzeljagd citances); arXiv metadata CC0', gold_sources: ['cited-paper-abstract'],
    dataset: 'citance_recall', dataset_records: [episode.episode_id] });
  record.semantics.oracle = { level: 'span', threshold: 0.5, normalization: 'qa',
    rubric: 'Judge whether the answer states what the reference (the cited paper\'s abstract or description) states about the paper: its problem, method and main result. Missing secondary details are fine; claims the reference contradicts or does not support are not.',
    context: { request } };
  return [record];
}
