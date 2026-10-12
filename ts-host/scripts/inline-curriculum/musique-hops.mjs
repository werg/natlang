// MuSiQue hops: multi-hop QA as a program of named calls whose written values flow into later calls.
// The root answer_question breaks the question into MuSiQue's own decomposition and calls find_fact once per hop.
// find_fact reads the paragraphs and returns, word for word, the one sentence that answers its step (MuSiQue's
// supporting paragraph for the hop, the sentence holding the hop's gold answer); a step that refers to an earlier
// answer (#1, #2) receives the facts found so far as `known`. The root prints each fact and returns the final answer.
// Every fact is a model-written value that later calls read: the end-to-end Neuralese stage's producer -> consumer edges.
//
// Gold: the hop answers and supporting paragraphs of MuSiQue answerable v1.0; the fact sentence is selected
// mechanically: the support paragraph's only sentence containing the hop answer, or, when several do, the one that
// also names strictly the most of the step's terms (selection recorded per hop); the case is held otherwise, and when
// the fact is not a sentence (table rows) or too long to be one.
// Splits: MuSiQue train -> train, dev -> test. A train question that shares a single-hop seed with any dev question is
// held (the same rule as directory-sources' held_out_seed_ids); every case carries `musique:seed:<id>` source groups.
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Random, curriculumCase, evalCall, nlFile, returnCall } from './lib.mjs';
import { SOURCES, cachePath } from './acquire.mjs';
import { pendingSourceReview } from '../../dist/teacher/source-review.js';

const CACHE = process.env.NATLANG_DATASETS ?? fileURLToPath(new URL('../../../vendor/datasets', import.meta.url));
const SOURCE = SOURCES.musique;
const sha = text => createHash('sha256').update(text).digest('hex');

/** A pinned MuSiQue file: the curriculum cache (acquire.mjs --source musique), else the Hugging Face hub cache. */
function sourceFile(file) {
  const candidates = [cachePath(CACHE, 'musique', SOURCE.revision, file.path),
    join(process.env.HF_HOME ?? join(homedir(), '.cache/huggingface'), 'hub/datasets--dgslibisey--MuSiQue/snapshots', SOURCE.revision, file.path)];
  const path = candidates.find(existsSync);
  if (!path) throw new Error('MuSiQue is not cached; run node scripts/inline-curriculum/acquire.mjs --source musique');
  const text = readFileSync(path, 'utf8');
  if (sha(text) !== file.sha256) throw new Error(`MuSiQue ${file.path}: checksum does not match the pinned ${file.sha256}`);
  return text.split('\n').filter(Boolean).map(line => JSON.parse(line));
}

const ABBREVIATION = /(?:\b(?:Mr|Mrs|Ms|Dr|St|Jr|Sr|Inc|Co|Ltd|Corp|No|vs|Mt|Ft|Gen|Col|Lt|Sgt|Capt|Rev|Prof|Gov|Sen|Rep|Pres|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|etc|approx|ca|al|U\.S|U\.K|D\.C)|\b[A-Z])\.$/;
/** A paragraph's sentences, verbatim: split after . ! ? before a capital or quote, never after an abbreviation or initial. */
export function sentences(text) {
  const out = [];
  let start = 0;
  for (const match of text.matchAll(/[.!?]["')\]]*\s+(?=["'(\[]?[A-Z0-9])/g)) {
    const end = match.index + match[0].trimEnd().length;
    if (ABBREVIATION.test(text.slice(start, end).replace(/["')\]]+$/, ''))) continue;
    out.push(text.slice(start, end).trim());
    start = match.index + match[0].length;
  }
  if (start < text.length && text.slice(start).trim()) out.push(text.slice(start).trim());
  return out;
}
const fold = text => text.normalize('NFKC').toLowerCase();
const mentions = (sentence, answer) => {
  const s = fold(sentence), a = fold(answer).trim();
  if (!a) return false;
  for (let at = s.indexOf(a); at >= 0; at = s.indexOf(a, at + 1)) {
    const before = s[at - 1], after = s[at + a.length];
    if (!(before && /[\p{L}\p{N}]/u.test(before)) && !(after && /[\p{L}\p{N}]/u.test(after))) return true;
  }
  return false;
};

const STOP = new Set(['what', 'when', 'where', 'which', 'who', 'whom', 'whose', 'with', 'from', 'that', 'this', 'there', 'their',
  'does', 'were', 'have', 'been', 'into', 'about', 'many', 'much', 'than', 'they', 'year', 'name', 'part', 'also']);
/** Why a row cannot become a case, or its hops (question, answer, fact sentence). */
function hopsOf(row) {
  if (!row.answerable) return { hold: 'unanswerable' };
  const steps = row.question_decomposition ?? [];
  if (steps.length < 2) return { hold: 'single-hop' };
  if (new Set(steps.map(step => step.question)).size !== steps.length) return { hold: 'repeated-step-question' };
  if (fold(steps.at(-1).answer).trim() !== fold(row.answer).trim()) return { hold: 'last-hop-not-answer' };
  const hops = [];
  for (const step of steps) {
    const paragraph = row.paragraphs.find(p => p.idx === step.paragraph_support_idx);
    if (!paragraph?.is_supporting) return { hold: 'support-paragraph-missing' };
    let found = sentences(paragraph.paragraph_text).filter(sentence => mentions(sentence, step.answer)), selection = 'only-answer-sentence';
    if (!found.length) return { hold: 'answer-not-in-a-sentence' };
    if (found.length > 1) {
      // Several sentences state the answer: take the one that also names the most of the step's terms (its subject,
      // relation and the earlier answers it refers to), when exactly one does.
      const step_ = step.question.replace(/#(\d+)/g, (_, k) => hops[Number(k) - 1]?.answer ?? '');
      const terms = [...new Set(fold(step_).match(/[\p{L}\p{N}]{4,}/gu) ?? [])].filter(term => !STOP.has(term));
      const scored = found.map(sentence => ({ sentence, score: terms.filter(term => mentions(sentence, term)).length }))
        .sort((a, b) => b.score - a.score);
      if (!scored[0].score || scored[0].score === scored[1].score) return { hold: 'answer-in-several-sentences' };
      found = [scored[0].sentence];
      selection = 'answer-sentence-with-most-step-terms';
    }
    if (found[0].length < 24) return { hold: 'fact-too-short' };
    // Table and infobox rows are not sentences; a very long span is several sentences the splitter could not separate.
    if (!/[.!?]["')\]]*$/.test(found[0])) return { hold: 'fact-not-a-sentence' };
    if (found[0].length > 400) return { hold: 'fact-too-long' };
    if (!paragraph.paragraph_text.includes(found[0])) return { hold: 'fact-not-verbatim' };
    for (const ref of step.question.match(/#(\d+)/g) ?? []) if (Number(ref.slice(1)) > hops.length) return { hold: 'forward-reference' };
    hops.push({ id: step.id, question: step.question, answer: step.answer, fact: found[0], title: paragraph.title, selection });
  }
  if (new Set(hops.map(hop => hop.fact)).size !== hops.length) return { hold: 'repeated-fact' };
  return { hops };
}

let loadedRows;
/** Eligible rows per split, and the hold reasons of the rest (reported by the pilot). */
export function musiqueHopRows() {
  if (loadedRows) return loadedRows;
  const [train, dev] = SOURCE.files.map(sourceFile);
  const heldSeeds = new Set(dev.flatMap(row => (row.question_decomposition ?? []).map(step => String(step.id))));
  const holds = {}, eligible = { train: [], test: [] };
  const hold = reason => { holds[reason] = (holds[reason] ?? 0) + 1; };
  for (const [split, rows] of [['train', train], ['test', dev]]) for (const row of rows) {
    if (split === 'train' && row.question_decomposition?.some(step => heldSeeds.has(String(step.id)))) { hold('held_out_musique_seed'); continue; }
    // A question whose gold is under source review (training/source-reviews) stays out until the review closes.
    if (pendingSourceReview('musique', row.id)) { hold('source_review_pending'); continue; }
    const { hops, hold: reason } = hopsOf(row);
    if (reason) { hold(reason); continue; }
    eligible[split].push({ row, hops });
  }
  return loadedRows = { eligible, holds };
}

const orders = new Map();
/** The seed's fixed order of a split's eligible rows: case index i is row order[i], so shards by --start never overlap. */
function orderFor(seed, split) {
  const key = `${seed}:${split}`;
  if (!orders.has(key)) orders.set(key, new Random(seed, `musique_hops:${split}`).shuffle(musiqueHopRows().eligible[split].map((_, i) => i)));
  return orders.get(key);
}

const FIND_FACT = `Find the fact that answers question in paragraphs and return that sentence word for word, exactly as the paragraph writes it.
question is one step of a larger question. It is either a question or a lookup written as "subject >> relation" (for example "Paris >> country" asks which country Paris is in). It may refer to the answers of earlier steps as #1, #2, ...; known lists the facts found for those steps, in order, so #1 is the answer stated in known[0].
Return only the one sentence from paragraphs that states the answer.`;

const ROOT = `Answer question from paragraphs (each one starts with its title).
Break question into single-fact steps and solve them in order. For each step call find_fact with the step, the paragraphs, and the facts found so far as known; a step may refer to an earlier step's answer as #1, #2, ... Print each fact as find_fact returns it.
Then return only the final answer, as a short phrase.`;

export function musiqueHops(seed, index, split = 'train') {
  const { eligible } = musiqueHopRows();
  const order = orderFor(seed, split);
  if (index >= order.length) return [];
  const { row, hops } = eligible[split][order[index]];
  // A paragraph is one string headed by its title, so a one-sentence paragraph is never a whole quoted value that
  // a later argument listing could confuse with the fact a call passed on.
  const paragraphs = row.paragraphs.map(p => `# ${p.title}\n${p.paragraph_text}`);
  const lines = ['const known: string[] = [];'];
  for (const hop of hops)
    lines.push(`known.push(await find_fact(${JSON.stringify(hop.question)}, paragraphs, [...known]));`, 'console.log(known.at(-1));');
  const children = hops.map(hop => ({ match: ['You are inside this call: find_fact', JSON.stringify(hop.question)], value: hop.fact }));
  const rng = new Random(seed, `musique_hops:${row.id}`);
  const record = curriculumCase({ family: 'musique_hops', shape: `musique-${row.id}`, variant: 'v0',
    splitGroup: `musique:${row.id}`, slice: 'nested_scoped', domain: 'relational', mode: 'single_call', inline: 'avoid', named: 'required',
    evidence: { world: [row.question], retrieved: hops.map(hop => hop.title), background: [] }, decisive: [],
    minimumSequence: ['split the question into single-fact steps', 'call find_fact for each step with the facts found so far',
      'return the final answer from the last fact'],
    reference: { root: [evalCall(lines.join('\n')), returnCall(row.answer)], children },
    root: { name: 'answer_question', args: { question: 'string', paragraphs: 'string[]' }, returns: 'string', instructions: ROOT },
    files: {
      'answer_question/find_fact.nl': nlFile({ args: { question: 'string', paragraphs: 'string[]', known: 'string[]' }, returns: 'string',
        description: 'The sentence of paragraphs that answers one step of a question, word for word.', instructions: FIND_FACT }),
    },
    inputs: { question: row.question, paragraphs: rng.shuffle(paragraphs) }, expected: row.answer, split });
  record.semantics.oracle = { level: 'normalized', alternates: row.answer_aliases ?? [] };
  record.split = split;
  record.license = SOURCE.license;
  record.gold_sources = ['musique-answer', 'musique-decomposition', 'musique-supporting-paragraph-sentence'];
  record.dataset = 'musique';
  record.dataset_records = [row.id];
  record.source_groups = [`musique:${row.id}`, ...hops.map(hop => `musique:seed:${hop.id}`),
    ...row.paragraphs.filter(p => p.is_supporting).map(p => `document:${sha(JSON.stringify([p.title, p.paragraph_text]))}`)];
  record.source_revisions = [`musique:${SOURCE.revision}`];
  record.external_source = { source: 'musique', source_id: row.id, original_split: split === 'train' ? 'train' : 'dev',
    revision: SOURCE.revision, files: SOURCE.files.map(({ path, sha256 }) => ({ path, sha256 })), license: SOURCE.license,
    adaptation: 'musique-decomposition-as-named-calls; fact = the support paragraph\'s sentence holding the hop answer (the only one, or the one naming the most step terms)' };
  record.source_support = { hops: hops.map(({ id, question, answer, fact, title, selection }) => ({ id, question, answer, fact, title, selection })) };
  return [record];
}
