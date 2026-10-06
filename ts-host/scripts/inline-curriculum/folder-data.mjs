/** Dataset records for directory reducers, partitioned before any case samples them. */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { SOURCES, cachePath } from './acquire.mjs';
import { pendingSourceReview } from '../../dist/teacher/source-review.js';
import { normalizeSourceText, sourceRecordId, sourceRecordSplit } from './source-split.mjs';
export { sourceRecordId, sourceRecordSplit } from './source-split.mjs';

const CACHE = process.env.NATLANG_DATASETS ?? fileURLToPath(new URL('../../../vendor/datasets', import.meta.url));
export const LABELED_FIELDS = {
  sms_spam: { text: 'sms', label: 'label', labels: ['ham', 'spam'], question: 'Is this message spam or ham?' },
  sst2: { text: 'sentence', label: 'label', labels: ['negative', 'positive'], question: 'Is this film review negative or positive?' },
  ag_news: { text: 'text', label: 'label', labels: ['World', 'Sports', 'Business', 'Sci/Tech'], question: 'Is this article World, Sports, Business or Sci/Tech news?' },
  emotion: { text: 'text', label: 'label', labels: ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'], question: 'Which emotion is expressed: sadness, joy, love, anger, fear or surprise?' },
  banking77: { text: 'text', label: 'category', question: 'Which banking intent does this request express?' },
  clinc_oos: { text: 'text', label: 'intent', question: 'Which assistant intent does this request express?' },
};

const quality = new Map();
const quarantine = (dataset, id, reason) => { quality.set(`${dataset}:${id}:${reason}`, { dataset, id, reason }); };
export const datasetQualityReport = () => [...quality.values()];
const loaded = new Map();
export function labeledRows(dataset, split = 'train') {
  if (!LABELED_FIELDS[dataset]) throw new Error(`unknown labeled dataset ${dataset}`);
  if (split !== 'train' && split !== 'test') throw new Error('dataset split must be train or test');
  if (!loaded.has(dataset)) {
    const source = SOURCES[dataset], [file] = source.files, fields = LABELED_FIELDS[dataset];
    let lines;
    try { lines = readFileSync(`${cachePath(CACHE, dataset, source.revision, file.path)}.jsonl`, 'utf8'); }
    catch { throw new Error(`${dataset} is not cached; run node scripts/inline-curriculum/acquire.mjs --source ${dataset}`); }
    const unique = new Map(), conflicts = new Set();
    for (const line of lines.split(/\r?\n/)) {
      if (!line) continue;
      const row = JSON.parse(line), text = normalizeSourceText(row[fields.text] ?? '');
      const label = String(row[fields.label] ?? '');
      if (text.length < 15 || text.length > 2000 || !label || label === 'oos') continue;
      // Partition by model-visible text so duplicate inputs with conflicting labels cannot cross train/eval.
      const id = sourceRecordId(dataset, text, '');
      if (pendingSourceReview(dataset, id)) {
        quarantine(dataset, id, 'source_review_pending'); continue;
      }
      if (conflicts.has(id)) continue;
      if (unique.has(id) && unique.get(id).label !== label) {
        unique.delete(id); conflicts.add(id); quarantine(dataset, id, 'conflicting_labels'); continue;
      }
      unique.set(id, { id, text, label });
    }
    loaded.set(dataset, [...unique.values()].sort((a, b) => a.id.localeCompare(b.id)));
  }
  return loaded.get(dataset).filter(row => sourceRecordSplit(row.id) === split);
}

const coeditCache = new Map();
export function coeditRows(task, split = 'train') {
  if (!['gec', 'simplification', 'paraphrase', 'coherence', 'neutralize'].includes(task))
    throw new Error(`unsupported CoEdIT task ${task}`);
  if (!coeditCache.has(task)) {
    const source = SOURCES.coedit, file = source.files[0];
    let lines;
    try { lines = readFileSync(cachePath(CACHE, 'coedit', source.revision, file.path), 'utf8'); }
    catch { throw new Error('CoEdIT is not cached; run node scripts/inline-curriculum/acquire.mjs --source coedit'); }
    const unique = new Map();
    for (const line of lines.split(/\r?\n/)) {
      if (!line) continue;
      const row = JSON.parse(line);
      if (row.task !== task) continue;
      const original = String(row.src ?? ''), colon = original.indexOf(': ');
      const text = (colon < 0 ? original : original.slice(colon + 2)).trim();
      const target = String(row.tgt ?? '').trim();
      if (text.length < 20 || text.length > 1500 || !target || target.length > 1500 || text === target) continue;
      // Several edit targets may share a draft; keep that draft entirely in one pool.
      const id = sourceRecordId('coedit', text, '');
      const targets = [...new Set([...(unique.get(id)?.targets ?? []), target])].sort();
      unique.set(id, { id, task, text, target: targets[0], targets });
    }
    coeditCache.set(task, [...unique.values()].sort((a, b) => a.id.localeCompare(b.id)));
  }
  return coeditCache.get(task).filter(row => sourceRecordSplit(row.id) === split);
}

let hotpotCache;
export function hotpotRows(split = 'train') {
  if (!hotpotCache) {
    const source = SOURCES.hotpotqa;
    const rows = [];
    for (const file of source.files) {
      let lines;
      try { lines = readFileSync(`${cachePath(CACHE, 'hotpotqa', source.revision, file.path)}.jsonl`, 'utf8'); }
      catch { throw new Error('HotpotQA is not cached; run node scripts/inline-curriculum/acquire.mjs --source hotpotqa'); }
      for (const line of lines.split(/\r?\n/)) {
        if (!line) continue;
        const row = JSON.parse(line);
        const titles = row.context?.title, sentences = row.context?.sentences;
        const supports = [...new Set(row.supporting_facts?.title ?? [])];
        if (!row.id || !row.question || !row.answer || !Array.isArray(titles) || !Array.isArray(sentences) ||
            titles.length !== sentences.length || supports.length !== 2 || !supports.every(title => titles.includes(title))) continue;
        const context = titles.map((title, index) => ({ title: String(title), text: (sentences[index] ?? []).join(' ').trim() }))
          .filter(item => item.text);
        if (context.length < 4) continue;
        const supportSentences = supports.map(title => {
          const sourceIndex = titles.indexOf(title);
          return (row.supporting_facts?.title ?? []).flatMap((name, index) => name === title ?
            [sentences[sourceIndex]?.[row.supporting_facts?.sent_id?.[index]]].filter(Boolean) : []);
        });
        if (supportSentences.some(group => !group.length)) {
          quarantine('hotpotqa', row.id, 'missing_support_sentence'); continue;
        }
        rows.push({ id: sourceRecordId('hotpotqa', String(row.question).normalize('NFKC').toLowerCase().replace(/\s+/g, ' ').trim(), ''), question: String(row.question),
          answer: String(row.answer), type: String(row.type ?? ''), supports, supportSentences, context });
      }
    }
    const unique = new Map(), conflicts = new Set();
    for (const row of rows) {
      if (conflicts.has(row.id)) continue;
      if (unique.has(row.id) && unique.get(row.id).answer !== row.answer) {
        unique.delete(row.id); conflicts.add(row.id); quarantine('hotpotqa', row.id, 'conflicting_answers'); continue;
      }
      if (!unique.has(row.id)) unique.set(row.id, row);
    }
    hotpotCache = [...unique.values()].sort((a, b) => a.id.localeCompare(b.id));
  }
  return hotpotCache.filter(row => sourceRecordSplit(row.id) === split);
}

let cuadCache;
export function cuadContracts(split = 'train') {
  if (!cuadCache) {
    const source = SOURCES.cuad, file = source.files[0];
    let parsed;
    try { parsed = JSON.parse(readFileSync(cachePath(CACHE, 'cuad', source.revision, file.path), 'utf8')); }
    catch { throw new Error('CUAD is not cached; run node scripts/inline-curriculum/acquire.mjs --source cuad'); }
    const contracts = [];
    for (const item of parsed.data ?? []) {
      const title = String(item.title ?? ''), paragraphs = item.paragraphs ?? [];
      const content = [...new Set(paragraphs.map(part => String(part.context ?? '')).filter(Boolean))].join('\n\n');
      const qas = paragraphs.flatMap(part => part.qas ?? []);
      const relevant = qas.filter(qa => String(qa.id ?? '').endsWith('__Non-Compete'));
      if (!title || !content || !relevant.length || content.length > 1_000_000) continue;
      const id = sourceRecordId('cuad', title, content);
      const annotated = relevant.flatMap(qa => qa.answers ?? []);
      const answers = [...new Set(annotated.map(answer => String(answer.text ?? '').trim())
        .filter(text => text && text.length <= 2000 && content.includes(text)))];
      if (relevant.some(qa => qa.is_impossible === false) && !annotated.length) {
        quarantine('cuad', id, 'missing_positive_annotation'); continue;
      }
      if (annotated.length && !answers.length) { quarantine('cuad', id, 'unusable_positive_spans'); continue; }
      if (relevant.some(qa => qa.is_impossible === true) && annotated.length) {
        quarantine('cuad', id, 'conflicting_positive_and_impossible'); continue;
      }
      contracts.push({ id, title, content, answer: answers[0] ?? '', answers });
    }
    cuadCache = contracts.sort((a, b) => a.id.localeCompare(b.id));
  }
  return cuadCache.filter(row => sourceRecordSplit(row.id) === split);
}
