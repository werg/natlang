/** Dataset records for directory reducers, partitioned before any case samples them. */
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { SOURCES, cachePath } from './acquire.mjs';

const CACHE = process.env.NATLANG_DATASETS ?? fileURLToPath(new URL('../../../vendor/datasets', import.meta.url));
export const LABELED_FIELDS = {
  sms_spam: { text: 'sms', label: 'label', labels: ['ham', 'spam'], question: 'Is this message spam or ham?' },
  sst2: { text: 'sentence', label: 'label', labels: ['negative', 'positive'], question: 'Is this film review negative or positive?' },
  ag_news: { text: 'text', label: 'label', labels: ['World', 'Sports', 'Business', 'Sci/Tech'], question: 'Is this article World, Sports, Business or Sci/Tech news?' },
  emotion: { text: 'text', label: 'label', labels: ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'], question: 'Which emotion is expressed: sadness, joy, love, anger, fear or surprise?' },
  banking77: { text: 'text', label: 'category', question: 'Which banking intent does this request express?' },
  clinc_oos: { text: 'text', label: 'intent', question: 'Which assistant intent does this request express?' },
};

const digest = value => createHash('sha256').update(value).digest('hex');
/** Content identity ensures duplicate source rows never cross train/eval. */
export function sourceRecordId(dataset, text, label) { return digest(`${dataset}\0${text}\0${label}`); }
export function sourceRecordSplit(id) { return Number.parseInt(id.slice(0, 8), 16) % 10 === 0 ? 'test' : 'train'; }

const loaded = new Map();
export function labeledRows(dataset, split = 'train') {
  if (!LABELED_FIELDS[dataset]) throw new Error(`unknown labeled dataset ${dataset}`);
  if (split !== 'train' && split !== 'test') throw new Error('dataset split must be train or test');
  if (!loaded.has(dataset)) {
    const source = SOURCES[dataset], [file] = source.files, fields = LABELED_FIELDS[dataset];
    let lines;
    try { lines = readFileSync(`${cachePath(CACHE, dataset, source.revision, file.path)}.jsonl`, 'utf8'); }
    catch { throw new Error(`${dataset} is not cached; run node scripts/inline-curriculum/acquire.mjs --source ${dataset}`); }
    const unique = new Map();
    for (const line of lines.split(/\r?\n/)) {
      if (!line) continue;
      const row = JSON.parse(line), text = String(row[fields.text] ?? '').replace(/\\/g, ' ').replace(/\s+/g, ' ').trim();
      const label = String(row[fields.label] ?? '');
      if (text.length < 15 || text.length > 2000 || !label || label === 'oos') continue;
      // Partition by model-visible text so duplicate inputs with conflicting labels cannot cross train/eval.
      const id = sourceRecordId(dataset, text, '');
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
      const id = sourceRecordId('coedit', original, '');
      unique.set(id, { id, task, text, target });
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
        rows.push({ id: sourceRecordId('hotpotqa', row.id, row.answer), question: String(row.question),
          answer: String(row.answer), supports, context });
      }
    }
    hotpotCache = rows.sort((a, b) => a.id.localeCompare(b.id));
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
      const relevant = qas.find(qa => String(qa.id ?? '').endsWith('__Non-Compete'));
      if (!title || !content || !relevant || content.length > 1_000_000) continue;
      const answers = (relevant.answers ?? []).map(answer => String(answer.text ?? '').trim())
        .filter(text => text && text.length <= 2000 && content.includes(text));
      contracts.push({ id: sourceRecordId('cuad', title, content), title, content, answer: answers[0] ?? '' });
    }
    cuadCache = contracts.sort((a, b) => a.id.localeCompare(b.id));
  }
  return cuadCache.filter(row => sourceRecordSplit(row.id) === split);
}
