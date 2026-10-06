// Labeled judgments: items from labeled text datasets (spam, sentiment, news topics, emotions, banking and assistant
// intents), each judged with an inline nl, composed with code that filters, counts, groups or picks. The datasets'
// labels are the children's answers and give the expected result; the code around the judgments is exact.
//
// The items are sampled from the datasets' training splits per seed, so like a generated family every seed makes new
// cases. A case needs its dataset in the cache (acquire.mjs --source NAME).
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { SOURCES, cachePath } from './acquire.mjs';
import { normalizeSourceText, sourceRecordId, sourceRecordSplit } from './source-split.mjs';
import { Random, curriculumCase, evalCall, returnCall } from './lib.mjs';

const CACHE = process.env.NATLANG_DATASETS ?? fileURLToPath(new URL('../../../vendor/datasets', import.meta.url));

/**
 * Per dataset: where its text and label are, what its items are called, and its labels, each with the words that
 * describe it. `question` asks one label of an item named `item`; `about` states it of items ("are spam").
 */
export const DATASETS = {
  sms_spam: { text: 'sms', label: 'label', type: 'Message', plural: 'messages', singular: 'message', idPrefix: 'M',
    labels: { spam: { question: 'Is message spam: an unsolicited advertisement, a prize offer or a scam?', about: 'are spam' } },
    only: ['spam'] },
  sst2: { text: 'sentence', label: 'label', type: 'Review', plural: 'reviews', singular: 'review', idPrefix: 'R', minLength: 60,
    union: 'Sentiment', classify: 'Is review positive or negative about the film?',
    labels: { positive: { question: 'Is review positive about the film?', about: 'are positive about the film' },
      negative: { question: 'Is review negative about the film?', about: 'are negative about the film' } } },
  ag_news: { text: 'text', label: 'label', type: 'Article', plural: 'articles', singular: 'article', idPrefix: 'A',
    union: 'Topic', classify: 'Which topic is article about: World, Sports, Business or Sci/Tech (science and technology)?',
    labels: { World: { question: 'Is article world news?', about: 'are world news' },
      Sports: { question: 'Is article about sports?', about: 'are about sports' },
      Business: { question: 'Is article about business?', about: 'are about business' },
      'Sci/Tech': { question: 'Is article about science or technology?', about: 'are about science or technology' } } },
  emotion: { text: 'text', label: 'label', type: 'Post', plural: 'posts', singular: 'post', idPrefix: 'P', minLength: 40,
    union: 'Emotion', classify: 'Which emotion does post mainly express: sadness, joy, love, anger, fear or surprise?',
    labels: Object.fromEntries(['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'].map(label =>
      [label, { question: `Does post mainly express ${label}?`, about: `mainly express ${label}` }])) },
  // Intent sets have many labels: a case uses a few of them, named in its instructions.
  banking77: { text: 'text', label: 'category', type: 'Query', plural: 'requests', singular: 'request', idPrefix: 'Q',
    union: 'Intent', subset: [3, 4], describe: label => label.toLowerCase().replace(/_/g, ' ') },
  clinc_oos: { text: 'text', label: 'intent', type: 'Query', plural: 'requests', singular: 'request', idPrefix: 'Q',
    union: 'Intent', subset: [3, 5], exclude: ['oos'], describe: label => label.replace(/_/g, ' ') },
};

const loaded = new Map();
export function rowsOf(name, split = 'train') {
  if (!loaded.has(name)) {
    const source = SOURCES[name], [file] = source.files, spec = DATASETS[name];
    let text;
    try { text = readFileSync(`${cachePath(CACHE, name, source.revision, file.path)}.jsonl`, 'utf8'); }
    catch { throw new Error(`${source.name} is not in the dataset cache; run node scripts/inline-curriculum/acquire.mjs --source ${name}`); }
    const rows = text.split('\n').filter(Boolean).map(line => JSON.parse(line))
      .map(row => ({ text: normalizeSourceText(row[spec.text]), label: String(row[spec.label]) }))
      .filter(row => row.text.length >= (spec.minLength ?? 15) && row.text.length <= 320 && !(spec.exclude ?? []).includes(row.label));
    loaded.set(name, rows);
  }
  return loaded.get(name).filter(row => sourceRecordSplit(sourceRecordId(name, row.text, '')) === split);
}

/** The labels a case uses, each with its question and statement. */
function labelsFor(rng, name, split) {
  const spec = DATASETS[name];
  if (spec.labels) return spec.labels;
  const all = [...new Set(rowsOf(name, split).map(row => row.label))].sort();
  return Object.fromEntries(rng.sample(all, rng.int(...spec.subset)).map(label => [label, {
    question: `Is request about ${spec.describe(label)}?`, about: `are about ${spec.describe(label)}` }]));
}

/** Items of the given labels, sampled with at least one of the target label and one other when asked. */
function itemsFor(rng, name, labels, count, target, split) {
  const pool = rowsOf(name, split).filter(row => labels.includes(row.label));
  for (;;) {
    const picked = rng.sample(pool, count);
    const hits = picked.filter(row => row.label === target).length;
    if (!target || (hits > 0 && hits < picked.length)) return picked;
  }
}

const quoteUnion = labels => labels.map(label => JSON.stringify(label)).join(' | ');

/** The tasks: instructions, reference code over the items, and the expected result from the labels. */
const TASKS = [
  // The ids of the items with the label.
  (d, target, items, v) => ({ returns: 'string[]', expected: items.filter(i => i.label === target).map(i => i.id),
    instructions: `Return the ids of the ${d.plural} in ${d.plural} that ${v.about}, in their order.`,
    code: `const verdicts = await Promise.all(${d.plural}.map(${d.singular} => nl<boolean>\`${v.question.replace(/\bitem\b/g, d.singular)}\`(${d.singular})));\nreturn ${d.plural}.filter((_, i) => verdicts[i]).map(${d.singular} => ${d.singular}.id);` }),
  // How many have the label.
  (d, target, items, v) => ({ returns: 'number', expected: items.filter(i => i.label === target).length,
    instructions: `How many of the ${d.plural} in ${d.plural} ${v.about}?`,
    code: `const verdicts = await Promise.all(${d.plural}.map(${d.singular} => nl<boolean>\`${v.question}\`(${d.singular})));\nreturn verdicts.filter(Boolean).length;` }),
  // The first with the label, judging in order and stopping there.
  (d, target, items, v) => ({ returns: 'string | null', expected: items.find(i => i.label === target)?.id ?? null,
    instructions: `Go through ${d.plural} in order and return the id of the first ${d.singular} that ${v.about.replace(/^are /, 'is ').replace(/^mainly express/, 'mainly expresses')}, or null if none does. Stop judging once you find it.`,
    code: `const matches = nl<(${d.singular}: ${d.type}) => Promise<boolean>>\`${v.question}\`;\nfor (const ${d.singular} of ${d.plural}) if (await matches(${d.singular})) return ${d.singular}.id;\nreturn null;` }),
  // Exact work first: only the recent items are judged.
  (d, target, items, v, rng) => { const day = rng.int(4, 12);
    return { returns: 'string[]', expected: items.filter(i => i.day >= day && i.label === target).map(i => i.id),
      instructions: `Of the ${d.plural} in ${d.plural} from day ${day} on, return the ids of those that ${v.about}, in their order.`,
      code: `const recent = ${d.plural}.filter(${d.singular} => ${d.singular}.day >= ${day});\nconst verdicts = await Promise.all(recent.map(${d.singular} => nl<boolean>\`${v.question}\`(${d.singular})));\nreturn recent.filter((_, i) => verdicts[i]).map(${d.singular} => ${d.singular}.id);` }; },
];

/** Tasks over every label at once, for datasets with a label union. */
const CLASS_TASKS = [
  (d, labels, items, question) => ({ returns: `Record<string, ${d.union}>`,
    expected: Object.fromEntries(items.map(i => [i.id, i.label])),
    instructions: `Label each of the ${d.plural} in ${d.plural} as one of ${labels.map(l => JSON.stringify(l)).join(', ')}. Return a record from ${d.singular} id to its label.`,
    code: `const labels = await Promise.all(${d.plural}.map(${d.singular} => nl<${d.union}>\`${question}\`(${d.singular})));\nreturn Object.fromEntries(${d.plural}.map((${d.singular}, i) => [${d.singular}.id, labels[i]]));` }),
  (d, labels, items, question) => { const counts = {};
    for (const i of items) counts[i.label] = (counts[i.label] ?? 0) + 1;
    return { returns: 'Record<string, number>', expected: counts,
      instructions: `Sort the ${d.plural} in ${d.plural} into ${labels.map(l => JSON.stringify(l)).join(', ')} and return how many fall under each label, leaving out labels with none.`,
      code: `const labels = await Promise.all(${d.plural}.map(${d.singular} => nl<${d.union}>\`${question}\`(${d.singular})));\nconst counts: Record<string, number> = {};\nfor (const label of labels) counts[label] = (counts[label] ?? 0) + 1;\nreturn counts;` }; },
  (d, labels, items, question) => { const counts = {};
    for (const i of items) counts[i.label] = (counts[i.label] ?? 0) + 1;
    const best = Object.keys(counts).sort((a, b) => counts[b] - counts[a] || a.localeCompare(b))[0];
    return { returns: d.union, expected: best,
      instructions: `Which of ${labels.map(l => JSON.stringify(l)).join(', ')} fits the most ${d.plural} in ${d.plural}? Judge each ${d.singular}; on a tie, return the label that comes first alphabetically.`,
      code: `const labels = await Promise.all(${d.plural}.map(${d.singular} => nl<${d.union}>\`${question}\`(${d.singular})));\nconst counts: Record<string, number> = {};\nfor (const label of labels) counts[label] = (counts[label] ?? 0) + 1;\nreturn Object.keys(counts).sort((a, b) => counts[b] - counts[a] || a.localeCompare(b))[0] as ${d.union};` }; },
];

export function labeledJudgments(seed, index, split = 'train') {
  const rng = new Random(seed, `labeled:${index}`);
  const name = rng.pick(Object.keys(DATASETS)), d = DATASETS[name];
  const labelSpecs = labelsFor(rng, name, split), labels = Object.keys(labelSpecs);
  const byClass = !!d.union && rng.next() < 0.4;
  const target = byClass ? undefined : rng.pick(d.only ?? labels);
  // Items of the case's labels; a single-label dataset (spam) takes its other items from the rest of the dataset.
  const pool = d.labels && d.only ? [...new Set(rowsOf(name, split).map(row => row.label))] : labels;
  const items = itemsFor(rng, name, pool, rng.int(5, 9), target, split)
    .map((row, i) => ({ id: `${d.idPrefix}${i + 1}`, text: row.text, day: rng.int(1, 15), label: row.label,
      sourceId: sourceRecordId(name, row.text, '') }));
  const question = d.classify ?? `Which of ${labels.map(l => JSON.stringify(l)).join(', ')} is request about?`;
  const task = byClass ? rng.pick(CLASS_TASKS)(d, labels, items, question) :
    rng.pick(TASKS)(d, target, items, labelSpecs[target], rng);
  const judgeEach = rng.next() < 0.5 ? ` Judge each ${d.singular} on its own.` : '';
  const inputs = { [d.plural]: items.map(({ id, text, day }) => ({ id, text, day })) };
  const types = `export type ${d.type} = { id: string, text: string, day: number };\n` +
    (d.union ? `export type ${d.union} = ${quoteUnion(labels)};\n` : '');
  const shape = `${name}${index}`;
  return [curriculumCase({ family: 'labeled_judgments', shape, variant: 'v0', splitGroup: `labeled:${shape}`,
    slice: 'inline_placement', domain: 'other', mode: 'single_call', inline: 'required',
    evidence: { world: [`${items.length} ${d.plural} from ${SOURCES[name].name}`], retrieved: [], background: [] },
    minimumSequence: ['judge each item with an inline nl', 'combine the judgments in code'],
    reference: { root: [evalCall(task.code), returnCall(task.expected)],
      children: items.map(item => ({ match: `"${item.id}"`, value: byClass ? item.label : item.label === target })) },
    root: { name: `review_${d.plural}`, args: { [d.plural]: `${d.type}[]` }, returns: task.returns,
      instructions: task.instructions + judgeEach },
    files: { 'types.ts': types }, inputs, expected: task.expected })].map(record => {
      record.license = SOURCES[name].license;
      record.gold_sources = [`${name}-labels`];
      record.dataset = name;
      record.dataset_records = items.map(item => item.sourceId);
      return record;
    });
}
