/** Dataset-backed folder cases. Reference actions run through the same collector as a teacher. */
import { Folder } from '../../dist/index.js';
import { Random, curriculumCase, evalCall, returnCall } from './lib.mjs';
import { LABELED_FIELDS, labeledRows, coeditRows, hotpotRows, cuadContracts, sourceRecordId } from './folder-data.mjs';
import { SOURCES } from './acquire.mjs';

const LABEL_DATASETS = Object.keys(LABELED_FIELDS);
const visibleName = label => label.replace(/_/g, ' ');
const labelType = labels => labels.map(label => JSON.stringify(label)).join(' | ');
const templateText = text => String(text).replaceAll('\\', '\\\\').replaceAll('`', '\\`').replaceAll('${', '\\${');

function pickRows(rng, dataset, split) {
  const rows = labeledRows(dataset, split), fields = LABELED_FIELDS[dataset];
  const labels = fields.labels ?? rng.sample([...new Set(rows.map(row => row.label))].sort(), rng.int(3, 5));
  const pool = rows.filter(row => labels.includes(row.label));
  const count = Math.min(pool.length, rng.next() < 0.5 ? rng.int(20, 45) : rng.int(140, 200));
  if (count < 20) throw new Error(`${dataset}/${split} has only ${count} suitable records; need at least 20`);
  return { labels, rows: rng.sample(pool, count).map(row => ({ ...row, fileId: row.id.slice(0, 16) })) };
}

function folderFiles(rows) {
  const data = rows.map(row => ({ id: row.fileId, body: row.text }));
  const folder = Folder.fromData(data, { id: 'id', path: 'inbox/{id}.md', body: 'body', format: 'frontmatter' });
  return Object.fromEntries(folder.listFiles().map(file =>
    [file.path, new TextDecoder().decode(folder.readBytesSync(file.path))]));
}

function childReferences(rows) {
  return rows.map(row => ({ match: row.fileId, evidence: [row.text.slice(0, 24)],
    calls: [['read_file', { path: `${row.fileId}.md` }],
      ['return_result', { status: 'success', value: row.label }]] }));
}

function caseFor(seed, index, split, kind, chosenDataset) {
  const rng = new Random(seed, `${kind}:${index}`), dataset = chosenDataset ?? rng.pick(LABEL_DATASETS);
  if (!LABELED_FIELDS[dataset]) throw new Error(`unknown labeled dataset ${dataset}`);
  const { labels, rows } = pickRows(rng, dataset, split), files = folderFiles(rows);
  const labelsText = labels.map(visibleName).join(', ');
  const question = `${LABELED_FIELDS[dataset].question} Choose exactly one of: ${labels.map(label => JSON.stringify(label)).join(', ')}.`;
  const code = kind === 'folder_triage' ?
    `const files = await folder.files('inbox/*.md');\n` +
    `for (const file of files) {\n` +
    `  const label = await nl<${labelType(labels)}>\`${question}\`(file);\n` +
    `  await file.moveTo('by-' + label + '/');\n` +
    `}\nreturn files.length;` :
    `const files = await folder.files('inbox/*.md');\n` +
    `const counts: Record<string, number> = {};\n` +
    `for (const file of files) {\n` +
    `  const label = await nl<${labelType(labels)}>\`${question}\`(file);\n` +
    `  counts[label] = (counts[label] ?? 0) + 1;\n` +
    `}\n` +
    `const report = Object.entries(counts).sort(([a], [b]) => a.localeCompare(b)).map(([name, count]) => name + ': ' + count).join('\\n') + '\\n';\n` +
    `await folder.file('INDEX.md').writeText(report);\nreturn counts;`;
  const counts = Object.fromEntries(labels.filter(label => rows.some(row => row.label === label))
    .sort().map(label => [label, rows.filter(row => row.label === label).length]));
  const report = Object.entries(counts).map(([label, count]) => `${label}: ${count}`).join('\n') + '\n';
  const expectedFiles = kind === 'folder_triage' ? Object.fromEntries(Object.entries(files).map(([path, text]) => {
    const row = rows.find(item => path.endsWith(`${item.fileId}.md`));
    return [`by-${row.label}/${row.fileId}.md`, text];
  })) : { ...files, 'INDEX.md': report };
  const expected = kind === 'folder_triage' ? rows.length : counts;
  const shape = `${dataset}-${index}`;
  const record = curriculumCase({ family: kind, shape, variant: 'v0', splitGroup: `${kind}:${shape}`,
    slice: 'inline_placement', domain: 'other', mode: 'single_call', inline: 'required',
    evidence: { world: [`${rows.length} ${dataset} records`], retrieved: [], background: [] },
    minimumSequence: ['list inbox files', 'classify each file through a child call',
      kind === 'folder_triage' ? 'move each file into its label directory' : 'write INDEX.md with counts'],
    reference: { root: [evalCall(code), returnCall(expected)], children: childReferences(rows) },
    root: { name: kind === 'folder_triage' ? 'triage_inbox' : 'index_inbox', kind: 'directory-reducer',
      args: {}, returns: kind === 'folder_triage' ? 'number' : 'Record<string, number>',
      instructions: kind === 'folder_triage' ?
        `Classify every file in inbox/ as one of ${labelsText}. Move it into by-<label>/, keeping its filename. Return the number moved.` :
        `Classify every file in inbox/ as one of ${labelsText}. Write INDEX.md with one "label: count" line per present label, sorted by label. Return the counts.` },
    folderFiles: files, expectedFiles, inputs: {}, expected, split });
  record.semantics.oracle = 'exact';
  record.license = SOURCES[dataset].license;
  record.gold_sources = [`${dataset}-labels`];
  record.dataset = dataset;
  record.dataset_records = rows.map(row => row.id);
  record.generation.layout = { id: 'id', path: 'inbox/{id}.md', body: 'body', format: 'frontmatter' };
  return [record];
}

export const folderTriage = (seed, index, split = 'train', dataset) => caseFor(seed, index, split, 'folder_triage', dataset);
export const folderIndex = (seed, index, split = 'train', dataset) => caseFor(seed, index, split, 'folder_index', dataset);

/** CSV arithmetic conditioned on judgments over separate text files. */
export function folderMixed(seed, index) {
  const rng = new Random(seed, `folder_mixed:${index}`);
  const count = rng.next() < 0.5 ? rng.int(20, 45) : rng.int(140, 200);
  const rows = Array.from({ length: count }, (_, item) => ({
    id: `INV${String(item + 1).padStart(4, '0')}`, amount: rng.int(5, 700),
    disputed: rng.next() < 0.38,
  }));
  const folder = Folder.fromData(rows.map(({ id, amount }) => ({ id, amount })),
    { id: 'id', table: 'invoices.csv', writable: 'table' });
  const notes = new Map(rows.map(row => [row.id, row.disputed ?
    `Regarding ${row.id}, I dispute this charge. The amount is incorrect and I request a reversal.\n` :
    `Regarding ${row.id}, I approve this charge. The amount is correct and payment is authorized.\n`]));
  for (const [id, note] of notes) folder.writeText(`notes/${id}.md`, note);
  const files = Object.fromEntries(folder.listFiles().map(file =>
    [file.path, new TextDecoder().decode(folder.readBytesSync(file.path))]));
  const total = rows.filter(row => row.disputed).reduce((sum, row) => sum + row.amount, 0);
  const code = `const lines = (await folder.file('invoices.csv').readText()).trim().split(/\\r?\\n/).slice(1);\n` +
    `let total = 0;\n` +
    `for (const line of lines) {\n` +
    `  const [id, amount] = line.split(',');\n` +
    `  const file = folder.file('notes/' + id + '.md');\n` +
    `  if (await nl<boolean>\`Does this note dispute the charge?\`(file)) total += Number(amount);\n` +
    `}\nreturn total;`;
  const record = curriculumCase({ family: 'folder_mixed', shape: `invoices${index}`, variant: 'v0',
    splitGroup: `folder_mixed:invoices${index}`, slice: 'inline_placement', domain: 'other', mode: 'single_call',
    inline: 'required', evidence: { world: [`${count} invoice rows and notes`], retrieved: [], background: [] },
    minimumSequence: ['read invoices.csv', 'judge each note', 'sum amounts for disputed notes'],
    reference: { root: [evalCall(code), returnCall(total)], children: rows.map(row => ({
      match: row.id, evidence: [notes.get(row.id).slice(0, 28)],
      calls: [['read_file', { path: `${row.id}.md` }],
        ['return_result', { status: 'success', value: row.disputed }]],
    })) },
    root: { name: 'sum_disputed_invoices', kind: 'directory-reducer', args: {}, returns: 'number',
      instructions: 'For each row in invoices.csv, read its corresponding notes/<id>.md. Return the total amount of invoices whose note disputes the charge.' },
    folderFiles: files, expectedFiles: files, inputs: {}, expected: total });
  record.semantics.oracle = 'exact';
  record.generation.layout = { id: 'id', table: 'invoices.csv', writable: 'table' };
  return [record];
}

const EDIT_TASKS = {
  gec: 'correct grammar and usage', simplification: 'simplify the wording',
  paraphrase: 'paraphrase the wording', coherence: 'improve coherence', neutralize: 'make the tone neutral',
};

export function folderEdit(seed, index, split = 'train', chosenTask) {
  const rng = new Random(seed, `folder_edit:${index}`);
  const task = chosenTask ?? rng.pick(Object.keys(EDIT_TASKS));
  if (!EDIT_TASKS[task]) throw new Error(`unknown CoEdIT task ${task}`);
  const pool = coeditRows(task, split);
  const count = Math.min(pool.length, rng.next() < 0.5 ? rng.int(20, 45) : rng.int(140, 200));
  if (count < 20) throw new Error(`CoEdIT ${task}/${split} has only ${count} suitable edits; need at least 20`);
  const rows = rng.sample(pool, count).map(row => ({ ...row, fileId: row.id.slice(0, 16) }));
  const folder = Folder.fromData(rows.map(row => ({ id: row.fileId, body: row.text })),
    { id: 'id', path: 'drafts/{id}.md', body: 'body', format: 'frontmatter' });
  const files = Object.fromEntries(folder.listFiles().map(file =>
    [file.path, new TextDecoder().decode(folder.readBytesSync(file.path))]));
  const expectedFiles = Object.fromEntries(Object.entries(files).map(([path, text]) => {
    const row = rows.find(item => path.endsWith(`${item.fileId}.md`));
    return [path, text.slice(0, -row.text.length) + row.target];
  }));
  const code = `const files = await folder.files('drafts/*.md');\n` +
    `for (const file of files) {\n` +
    `  const revised = await nl<string>\`Please ${EDIT_TASKS[task]} in this draft. Return only the revised text.\`(file);\n` +
    `  const original = await file.readText();\n` +
    `  const divider = original.indexOf('\\n---\\n');\n` +
    `  if (divider < 0) throw new Error('draft is missing front matter');\n` +
    `  await file.editText(original.slice(divider + 5), revised);\n` +
    `}\nreturn files.length;`;
  const record = curriculumCase({ family: 'folder_edit', shape: `${task}-${index}`, variant: 'v0',
    splitGroup: `folder_edit:${task}:${index}`, slice: 'inline_placement', domain: 'other', mode: 'single_call',
    inline: 'required', evidence: { world: [`${count} CoEdIT ${task} drafts`], retrieved: [], background: [] },
    minimumSequence: ['list drafts', 'read each file in a child call', 'edit each draft exactly'],
    reference: { root: [evalCall(code), returnCall(count)], children: rows.map(row => ({
      match: row.fileId, evidence: [row.text.slice(0, 24)],
      calls: [['read_file', { path: `${row.fileId}.md` }],
        ['return_result', { status: 'success', value: row.target }]],
    })) },
    root: { name: 'edit_drafts', kind: 'directory-reducer', args: {}, returns: 'number',
      instructions: `For every file in drafts/, ${EDIT_TASKS[task]} while preserving its front matter and filename. Return the number of drafts edited.` },
    folderFiles: files, expectedFiles, inputs: {}, expected: count, split });
  record.semantics.oracle = 'exact';
  record.license = SOURCES.coedit.license;
  record.gold_sources = ['coedit-targets'];
  record.dataset = 'coedit';
  record.dataset_records = rows.map(row => row.id);
  record.generation.layout = { id: 'id', path: 'drafts/{id}.md', body: 'body', format: 'frontmatter' };
  return [record];
}

/** A two-document question amid distractors: the reference searches, reads, then delegates the answer. */
export function folderFind(seed, index, split = 'train') {
  const rng = new Random(seed, `folder_find:${index}`), pool = hotpotRows(split);
  if (pool.length < 3) throw new Error(`HotpotQA ${split} needs at least three records`);
  const question = rng.pick(pool);
  const additional = rng.sample(pool.filter(row => row.id !== question.id), rng.next() < 0.5 ? 2 : 16);
  const documents = new Map();
  for (const row of [question, ...additional]) for (const item of row.context)
    if (!documents.has(item.title)) documents.set(item.title, item.text);
  const records = [...documents].map(([title, body]) => ({
    id: sourceRecordId('hotpot-document', title, body).slice(0, 16), title, body,
  }));
  const folder = Folder.fromData(records, { id: 'id', path: 'wiki/{id}.md', body: 'body', format: 'frontmatter' });
  const files = Object.fromEntries(folder.listFiles().map(file =>
    [file.path, new TextDecoder().decode(folder.readBytesSync(file.path))]));
  const support = question.supports.map(title => records.find(item => item.title === title));
  const paths = support.map(item => `wiki/${item.id}.md`);
  const code = `const first = folder.file(${JSON.stringify(paths[0])});\n` +
    `const second = folder.file(${JSON.stringify(paths[1])});\n` +
    `return await nl<string>\`${templateText(question.question)} Read both supplied articles before answering.\`(first, second);`;
  const record = curriculumCase({ family: 'folder_find', shape: `hotpot-${index}`, variant: 'v0',
    splitGroup: `folder_find:${question.id}`, slice: 'inline_placement', domain: 'other', mode: 'single_call',
    inline: 'required', evidence: { world: [question.question], retrieved: support.map(item => item.title), background: [] },
    minimumSequence: ['search the wiki files', 'read the relevant articles', 'answer using both'],
    reference: { root: [['search_files', { path: 'wiki', query: support[0].title }],
      ['read_file', { path: paths[0] }], ['search_files', { path: 'wiki', query: support[1].title }],
      evalCall(code), returnCall(question.answer)], children: [{
      match: question.question.slice(0, 35), evidence: support.map(item => item.body.slice(0, 24)),
      calls: [evalCall('console.log(await first.readText()); console.log(await second.readText());'),
        returnCall(question.answer)],
    }] },
    root: { name: 'answer_from_wiki', kind: 'directory-reducer', args: {}, returns: 'string',
      instructions: `Answer this question using wiki/ and its supporting articles: ${question.question}` },
    folderFiles: files, expectedFiles: files, inputs: {}, expected: question.answer, split });
  record.semantics.oracle = 'normalized';
  record.license = SOURCES.hotpotqa.license;
  record.gold_sources = ['hotpotqa-answer'];
  record.dataset = 'hotpotqa';
  record.dataset_records = [question.id, ...additional.map(row => row.id)];
  record.generation.layout = { id: 'id', path: 'wiki/{id}.md', body: 'body', format: 'frontmatter' };
  return [record];
}

const csvCell = value => `"${String(value).replaceAll('"', '""')}"`;
/** Non-compete extraction from contract texts, with an exact report oracle. */
export function folderExtract(seed, index, split = 'train') {
  const rng = new Random(seed, `folder_extract:${index}`), pool = cuadContracts(split);
  const count = Math.min(pool.length, rng.next() < 0.5 ? rng.int(20, 35) : rng.int(70, 120));
  if (count < 20) throw new Error(`CUAD ${split} has only ${count} suitable contracts; need at least 20`);
  const chosen = rng.sample(pool, count).map(row => ({ ...row, fileId: row.id.slice(0, 16) }));
  const folder = Folder.fromData(chosen.map(row => ({ id: row.fileId, title: row.title, body: row.content })),
    { id: 'id', path: 'contracts/{id}.md', body: 'body', format: 'frontmatter' });
  const files = Object.fromEntries(folder.listFiles().map(file =>
    [file.path, new TextDecoder().decode(folder.readBytesSync(file.path))]));
  const answers = new Map(chosen.map(row => [row.fileId, row.answer]));
  const report = 'id,clause\n' + [...answers].sort(([a], [b]) => a.localeCompare(b))
    .map(([id, answer]) => `${csvCell(id)},${csvCell(answer)}`).join('\n') + '\n';
  const expectedFiles = { ...files, 'clauses.csv': report };
  const code = `const entries: Array<[string, string]> = [];\n` +
    `for (const file of await folder.files('contracts/*.md')) {\n` +
    `  const clause = await nl<string>\`Find the non-compete clause in this contract and quote its exact text. If there is none, return an empty string.\`(file);\n` +
    `  entries.push([file.name.slice(0, -3), clause]);\n` +
    `}\n` +
    `const cell = (value: string) => '"' + value.replaceAll('"', '""') + '"';\n` +
    `const csv = 'id,clause\\n' + entries.sort(([a], [b]) => a.localeCompare(b)).map(([id, clause]) => cell(id) + ',' + cell(clause)).join('\\n') + '\\n';\n` +
    `await folder.file('clauses.csv').writeText(csv);\n` +
    `return entries.filter(([, clause]) => clause.length > 0).length;`;
  const children = chosen.map(row => {
    const content = files[`contracts/${row.fileId}.md`];
    const match = row.answer ? content.indexOf(row.answer) : -1;
    const line = match >= 0 ? content.slice(0, match).split('\n').length : 1;
    // CUAD contracts sometimes put thousands of characters on one line. A line-range read can be truncated by the
    // tool display before the annotated span; a short second view makes the span itself observable to the child.
    const spanView = match < 0 ? [] : [evalCall(
      `console.log((await folder.file(${JSON.stringify(`${row.fileId}.md`)}).readText())` +
      `.slice(${Math.max(0, match - 80)}, ${match + Math.min(row.answer.length + 80, 800)}));`)];
    return { match: row.fileId, evidence: [row.answer ? row.answer.slice(0, 24) : row.title.slice(0, 24)],
      calls: [['read_file', { path: `${row.fileId}.md`, start_line: Math.max(1, line - 1), end_line: line + 2 }],
        ...spanView,
        ['return_result', { status: 'success', value: row.answer }]],
    };
  });
  const positive = chosen.filter(row => row.answer).length;
  const record = curriculumCase({ family: 'folder_extract', shape: `cuad-${index}`, variant: 'v0',
    splitGroup: `folder_extract:${index}`, slice: 'inline_placement', domain: 'other', mode: 'single_call',
    inline: 'required', evidence: { world: [`${count} contracts`], retrieved: [], background: [] },
    minimumSequence: ['inspect contracts', 'extract clauses per file', 'write clauses.csv'],
    reference: { root: [evalCall(code), returnCall(positive)], children },
    root: { name: 'extract_noncompete', kind: 'directory-reducer', args: {}, returns: 'number',
      instructions: 'For every contract in contracts/, find a non-compete clause. Write clauses.csv with id and exact clause text; use an empty clause when absent. Sort rows by id and return the number with a clause.' },
    folderFiles: files, expectedFiles, inputs: {}, expected: positive, split });
  record.semantics.oracle = 'exact';
  record.license = SOURCES.cuad.license;
  record.gold_sources = ['cuad-non-compete-spans'];
  record.dataset = 'cuad';
  record.dataset_records = chosen.map(row => row.id);
  record.generation.layout = { id: 'id', path: 'contracts/{id}.md', body: 'body', format: 'frontmatter' };
  return [record];
}
