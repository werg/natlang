import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { verifyCases } from '../dist/teacher/curriculum.js';
import { TOOLS_PROMPT } from '../dist/native/prompt.js';

test('dataset folder cases use disjoint records and replay per-file evidence', async () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-folder-data-'));
  const revision = 'cae486f927c250fe1d4a5b55f11357964ed1646c';
  const directory = join(root, 'sms_spam', revision, 'plain_text');
  mkdirSync(directory, { recursive: true });
  writeFileSync(join(directory, 'train-00000-of-00001.parquet.jsonl'),
    Array.from({ length: 700 }, (_, index) => JSON.stringify({
      sms: `Message ${index}: ${index % 2 ? 'Claim your promotional prize today' : 'The meeting starts at nine tomorrow'}.`,
      label: index % 2 ? 'spam' : 'ham',
    })).join('\n') + '\n');
  const coeditDirectory = join(root, 'coedit', 'e9a255c33ef910bc33a9d2b522653fa87521583e');
  mkdirSync(coeditDirectory, { recursive: true });
  writeFileSync(join(coeditDirectory, 'train.jsonl'),
    Array.from({ length: 500 }, (_, index) => JSON.stringify({ _id: String(index), task: 'gec',
      src: `Fix the grammar: Invoice ${index} have a incorrect amount in the archive.`,
      tgt: `Invoice ${index} has an incorrect amount in the archive.` })).join('\n') + '\n');
  const hotpotDirectory = join(root, 'hotpotqa', '1908d6afbbead072334abe2965f91bd2709910ab', 'distractor');
  mkdirSync(hotpotDirectory, { recursive: true });
  const hotpot = Array.from({ length: 200 }, (_, index) => ({ id: `question-${index}`,
    question: `Where did Event ${index} occur according to the two articles?`, answer: `City ${index}`,
    supporting_facts: { title: [`Event ${index}`, `City ${index}`], sent_id: [0, 0] },
    context: { title: [`Event ${index}`, `City ${index}`, ...Array.from({ length: 4 }, (_, n) => `Distractor ${index}-${n}`)],
      sentences: [[`Event ${index} occurred in City ${index}.`], [`City ${index} hosted Event ${index}.`],
        ...Array.from({ length: 4 }, (_, n) => [`Unrelated fact ${index}-${n}.`])] },
  }));
  for (const [number, shard] of [[0, hotpot.slice(0, 100)], [1, hotpot.slice(100)]])
    writeFileSync(join(hotpotDirectory, `train-0000${number}-of-00002.parquet.jsonl`),
      shard.map(row => JSON.stringify(row)).join('\n') + '\n');
  const cuadDirectory = join(root, 'cuad', 'a3c393f5d103fd0c516374e4fdff676c8176dcb1', 'CUAD_v1');
  mkdirSync(cuadDirectory, { recursive: true });
  writeFileSync(join(cuadDirectory, 'CUAD_v1.json'), JSON.stringify({ data: Array.from({ length: 400 }, (_, index) => {
    const title = `Contract ${index}`;
    const clause = `The employee shall not compete with Company ${index} for two years.`;
    return { title, paragraphs: [{ context: `Agreement ${index}.\n${index % 2 ? clause : 'The parties agree to cooperate.'}\nEnd of agreement.`,
      qas: [{ id: `${title}__Non-Compete`, answers: index % 2 ? [{ text: clause, answer_start: 14 }] : [] }] }] };
  }) }));
  const bankingDirectory = join(root, 'banking77', '57ec275d8078af65b7731c2a98be812d844a6d6b', 'banking_data');
  mkdirSync(bankingDirectory, { recursive: true });
  const intents = ['card_payment_not_recognised', 'transaction_charged_twice', 'card_arrival', 'change_pin', 'exchange_rate'];
  writeFileSync(join(bankingDirectory, 'train.csv.jsonl'), Array.from({ length: 600 }, (_, index) => JSON.stringify({
    text: `Request ${index} about ${intents[index % intents.length].replaceAll('_', ' ')}, please help.`,
    category: intents[index % intents.length] })).join('\n') + '\n');
  const { appendFileSync } = await import('node:fs');
  const { SOURCE_REVIEWS } = await import('../dist/teacher/source-review.js');
  appendFileSync(join(bankingDirectory, 'train.csv.jsonl'), SOURCE_REVIEWS.map(review =>
    JSON.stringify({ text: review.text, category: review.annotatedLabel })).join('\n') + '\n');
  appendFileSync(join(directory, 'train-00000-of-00001.parquet.jsonl'),
    [{ sms: 'Identical disputed message with two conflicting labels.', label: 'ham' },
      { sms: 'Identical disputed message with two conflicting labels.', label: 'spam' }].map(JSON.stringify).join('\n') + '\n');
  appendFileSync(join(coeditDirectory, 'train.jsonl'), [
    { task: 'gec', src: 'Fix: Shared draft with a spelling eror in it.', tgt: 'Shared draft with a spelling error in it.' },
    { task: 'gec', src: 'Correct: Shared draft with a spelling eror in it.', tgt: 'A shared draft has a spelling error in it.' },
    { task: 'neutralize', src: 'Neutralize: Shared draft with a spelling eror in it.', tgt: 'This draft contains a spelling error.' },
  ].map(JSON.stringify).join('\n') + '\n');
  const { readFileSync } = await import('node:fs');
  const cuadFile = join(cuadDirectory, 'CUAD_v1.json');
  const cuad = JSON.parse(readFileSync(cuadFile, 'utf8'));
  const longSpan = 'shall not compete '.repeat(150);
  cuad.data.push({ title: 'Unusable positive', paragraphs: [{ context: longSpan,
    qas: [{ id: 'long__Non-Compete', answers: [{ text: longSpan }] }] }] });
  cuad.data.push({ title: 'Multiple spans', paragraphs: [{ context: 'Do not compete. Do not compete elsewhere.',
    qas: [{ id: 'multiple__Non-Compete', answers: [{ text: 'Do not compete.' }, { text: 'Do not compete elsewhere.' }] }] }] });
  writeFileSync(cuadFile, JSON.stringify(cuad));
  process.env.NATLANG_DATASETS = root;
  const { labeledRows, coeditRows, cuadContracts, datasetQualityReport } = await import('../scripts/inline-curriculum/folder-data.mjs');
  const bankingRows = [...labeledRows('banking77'), ...labeledRows('banking77', 'test')];
  assert.equal(bankingRows.some(row => SOURCE_REVIEWS.some(review => review.id === row.id)), false);
  assert.equal(datasetQualityReport().filter(row => row.reason === 'source_review_pending').length, SOURCE_REVIEWS.length);
  assert.equal([...labeledRows('sms_spam'), ...labeledRows('sms_spam', 'test')]
    .some(row => row.text.startsWith('Identical disputed')), false);
  const allEdits = [...coeditRows('gec'), ...coeditRows('gec', 'test')];
  const shared = allEdits.find(row => row.text.startsWith('Shared draft'));
  const neutral = [...coeditRows('neutralize'), ...coeditRows('neutralize', 'test')][0];
  assert.equal(shared.targets.length, 2);
  assert.equal(shared.id, neutral.id, 'split identity is the visible draft, across tasks/instruction wording');
  const allContracts = [...cuadContracts(), ...cuadContracts('test')];
  assert.equal(allContracts.some(row => row.title === 'Unusable positive'), false);
  assert.equal(allContracts.find(row => row.title === 'Multiple spans').answers.length, 2);
  assert.ok(datasetQualityReport().some(row => row.reason === 'conflicting_labels'));
  assert.ok(datasetQualityReport().some(row => row.reason === 'unusable_positive_spans'));
  const { folderTriage, folderIndex, folderEdit, folderFind, folderExtract } = await import('../scripts/inline-curriculum/folder-families.mjs');
  const train = folderTriage(3, 0, 'train', 'sms_spam')[0];
  const heldOut = folderIndex(3, 0, 'test', 'sms_spam')[0];
  const edits = folderEdit(3, 0, 'train', 'gec')[0];
  const heldOutEdits = folderEdit(3, 0, 'test', 'gec')[0];
  const find = folderFind(3, 0, 'train')[0];
  const heldOutFind = folderFind(3, 0, 'test')[0];
  const extract = folderExtract(3, 0, 'train')[0];
  const heldOutExtract = folderExtract(3, 0, 'test')[0];
  assert.ok(train.dataset_records.length >= 20);
  assert.equal(train.semantics.files_oracle.compare, 'moves');
  assert.equal(edits.semantics.files_oracle.return_count, 'changed');
  assert.equal(extract.semantics.files_oracle.return_count, 'csv_nonempty');
  assert.equal(find.curriculum.decisive.length, 0);
  assert.ok(find.curriculum.answer_evidence.length >= 2);
  assert.ok(heldOut.dataset_records.length >= 20);
  assert.equal(train.dataset_records.some(id => heldOut.dataset_records.includes(id)), false);
  assert.equal(edits.dataset_records.some(id => heldOutEdits.dataset_records.includes(id)), false);
  assert.equal(find.dataset_records.some(id => heldOutFind.dataset_records.includes(id)), false);
  assert.equal(extract.dataset_records.some(id => heldOutExtract.dataset_records.includes(id)), false);
  assert.equal(Object.values(train.semantics.folder_files).some(text => text.includes('label:')), false,
    'gold labels must not leak into the model-visible files');
  for (const record of [train, heldOut, edits, heldOutEdits, find, heldOutFind, extract, heldOutExtract]) {
    const [verified] = await verifyCases([record], TOOLS_PROMPT);
    assert.equal(verified.ok, true, verified.problems.join('\n'));
  }
});

test('mixed CSV and customer messages case replays a checked aggregate', async () => {
  const { folderMixed } = await import('../scripts/inline-curriculum/folder-families.mjs');
  const { quarantineReason } = await import('../dist/teacher/curriculum-policy.js');
  const [record] = folderMixed(3, 0);
  assert.equal(record.curriculum.payment_scope_version, 2);
  assert.match(record.semantics.files[record.semantics.root], /direct debits and cash withdrawals/);
  assert.equal(quarantineReason(record), undefined);
  const legacy = structuredClone(record); delete legacy.curriculum.payment_scope_version;
  assert.equal(quarantineReason(legacy), 'legacy_payment_scope');
  assert.ok(record.semantics.folder_files['payments.csv']);
  const messages = Object.entries(record.semantics.folder_files).filter(([path]) => path.startsWith('messages/'));
  assert.ok(messages.length >= 20);
  assert.ok(new Set(messages.map(([, text]) => text)).size > messages.length / 2, 'messages are real, varied text');
  const [verified] = await verifyCases([record], TOOLS_PROMPT);
  assert.equal(verified.ok, true, verified.problems.join('\n'));
});
