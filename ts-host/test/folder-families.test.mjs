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
  process.env.NATLANG_DATASETS = root;
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
  const [record] = folderMixed(3, 0);
  assert.ok(record.semantics.folder_files['payments.csv']);
  const messages = Object.entries(record.semantics.folder_files).filter(([path]) => path.startsWith('messages/'));
  assert.ok(messages.length >= 20);
  assert.ok(new Set(messages.map(([, text]) => text)).size > messages.length / 2, 'messages are real, varied text');
  const [verified] = await verifyCases([record], TOOLS_PROMPT);
  assert.equal(verified.ok, true, verified.problems.join('\n'));
});
