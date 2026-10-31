import { test } from 'node:test';
import assert from 'node:assert/strict';
import { broadGraphCase, caseProblem, CASE_SCHEMA } from '../scripts/inline-curriculum/build-broad-graph-calls.mjs';

const item = (pattern) => ({ schema: CASE_SCHEMA, id: 'hotpotqa:1', source: 'hotpotqa', split: 'train', pattern,
  groups: ['hotpotqa-q:1'], request: { name: 'question', text: 'Which magazine started first?' }, answer: "Arthur's Magazine",
  view: { name: 'find_evidence', description: 'd', instructions: 'i', none: 'No relevant facts.' },
  consumer: { name: 'answer_from_notes', description: 'd', instructions: 'i' },
  root: { name: 'answer_question', instructions_root: 'print each result', instructions_child: 'collect the results' },
  documents: [{ title: 'A', text: "# A\nArthur's Magazine was started in 1844.", view: "Arthur's Magazine was started in 1844." },
    { title: 'B', text: '# B\nEchosmith is a band.', view: 'No relevant facts.' }] });

test('a case becomes a folder program whose root never reads the documents', () => {
  for (const pattern of ['fan-in/root', 'fan-in/child']) {
    assert.equal(caseProblem(item(pattern)), null);
    const { record } = broadGraphCase(item(pattern));
    const code = record.curriculum.reference.root[0][1].code;
    assert.match(code, /await find_evidence\(await file\.readText\(\), question\)/);
    assert.doesNotMatch(code, /console\.log\(await file/);
    assert.deepEqual(Object.keys(record.semantics.inputs), ['question']);
    assert.equal(Object.keys(record.semantics.folder_files).length, 2);
    assert.equal(record.broad_graph.producers, 2);
    const children = record.curriculum.reference.children;
    assert.equal(children.length, pattern === 'fan-in/root' ? 2 : 3);
    if (pattern === 'fan-in/child') assert.match(code, /return await answer_from_notes\(question, notes\)/);
  }
});

test('cases without an informative view or with short views are refused', () => {
  const none = item('fan-in/root');
  none.documents = none.documents.map(doc => ({ ...doc, view: 'No relevant facts.' }));
  assert.equal(caseProblem(none), 'no-informative-view');
  const short = item('fan-in/root');
  short.documents[0].view = 'too short';
  assert.equal(caseProblem(short), 'view-under-edge-minimum');
});
