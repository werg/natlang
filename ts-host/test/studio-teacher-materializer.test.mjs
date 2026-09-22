import assert from 'node:assert/strict';
import { test } from 'node:test';
import { materializeStudioRow } from '../dist/teacher/studio-materializer.js';

function fixture(accepted = true) {
  return { schema: 'natlang.studio_teacher_trajectory/1', id: 'studio-teacher-1',
    case: { id: 'case-1', target: 'studio:wiki', split: 'train', source_revision: 'rev-1',
      expected: { state: { value: 4 }, ok: true } },
    provenance: { model: 'teacher', source_revision: 'rev-1' }, outcome: { accepted },
    runs: { reducer: { trace: [{ seq: 1 }] }, view: { trace: [{ seq: 2 }] } },
    exchanges: [{ request: { messages: [{ role: 'user', content: 'Do it.' }],
      tools: [{ type: 'function', function: { name: 'eval', parameters: { type: 'object' },
        'x-private': true } }] }, assistant: { content: '', reasoning: 'Compute first.',
        calls: [{ tool: 'eval', arguments: { code: '2 + 2' } }] } }] };
}

test('Studio projection emits exporter-ready, reasoning-preserving training turns', () => {
  const turns = materializeStudioRow(fixture());
  assert.equal(turns.length, 1);
  assert.equal(turns[0].skill, 'eval');
  assert.equal(turns[0].teacher_reasoning, 'Compute first.');
  assert.deepEqual(turns[0].messages, [{ role: 'user', content: 'Do it.' }]);
  assert.equal(turns[0].tools[0].function['x-private'], undefined);
  assert.deepEqual(turns[0].target.tool_calls[0].function,
    { name: 'eval', arguments: JSON.stringify({ code: '2 + 2' }) });
});

test('Studio projection skips rejected rows and rejects revision mismatches', () => {
  assert.deepEqual(materializeStudioRow(fixture(false)), []);
  const row = fixture(); row.provenance.source_revision = 'other';
  assert.throws(() => materializeStudioRow(row), /revision mismatch/);
});
