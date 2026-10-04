import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, mkdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { loadSkills, memorySkillSource, parseMarkdownSkill, renderSkillListing } from '../dist/skills/index.js';
import { neuraleseRef, encodeMessages } from '../dist/native/neuralese.js';
import { MemoryNeuraleseStore, StandInNeuralesePort, hashingEmbedder } from '../dist/native/neuralese-store.js';
import { createNatlangRuntime, loadNatlang } from '../dist/index.js';

const ref = neuraleseRef('Neuralese<string>', 'nz1_' + 'a'.repeat(52));
const document = (description, summary) => '---\n' + JSON.stringify({ name: 'review', description, ...(summary !== undefined ? { summary } : {}) }) + '\n---\nCheck the evidence.\n';

test('semantic discovery metadata accepts soft values; identity remains crisp', async () => {
  const source = memorySkillSource({ 'skills/review/SKILL.md': document(ref, ref) });
  const { set, diagnostics } = await loadSkills(source);
  assert.deepEqual(diagnostics, []);
  assert.deepEqual(set.get('review').description, ref);
  assert.deepEqual(set.get('review').summary, ref);
  assert.ok(Object.isFrozen(set.get('review').description.$neuralese));
  const { messages, blocks } = encodeMessages([{ role: 'system', content: renderSkillListing(set) }]);
  assert.equal(blocks, 2);
  assert.deepEqual(messages[0].content.filter(p => p.type === 'neuralese'), [{ type: 'neuralese', id: ref.$neuralese.id }, { type: 'neuralese', id: ref.$neuralese.id }]);
  assert.ok(!JSON.stringify(messages).includes('[object Object]'));
});

test('malformed or wrongly typed descriptions/summaries are rejected', () => {
  for (const [description, summary] of [[{ $neuralese: { type: 'Neuralese<string>', id: 'bad' } }],
    [neuraleseRef('Neuralese<number>', ref.$neuralese.id)], ['valid', 42], ['valid', '']]) {
    const diagnostics = [];
    assert.equal(parseMarkdownSkill('skills/review', { 'SKILL.md': document(description, summary) }, diagnostics), undefined);
    assert.ok(diagnostics.some(d => d.severity === 'error'));
  }
  const diagnostics = [];
  const qualified = neuraleseRef('Neuralese<string, "nd:natlang@1">', ref.$neuralese.id);
  assert.ok(parseMarkdownSkill('skills/review', { 'SKILL.md': document(qualified) }, diagnostics));
  assert.deepEqual(diagnostics, []);
});

test('.nz metadata retains soft descriptions through the existing host hook', async () => {
  const { set, diagnostics } = await loadSkills(memorySkillSource({ 'skills/review.nz': new Uint8Array([0]) }),
    { nz: { async readMetadata() { return { skill: { name: 'review', description: ref, summary: ref, body: 'instructions' } }; } } });
  assert.deepEqual(diagnostics, []);
  assert.deepEqual(set.get('review').description, ref);
  assert.equal(encodeMessages([{ content: renderSkillListing(set) }]).blocks, 2);
});

test('runtime discovery sends soft descriptions to a Neuralese-capable driver', async () => {
  const store = new MemoryNeuraleseStore();
  const port = new StandInNeuralesePort(store, hashingEmbedder(8), 8, 'nd:natlang@1');
  const block = await port.write('Review payment disputes and check cited evidence.');
  const description = neuraleseRef('Neuralese<string>', block.id);
  const root = mkdtempSync(join(tmpdir(), 'natlang-soft-skill-'));
  mkdirSync(join(root, 'triage', 'skills', 'review'), { recursive: true });
  writeFileSync(join(root, 'triage.nl'), '---\nargs:\n  issue: string\nreturns: string\n---\nUse relevant skills.\n');
  writeFileSync(join(root, 'triage', 'skills', 'review', 'SKILL.md'), document(description));
  const triage = loadNatlang(join(root, 'triage.nl'), root);
  const seen = [];
  const model = Object.assign(async request => {
    seen.push(request.messages);
    return { calls: [['return_result', { status: 'success', value: 'reviewed' }]] };
  }, { neuralese: true });
  const runtime = createNatlangRuntime({ model, neuralese: { store, port } });
  assert.equal(await runtime.run(() => triage('payment dispute')), 'reviewed');
  const parts = seen.flat().flatMap(m => Array.isArray(m.content) ? m.content : []);
  assert.ok(parts.some(p => p.type === 'neuralese' && p.id === block.id));
  const unsupported = createNatlangRuntime({ model: async () => { throw new Error('should not call text-only driver'); }, neuralese: { store, port } });
  await assert.rejects(() => unsupported.run(() => triage('payment dispute')), /neuralese-unsupported-backend/);
});
