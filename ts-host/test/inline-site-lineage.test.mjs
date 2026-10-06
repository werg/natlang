import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createHash } from 'node:crypto';
import { createNatlangRuntime, defineNatlang } from '../dist/index.js';
import { scriptedModel } from './support/natlang.mjs';

const digest = value => createHash('sha256').update(value).digest('hex');

test('actual eval sites retain action identity and distinct repeated child lineage', async () => {
  const code = 'const limit = 17; const first = nl<boolean>`Check note against ${limit}.`; const second = nl<boolean>`Check note against ${limit}.`; const a = await first(note); const b = await first(note); const c = await second(note); return a && b && c;';
  const root = defineNatlang('---\nargs: { note: string }\nreturns: boolean\n---\nRun three checks.\n');
  const traces = [];
  const model = scriptedModel(opening => opening.includes('Run three checks.') ? code : 'return true;');
  assert.equal(await createNatlangRuntime({ model: model.driver, trace: t => traces.push(t) }).run(() => root('evidence')), true);
  const parent = traces.find(t => !t.parentCallId);
  const action = parent.events.find(e => e.kind === 'action' && e.name === 'eval');
  assert.equal(action.arguments.code, code);
  assert.equal(typeof action.tool_call_id, 'string');
  const children = traces.filter(t => t.parentCallId === parent.callId);
  assert.equal(children.length, 3);
  const sites = children.map(t => t.events.find(e => e.kind === 'manifest').inline_instruction_site);
  assert.equal(sites[0].definition_id, sites[1].definition_id);
  assert.notEqual(sites[0].definition_id, sites[2].definition_id);
  for (const site of sites) {
    assert.equal(site.schema, 'natlang.inline_instruction_site/1');
    assert.equal(site.origin.parentInvocationId, parent.callId);
    assert.equal(site.origin.toolCallId, action.tool_call_id);
    assert.equal(site.origin.writtenCodeSha256, digest(code));
    assert.equal(site.origin.checkedCodeSha256, digest(code));
    assert.equal(code.slice(site.template_span.start, site.template_span.end), '`Check note against ${limit}.`');
    assert.deepEqual(site.template_segments, ['Check note against ', '.']);
    assert.equal(site.interpolations[0].expression, 'limit');
    assert.equal(site.interpolations[0].rendered, '17');
    assert.equal(site.interpolations[0].type.text, '17');
  }
});

test('interpolation trace records the creation-time rendering once', async () => {
  const code = 'const rule = { limit: 17 }; const judge = nl<boolean>`Check note using ${rule}.`; rule.limit = 99; return await judge(note);';
  const root = defineNatlang('---\nargs: { note: string }\nreturns: boolean\n---\nCheck using a rule.\n');
  const traces = [];
  const model = scriptedModel(opening => opening.includes('Check using a rule.') ? code : 'return true;');
  assert.equal(await createNatlangRuntime({ model: model.driver, trace: t => traces.push(t) }).run(() => root('evidence')), true);
  const child = traces.find(t => t.parentCallId);
  const site = child.events.find(e => e.kind === 'manifest').inline_instruction_site;
  assert.equal(site.interpolations[0].rendered, '{"limit":17}');
  assert.ok(model.openings[1].includes('Check note using {"limit":17}.'));
});
