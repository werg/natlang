import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { execFile as execFileCallback } from 'node:child_process';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import { codeProposalTurns, compileCurriculum } from '../scripts/code-corpus/curriculum.mjs';
import { CODE_ONLY_FAMILIES, CODE_ONLY_GENERATOR, syntheticCodeOnlyTasks } from '../scripts/code-corpus/code-only-families.mjs';

const functionName = family => family.name.replace(/_([a-z])/g, (_match, letter) => letter.toUpperCase());
const load = async family => {
  const moduleSource = `${family.source}\nexport { ${functionName(family)} };`;
  return import(`data:text/javascript;base64,${Buffer.from(moduleSource).toString('base64')}`);
};
const execFile = promisify(execFileCallback);

test('code-only lane is seven unique code proposals and never enters replay cases', async () => {
  const tasks = syntheticCodeOnlyTasks(42);
  assert.equal(tasks.length, 7);
  assert.equal(new Set(tasks.map(task => task.generation.family)).size, 7);
  assert.ok(tasks.every(task => task.generation.generator === CODE_ONLY_GENERATOR && task.cases.length === 0));
  assert.ok(tasks.every(task => task.behavioral_evidence.kind === 'syntax_only' && task.behavioral_evidence.status === 'not_native_replay_verified'));
  assert.deepEqual(compileCurriculum(tasks).map(row => row.kind), Array(7).fill('instruction_code_proposal'));
  const turns = await codeProposalTurns(tasks);
  assert.equal(turns.length, 7);
  assert.deepEqual(turns.map(turn => turn.family), CODE_ONLY_FAMILIES.map(family => family.name));
  assert.ok(turns.every(turn => turn.execution_verified === false && turn.behavioral_evidence.status === 'not_native_replay_verified'));
});

test('code-only implementations execute correctly on local object, async, error, and builtin fixtures', async () => {
  const modules = Object.fromEntries(await Promise.all(CODE_ONLY_FAMILIES.map(async family => [family.name, await load(family)])));
  const profile = modules.select_profile_fields[functionName(CODE_ONLY_FAMILIES[0])];
  assert.deepEqual(profile({ name:'Ada', email:'a@example.test', role:'admin' }), { name:'Ada', email:'a@example.test' });
  assert.deepEqual(profile(Object.assign(Object.create({ email:'inherited' }), { name:'Lin'})), { name:'Lin' });

  const normalize = modules.normalize_inventory[functionName(CODE_ONLY_FAMILIES[1])];
  const rows = [{ sku:' ab-2 ', quantity:4 }];
  assert.deepEqual(normalize(rows), [{ sku:'AB-2', quantity:4 }]);
  assert.equal(rows[0].sku, ' ab-2 ');

  const valid = modules.validate_service_config[functionName(CODE_ONLY_FAMILIES[2])];
  assert.equal(valid({ port:443, mode:'production' }), true);
  assert.equal(valid({ port:0, mode:'production' }), false);
  assert.equal(valid(null), false);

  const parse = modules.parse_positive_integer[functionName(CODE_ONLY_FAMILIES[3])];
  assert.deepEqual(parse('42'), { ok:true, value:42 });
  assert.deepEqual(parse('0'), { ok:false, error:'invalid positive integer' });
  assert.deepEqual(parse('9007199254740992'), { ok:false, error:'invalid positive integer' });

  const dashboard = modules.load_dashboard[functionName(CODE_ONLY_FAMILIES[4])];
  const calls = [];
  const pending = dashboard('u7', {
    async getUser(id) { calls.push(['user', id]); return { id }; },
    async getNotifications(id) { calls.push(['notifications', id]); return ['n1']; },
  });
  assert.deepEqual(calls, [['user','u7'], ['notifications','u7']]);
  assert.deepEqual(await pending, { user:{ id:'u7' }, notifications:['n1'] });

  const contextual = modules.with_error_context[functionName(CODE_ONLY_FAMILIES[5])];
  assert.equal(await contextual(async ()=>'ok'), 'ok');
  const original = new Error('disk offline');
  await assert.rejects(contextual(async()=>{ throw original; }), error => error.message === 'operation failed: disk offline' && error.cause === original);

  const sha256 = modules.sha256_hex[functionName(CODE_ONLY_FAMILIES[6])];
  assert.equal(sha256('hello'), createHash('sha256').update('hello', 'utf8').digest('hex'));
});

test('CLI includes and fingerprints the code-only lane without sending it to replay', async () => {
  const output = await mkdtemp(join(tmpdir(), 'natlang-code-only-'));
  try {
    await execFile(process.execPath, [fileURLToPath(new URL('../scripts/code-corpus/curriculum.mjs', import.meta.url)), '--out', output, '--seed', '7', '--synthetic', '1'], { timeout:15000 });
    const manifest = JSON.parse(await readFile(join(output, 'manifest.json'), 'utf8'));
    const readRows = async name => (await readFile(join(output, name), 'utf8')).trim().split('\n').filter(Boolean).map(line=>JSON.parse(line));
    assert.equal(manifest.code_only_tasks, 7);
    assert.match(manifest.config.code_only_families_sha256, /^[a-f0-9]{64}$/);
    const curriculum = await readRows('curriculum.jsonl');
    const codeOnly = curriculum.filter(row => row.generation?.generator === CODE_ONLY_GENERATOR);
    assert.equal(codeOnly.length, 7);
    assert.ok(codeOnly.every(row => row.kind === 'instruction_code_proposal' && row.target === 'code_generation'));
    assert.equal((await readRows('code-proposals.jsonl')).filter(row=>row.family!=='general_code_proposal').length, 7);
    assert.equal((await readRows('teacher-programs.jsonl')).length, 2);
  } finally {
    await rm(output, { recursive:true, force:true });
  }
});
