import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { chmodSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';
import test from 'node:test';

const codebaseRoot = process.env.NATLANG_TELECOM_CODEBASE_ROOT;
const require = createRequire(import.meta.url);
const ts = require('../ts-host/node_modules/typescript/lib/typescript.js');

test('telecom journals isolate fresh same-task sessions and replay only within a session',
  { skip: !codebaseRoot }, async t => {
    const root = resolve(codebaseRoot);
    const metadata = JSON.parse(readFileSync(join(root, 'metadata.json'), 'utf8'));
    const servicePath = join(root, metadata.external_services.telecom.source_path);
    const serviceBytes = readFileSync(servicePath);
    assert.equal(createHash('sha256').update(serviceBytes).digest('hex'), metadata.external_services.telecom.source_sha256);
    const workerPath = join(root, 'source/telecom_worker.py');
    assert.equal(createHash('sha256').update(readFileSync(workerPath)).digest('hex'),
      metadata.execution_adapter.paths['source/telecom_worker.py']);
    const fixtureLine = readFileSync(join(root, metadata.external_services.telecom.fixtures_path), 'utf8')
      .split('\n').find(line => line.includes('tau2-telecom-train-047'));
    const fixtureRow = JSON.parse(fixtureLine);
    const fakeRoot = mkdtempSync(join(tmpdir(), 'telecom-journal-contract-'));
    t.after(() => rmSync(fakeRoot, { recursive: true, force: true }));
    const worker = join(fakeRoot, 'fake-python.py');
    writeFileSync(worker, `#!/usr/bin/env python3
import json, os, sys
for line in sys.stdin:
    request = json.loads(line)
    if request.get('kind') == 'init':
        with open(os.environ['NATLANG_TEST_INIT_LOG'], 'a', encoding='utf-8') as stream:
            stream.write(json.dumps(request['initialState']) + '\\n')
    value = None if request.get('kind') == 'init' else 'mock read result'
    sys.stdout.write(json.dumps({'id': request['id'], 'ok': True, 'value': value}) + '\\n')
    sys.stdout.flush()
`);
    chmodSync(worker, 0o755);
    const source = serviceBytes.toString('utf8')
      .replace('/*__NATLANG_WORLD_FIXTURE__*/', JSON.stringify(fixtureRow.fixture))
      .replace('/*__NATLANG_CODEBASE_ROOT__*/', JSON.stringify(root))
      .replace('/*__NATLANG_RUNTIME_TREE_tau2_python_deps__*/',
        JSON.stringify(metadata.execution_adapter.runtime_trees[0].root));
    const compiled = ts.transpile(`${source}\nexport { createTau2Bridge as testCreateTau2Bridge };`, {
      module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022,
    });

    async function freshModule(label) {
      const modulePath = join(fakeRoot, `${label}.mjs`);
      writeFileSync(modulePath, compiled);
      const journalRoot = join(fakeRoot, `${label}-spool`);
      const initLog = join(fakeRoot, `${label}-worker-inits.jsonl`);
      process.env.NATLANG_TEACHER_JOURNAL_SPOOL_ROOT = journalRoot;
      process.env.NATLANG_TEST_INIT_LOG = initLog;
      const module = await import(pathToFileURL(modulePath).href);
      const stateDir = readdirSync(journalRoot).find(name => name.startsWith(`telecom-${fixtureRow.fixture.caseRef}-`));
      assert.ok(stateDir, 'service creates a fresh case-scoped journal directory');
      return { module, journalPath: join(journalRoot, stateDir, 'journal.jsonl'), initLog, stateDir };
    }
    async function oneRead(module, args) {
      const bridge = await module.testCreateTau2Bridge(args);
      try { return await bridge.device.checkStatusBar(); }
      finally { await bridge.shutdown(); }
    }

    const first = await freshModule('attempt-one');
    const args = {
      ...fixtureRow.fixture,
      caseSessionId: first.stateDir,
      workerSha256: metadata.execution_adapter.paths['source/telecom_worker.py'],
      journalPath: first.journalPath,
      python: worker,
      repo: '/mnt/external/external-rl-sources/tau2_bench/repo',
      sourceRoot: root,
    };
    const firstReceipt = await oneRead(first.module, args);
    assert.equal(firstReceipt.sequence, 1);
    assert.equal(firstReceipt.caseSessionId, args.caseSessionId);

    const resumedReceipt = await oneRead(first.module, args);
    assert.equal(resumedReceipt.sequence, 2);
    assert.equal(resumedReceipt.caseSessionId, args.caseSessionId);
    const firstInitStates = readFileSync(first.initLog, 'utf8').trim().split('\n').map(JSON.parse);
    assert.deepEqual(firstInitStates[0].replay_calls, []);
    assert.equal(firstInitStates[1].replay_calls.length, 1, 'same session restores its committed receipt');

    const second = await freshModule('attempt-two');
    const freshArgs = { ...args, caseSessionId: second.stateDir, journalPath: second.journalPath };
    const secondReceipt = await oneRead(second.module, freshArgs);
    assert.equal(secondReceipt.sequence, 1, 'fresh same-task world starts its own sequence');
    assert.notEqual(freshArgs.caseSessionId, args.caseSessionId);
    const secondInit = JSON.parse(readFileSync(second.initLog, 'utf8').trim());
    assert.deepEqual(secondInit.replay_calls, [], 'fresh session does not replay another attempt');

    const firstRows = readFileSync(first.journalPath, 'utf8').trim().split('\n').map(JSON.parse);
    assert.equal(firstRows[0].kind, 'case-session-start');
    assert.equal(firstRows[0].caseSessionId, args.caseSessionId);
    assert.equal(firstRows[0].workerSha256, args.workerSha256);
    assert.deepEqual(firstRows[0].initialCounters, { committedReceipts: 0, nextSequence: 0 });
    assert.ok(firstRows.slice(1).every(row => row.caseSessionId === args.caseSessionId &&
      row.workerSha256 === args.workerSha256));
    const secondRows = readFileSync(second.journalPath, 'utf8').trim().split('\n').map(JSON.parse);
    assert.equal(secondRows[0].sourceTaskKey, firstRows[0].sourceTaskKey);
    assert.notEqual(secondRows[0].caseSessionId, firstRows[0].caseSessionId);
    assert.deepEqual(secondRows[0].initialState, firstRows[0].initialState);
  });
