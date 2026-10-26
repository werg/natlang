import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdir, mkdtemp, rm, symlink, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { applicationSplit, checkApplicationInterfaces, checkDomainContracts, checkFixture, checkNativeCalls, checkOperationalCoverage, checkSourceCoverage, confinedFile, inspectPotentialCalls, json, makeRecord, oracleLevel,
  REPO, sha256, sourceDigest, sourceFiles, verifyPacket } from './lib.mjs';
import { checkOracle } from '../../dist/evaluation/oracles.js';

const files = {
  'types.ts': 'export type Evidence = { id: string; text: string };\n',
  'qualify.nl': '---\nargs:\n  evidence: Evidence[]\n  claim: string\nreturns: \'"pass" | "fail" | "needs-review"\'\n---\n1. Compare the claim with the supplied evidence. Return pass when established, fail when contradicted, and needs-review when undecidable.\n',
};
const fixture = {
  id: 'acceptance', root: 'qualify.nl', split_group: 'one-world',
  inputs: { evidence: [{ id: 'record-1', text: 'Mira accepted the European launch on 1 October.' }], claim: 'Mira accepted the European launch.' },
  expected: 'pass', oracle: 'exact', scenario: 'Explicit owner acceptance supports a scoped launch decision.', difficulty: 'leaf',
};
const app = { application_id: '001', source_sha256: 'a'.repeat(64), source_relative_path: 'applications/001.md',
  stages: [{ name: 'extractDecisions', algorithm: 'Resolve attributable acceptance.', gate: 'Acceptance and scope are supported.' }] };

async function temporary(t) {
  await mkdir(join(REPO, '.natlang'), { recursive: true });
  const path = await mkdtemp(join(REPO, '.natlang/design-brief-test-'));
  t.after(() => rm(path, { recursive: true, force: true }));
  return path;
}

test('the declared split holds entire application families out in all twenty domains', () => {
  const counts = { train: 0, dev: 0, test: 0 };
  for (let domain = 0; domain < 20; domain++) {
    const assignments = Array.from({ length: 5 }, (_, index) => applicationSplit(domain * 5 + index + 1));
    assert.deepEqual(assignments, ['train', 'train', 'train', 'dev', 'test']);
    for (const split of assignments) counts[split]++;
  }
  assert.deepEqual(counts, { train: 60, dev: 20, test: 20 });
  for (const id of [0, 101, 'unknown', 1.5]) assert.throws(() => applicationSplit(id));
});

test('native input and expected-result contracts reject invented or incomplete fixture shapes', () => {
  assert.equal(checkFixture(files, fixture).level, 'exact');
  assert.throws(() => checkFixture(files, { ...fixture, inputs: { ...fixture.inputs, evidence: [{ id: 'record-1' }] } }));
  assert.throws(() => checkFixture(files, { ...fixture, inputs: { ...fixture.inputs, extra: 'hidden answer' } }));
  assert.throws(() => checkFixture(files, { ...fixture, expected: 'approved' }));
  assert.throws(() => checkFixture(files, { ...fixture, root: '../qualify.nl' }));
  assert.throws(() => checkFixture(files, { ...fixture, oracle: { kind: 'exact' } }));
});

test('domain contracts reject erased enums, missing fields, and floating money amounts', () => {
  const domainFiles = {
    'runCase.nl': '---\nargs: { invoice: Invoice }\nreturns: Invoice\n---\nReturn the supplied invoice.\n',
    'types.ts': 'export type Decimal = string; export type Ratio = { numerator: string; denominator: string };\n' +
      'export type Quantity = { value: Decimal; unit: string };\n' +
      'export type Money = { minorUnits: string; currency: string; scale: number };\n' +
      'export type Invoice = { id: string; total: Money; state: "open" | "paid" };\n',
  };
  const contract = { Invoice: { id: 'ID', total: 'Money', state: '"open" | "paid"' } };
  assert.deepEqual(checkDomainContracts(domainFiles, contract), { records: 1, fields: 3 });
  for (const changed of [
    domainFiles['types.ts'].replace('state: "open" | "paid"', 'state: string'),
    domainFiles['types.ts'].replace('id: string; total', 'total'),
    domainFiles['types.ts'].replace('minorUnits: string', 'minorUnits: number'),
  ]) assert.throws(() => checkDomainContracts({ ...domainFiles, 'types.ts': changed }, contract));
});

test('a sibling mentioned in prose must be moved into actual native callable scope', () => {
  const audit = '---\nargs: { claim: string }\nreturns: string\n---\nCall qualify(claim) and return its result.\n';
  const qualify = '---\nargs: { claim: string }\nreturns: string\n---\nReturn the supplied claim.\n';
  assert.throws(() => checkNativeCalls({ 'audit.nl': audit, 'qualify.nl': qualify }), /outside its native callable scope/);
  assert.deepEqual(checkNativeCalls({ 'audit.nl': audit, 'audit/qualify.nl': qualify }),
    { functions: 2, visible_references: 1 });
});

test('explicit native calls obey positional signatures rather than a single named-argument map', () => {
  const child = '---\nargs: { mission: string, change: string }\nreturns: string\n---\nInterpret the changed mission.\n';
  const parent = body => ({ 'plan.nl': `---\nargs: { mission: string, change: string }\nreturns: string\n---\n${body}\n`, 'plan/interpret.nl': child });
  assert.equal(checkNativeCalls(parent('Bind result = interpret(mission, change).'), 'plan.nl').visible_references, 1);
  assert.throws(() => checkNativeCalls(parent('Call interpret({ mission, change }).'), 'plan.nl'), /2 positional arguments.*supplies 1/);
  assert.throws(() => checkNativeCalls(parent('Call interpret(mission, change, "extra").'), 'plan.nl'), /supplies 3/);
  assert.throws(() => checkNativeCalls(parent('Bind result = plan/interpret(mission, change).'), 'plan.nl'), /callable name/);
  const duplicate = { 'plan.nl': '---\nargs: { mission: string; mission: string }\nreturns: string\n---\nReturn mission.\n' };
  assert.throws(() => checkNativeCalls(duplicate, 'plan.nl'), /duplicate native parameter/);
});

test('existing exact scorer rewards only the finite answer; judged workflow never auto-passes', async () => {
  for (const answer of ['pass', 'fail', 'needs-review']) {
    const correct = await checkOracle(answer, answer, 'exact');
    assert.equal(correct.accepted, true);
    for (const wrong of ['pass', 'fail', 'needs-review'].filter(value => value !== answer))
      assert.equal((await checkOracle(wrong, answer, 'exact')).accepted, false);
  }
  const pending = await checkOracle({ plausible: true }, { require: 'source support' },
    { level: 'judged', rubric: 'Compare each concrete source claim and all domain invariants independently.' });
  assert.equal(pending.accepted, false);
  assert.equal(pending.needs_review, true);
});

test('source snapshots exclude oracle files, generated declarations and symlink escapes', async t => {
  const root = await temporary(t);
  await writeFile(join(root, 'qualify.nl'), files['qualify.nl']);
  await writeFile(join(root, 'types.ts'), files['types.ts']);
  await writeFile(join(root, 'cases.json'), json([fixture]));
  await writeFile(join(root, 'qualify.d.nl.ts'), 'generated declarations');
  assert.deepEqual(await sourceFiles(root), files);
  await symlink(join(root, 'cases.json'), join(root, 'secret.ts'));
  await assert.rejects(() => sourceFiles(root), /symlink/);
  await assert.rejects(() => confinedFile(root, '../elsewhere'));
});

test('record identity and source hash are deterministic; expected answers stay outside visible source and inputs', () => {
  const a = makeRecord(app, files, fixture, 'b'.repeat(64));
  const b = makeRecord(app, Object.fromEntries(Object.entries(files).reverse()), fixture, 'b'.repeat(64));
  assert.equal(sourceDigest(a.semantics.files), sourceDigest(b.semantics.files));
  assert.equal(a.id, b.id);
  assert.deepEqual(a.source_groups, ['design-briefs:application-001']);
  assert.equal(a.generation.provider_calls, 0);
  assert.equal(a.generation.teacher_trajectory, false);
  assert.equal(a.collection_guidance.training_admission, false);
  assert.equal(a.curriculum.rl.qualification, 'source-only-live-pending');
  assert.deepEqual(a.semantics.inputs, fixture.inputs);
  assert.equal(Object.hasOwn(a.semantics.files, 'cases.json'), false);
});

test('explicit source inventories exclude retired implementations and reject unsafe additions', async t => {
  const root = await temporary(t);
  for (const [path, text] of Object.entries(files)) await writeFile(join(root, path), text);
  await writeFile(join(root, 'retired.nl'), 'this abandoned implementation must never execute');
  const manifest = { source_files: Object.keys(files) };
  await writeFile(join(root, 'application.json'), json(manifest));
  assert.deepEqual(await sourceFiles(root), files);
  for (const paths of [['qualify.nl', 'qualify.nl'], ['../other.nl'], ['cases.json'], ['qualify.d.nl.ts']]) {
    await writeFile(join(root, 'application.json'), json({ source_files: paths }));
    await assert.rejects(() => sourceFiles(root));
  }
});

test('packet validation detects content tampering and unlisted path traversal', async t => {
  const root = await temporary(t), bytes = Buffer.from('one source row\n');
  await writeFile(join(root, 'train.gates.ir.jsonl'), bytes);
  const manifest = { schema: 'natlang.design-brief-packet/1', files: [
    { path: 'train.gates.ir.jsonl', bytes: bytes.length, sha256: sha256(bytes) },
  ] };
  await writeFile(join(root, 'manifest.json'), json(manifest));
  assert.equal((await verifyPacket(root)).files.length, 1);
  await writeFile(join(root, 'train.gates.ir.jsonl'), 'corrupted data\n');
  await assert.rejects(() => verifyPacket(root), /content changed/);
  manifest.files[0].path = '../outside.jsonl';
  await writeFile(join(root, 'manifest.json'), json(manifest));
  await assert.rejects(() => verifyPacket(root), /unsafe relative/);
});

test('only native oracle declarations are accepted', () => {
  for (const level of ['exact', 'normalized', 'judged']) assert.equal(oracleLevel({ level }), level);
  for (const spec of [null, {}, { kind: 'exact' }, { level: 'always-pass' }]) assert.throws(() => oracleLevel(spec));
});

test('original use cases cannot be replaced by stage descriptions or duplicate workflows', () => {
  const details = [
    'Accept a correct unconventional solution without forcing the standard wording.',
    'Repair an ambiguous task before diagnosing a misconception.',
    'Resume an interrupted session without revealing hidden diagnostic labels.',
    'Change the presentation while preserving the competency under assessment.',
    'Offer a bounded instructor handoff after repeated unsuccessful attempts.',
  ];
  const source = { source_sha256: app.source_sha256, details };
  const metadata = { ...app, operational_details: details };
  const cases = details.map((detail, i) => ({ id: `workflow-${i}`, root: 'runCase.nl',
    operational_detail_id: i + 1, brief_detail: detail, inputs: { session: i, observation: detail },
    oracle: { level: 'judged' }, boundary_assertions: {
      view: 'The learner view must not reveal the hidden diagnostic label.',
      durable_state: 'The proposed saved state retains the last independently completed task.',
      effects: 'The handoff remains an unexecuted proposal until the host authorizes it.',
    } }));
  assert.deepEqual(checkOperationalCoverage(metadata, cases, source), { original_details: 5, mapped_workflows: 5 });
  assert.throws(() => checkOperationalCoverage({ ...metadata, operational_details: details.map(() => 'Run the stage.') }, cases, source), /original brief use cases/);
  assert.throws(() => checkOperationalCoverage(metadata, [cases[0], cases[0], ...cases.slice(2)], source), /distinct original/);
  assert.throws(() => checkOperationalCoverage(metadata, [{ ...cases[0], brief_detail: details[1] }, ...cases.slice(1)], source), /bind its original/);
  assert.throws(() => checkOperationalCoverage(metadata, [{ ...cases[0], boundary_assertions: {} }, ...cases.slice(1)], source), /proposal assertion/);
  assert.throws(() => checkOperationalCoverage(metadata, cases, { ...source, source_sha256: 'wrong' }), /source binding/);
  const copied = { ...cases[0], inputs: { evidence: [{ text: details[0] }] } };
  assert.throws(() => checkOperationalCoverage(metadata, [copied, ...cases.slice(1)], source), /copied into observed evidence/);
});

test('each application owns its native interface; unrelated API shapes stay distinct', () => {
  const sources = {
    'types.ts': 'export type Interval = { start: string; end: string };',
    'reopenPremises.nl': '---\nargs: { changed: Interval, decisionIds: string[] }\nreturns: string[]\n---\nReturn decisionIds affected by the changed interval.\n',
    'reviseScore.nl': '---\nargs: { voices: string[], tempo: number, annotations: string[] }\nreturns: { measures: string[]; unresolved: string[] }\n---\nRead the score annotations and preserve distinct voices.\n',
  };
  const own = (path, name) => ({ schema: 'natlang.application/1', id: '001', title: 'Test', entrypoint: path,
    interfaces: [{ name, path, purpose: 'A distinct domain operation with its own contract.' }] });
  const a = checkApplicationInterfaces(sources, own('reopenPremises.nl', 'reopenPremises'), { ...app, title: 'Test' });
  const b = checkApplicationInterfaces(sources, own('reviseScore.nl', 'reviseScore'), { ...app, title: 'Test' });
  assert.deepEqual(a.interfaces[0].parameters, ['changed', 'decisionIds']);
  assert.deepEqual(b.interfaces[0].parameters, ['voices', 'tempo', 'annotations']);
  const bad = own('reopenPremises.nl', 'reopenPremises');
  bad.interfaces[0].stateParameter = 'genericCase';
  assert.throws(() => checkApplicationInterfaces(sources, bad, { ...app, title: 'Test' }), /not a declared/);
  assert.throws(() => checkApplicationInterfaces(sources, own('../escape.nl', 'escape'), { ...app, title: 'Test' }), /unsafe/);
});

test('potential-call evidence distinguishes referenced children from unused wrappers', () => {
  const native = (body) => `---\nargs: { value: string }\nreturns: string\n---\n${body}\n`;
  const graph = inspectPotentialCalls({
    'reconcile.nl': native('Call compare(value) and use the returned conflict.'),
    'reconcile/compare.nl': native('Call readWitness(value) before comparing authority.'),
    'reconcile/compare/readWitness.nl': native('Read the original witness and return its statement.'),
    'reconcile/unused.nl': native('Return value.'),
  }, ['reconcile.nl']);
  assert.equal(graph.edges.length, 2);
  assert.deepEqual(graph.unreferenced, ['reconcile/unused.nl']);
  assert.match(graph.evidence, /not observed/);
});

test('saved native state must fit the declared resume parameter and exist in every result', () => {
  const sources = { 'types.ts': 'export type Ledger = { revision: number; entries: string[] };',
    'review.nl': '---\nargs: { ledger: Ledger }\nreturns: { revision: number; next: Ledger }\n---\nReview each entry and return its resulting ledger.\n' };
  const manifest = { schema: 'natlang.application/1', id: '001', title: 'Ledger', entrypoint: 'review.nl',
    interfaces: [{ name: 'review', path: 'review.nl', purpose: 'Review the ledger and resume its exact approved state.',
      stateParameter: 'ledger', stateResultPath: ['next'] }] };
  const metadata = { ...app, title: 'Ledger' };
  assert.equal(checkApplicationInterfaces(sources, manifest, metadata).interfaces.length, 1);
  const change = path => ({ ...manifest, interfaces: [{ ...manifest.interfaces[0], stateResultPath: path }] });
  assert.throws(() => checkApplicationInterfaces(sources, change(['revision']), metadata), /does not fit/);
  assert.throws(() => checkApplicationInterfaces(sources, change(['absent']), metadata), /every result/);
  assert.throws(() => checkApplicationInterfaces(sources, change(['__proto__']), metadata), /safe property path/);
  const optional = { ...sources, 'review.nl': sources['review.nl'].replace('next: Ledger', 'next?: Ledger') };
  assert.throws(() => checkApplicationInterfaces(optional, manifest, metadata), /every result/);
});

test('package uses edges survive source freezing and are counted by native callable identity', async t => {
  const root = await temporary(t);
  await mkdir(join(root, 'evidence'));
  await writeFile(join(root, 'natlang.json'), '{}\n');
  await writeFile(join(root, 'review.nl'), '---\nargs: { text: string }\nreturns: string\nuses: [evidence/readWitness]\n---\nCall readWitness(text) before reviewing the claim.\n');
  await writeFile(join(root, 'evidence/readWitness.nl'), '---\nargs: { text: string }\nreturns: string\n---\nReturn the attributed witness passage.\n');
  const frozen = await sourceFiles(root);
  assert.equal(frozen['natlang.json'], '{}\n');
  assert.equal(checkNativeCalls(frozen, 'review.nl').visible_references, 1);
  const graph = inspectPotentialCalls(frozen, ['review.nl']);
  assert.deepEqual(graph.edges, [{ caller: 'review.nl', callee: 'evidence/readWitness.nl' }]);
  assert.deepEqual(graph.unreferenced, []);
});

test('original requirements require concrete source owners without prescribing a common graph', () => {
  const metadata = { stages: [{ name: 'reopen' }, { name: 'resolve' }], operational_details: ['A changed premise', 'An owner response'] };
  const manifest = { coverage: {
    domainGates: [{ stage: 'reopen', paths: ['change.nl'] }, { stage: 'resolve', paths: ['response.nl'] }],
    operationalRules: [{ id: 1, paths: ['change.nl'] }, { id: 2, paths: ['response.nl'] }],
  } };
  const sources = { 'change.nl': 'A premise change.', 'response.nl': 'An owner response.' };
  assert.equal(checkSourceCoverage(sources, manifest, metadata).domain_gates, 2);
  assert.throws(() => checkSourceCoverage({ 'change.nl': sources['change.nl'] }, manifest, metadata), /missing NatLang owner/);
  const duplicate = structuredClone(manifest); duplicate.coverage.domainGates[1].stage = 'reopen';
  assert.throws(() => checkSourceCoverage(sources, duplicate, metadata), /every original/);
});
