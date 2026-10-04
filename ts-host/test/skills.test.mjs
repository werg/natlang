import assert from 'node:assert/strict';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { mkdtempSync, mkdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import ts from 'typescript';
import { addedExecutables, authorView, bindSkills, evaluationTicket, isSkillTarget, loadSkills, memorySkillSource,
  parseMarkdownSkill, readSkillDocument, renderScopeDeclarations, renderSkillListing, scopeBindings, SkillPool, SkillSet,
  readSkillFile, skillContextFiles, skillExecutables, ticketMatches, validateEpisode, SKILL_EPISODE_SCHEMA } from '../dist/skills/index.js';
import { directorySkillSource } from '../dist/skills/node.js';
import { isNeuraleseRef } from '../dist/native/neuralese.js';
import { Context, createNatlangRuntime, live, loadNatlang, nodeSourceFiles, skillContextFiles as publicSkillContextFiles } from '../dist/index.js';

const fixture = name => fileURLToPath(new URL(`./fixtures/skills/${name}`, import.meta.url));
const program = () => loadSkills(directorySkillSource(fixture('program')));

test('standard skill folders load unchanged, with the natlang block in either spelling', async () => {
  const { set, diagnostics } = await program();
  assert.deepEqual(diagnostics.filter(d => d.severity === 'error'), []);
  assert.deepEqual(set.names, ['changelog-writer', 'decision-finality', 'exact-file-edits', 'sql-join-keys']);
  const changelog = set.get('changelog-writer');
  assert.equal(changelog.format, 'markdown');
  assert.equal(changelog.frontmatter.license, 'MIT');
  assert.deepEqual(changelog.files, ['references/style.md', 'scripts/collect.sh']);
  assert.deepEqual(changelog.natlang.scope, {});
  assert.match(changelog.body, /^# Changelog writer/);
  assert.deepEqual(set.get('sql-join-keys').natlang.requires, { skills: [], services: ['sql'] }, 'metadata.natlang as a YAML string');
  assert.equal(set.get('sql-join-keys').natlang.scope.maxJoins.value, 3);
  assert.equal(set.get('decision-finality').natlang.tests.length, 2);
  assert.deepEqual(set.get('exact-file-edits').natlang.requires.skills, ['decision-finality']);
  assert.match(set.get('decision-finality').revision, /^sk1_[0-9a-f]{32}$/);
});

test('invalid skills are reported and left out', async () => {
  const { set, diagnostics } = await loadSkills(directorySkillSource(fixture('invalid')));
  assert.equal(set.size, 0);
  const codes = diagnostics.filter(d => d.severity === 'error').map(d => d.code).sort();
  assert.deepEqual(codes, ['skill-description-missing', 'skill-name-folder', 'skill-name-format', 'skill-scope-missing-file', 'skill-type']);
});

test('the opening lists names and descriptions; bodies and files are read on demand', async () => {
  const { set } = await program();
  const listing = renderSkillListing(set);
  assert.match(listing, /read_code\("skills\.<name>"\)/);
  assert.match(listing, /^- decision-finality: Decides whether a note states a current, final decision/m);
  assert.match(listing, /Read instructions: read_code\("skills\.decision-finality"\)/);
  assert.match(listing, /do not need to read every skill/);
  assert.match(listing, /Reading instructions does not execute a procedure/);
  assert.match(listing, /Read another skill when it addresses a distinct need/);
  assert.match(listing, /needed details are no longer available/);
  assert.doesNotMatch(listing, /The last explicit decision wins/, 'bodies are not in the opening');
  assert.equal(renderSkillListing(new SkillSet()), '');
  assert.ok(isSkillTarget('skills.decision-finality'));
  const body = await readSkillDocument(set, 'skills.decision-finality');
  assert.equal(body.kind, 'text');
  assert.match(body.text, /The last explicit decision wins/);
  assert.match(body.text, /- examples\/withdrawn\.md: read_code\("skills\.decision-finality\/examples\/withdrawn\.md"\)/);
  const file = await readSkillDocument(set, 'skills.changelog-writer/references/style.md');
  assert.match(file.text, /Present tense/);
  await assert.rejects(readSkillDocument(set, 'skills.nope'), /no skill named "nope"/);
  await assert.rejects(readSkillDocument(set, 'skills.changelog-writer/../../secret'), /has no file/);
});

test('scope bindings are injected with checked types', async () => {
  const { set } = await program();
  const { bindings, diagnostics } = await scopeBindings(set);
  assert.deepEqual(diagnostics, []);
  assert.deepEqual(bindings.map(b => [b.name, b.skill]), [['finalityRubric', 'decision-finality'], ['maxJoins', 'sql-join-keys']]);
  assert.equal(bindings[0].value.length, 4);
  assert.equal(bindings[1].value, 3);
  const declarations = renderScopeDeclarations(bindings);
  assert.match(declarations, /^declare const finalityRubric: .*\[\];  \/\/ from skill decision-finality: cue phrases/m);
  assert.match(declarations, /^declare const maxJoins: number;  \/\/ from skill sql-join-keys$/m);
  const reserved = await scopeBindings(set, { reserved: ['maxJoins'] });
  assert.deepEqual(reserved.diagnostics.map(d => d.code), ['skill-scope-reserved']);
});

test('scope values that do not fit their type, duplicate names and .nz sources are diagnosed', async () => {
  const skill = (name, scope) => `---\nname: ${name}\ndescription: Fixture skill ${name}.\nnatlang:\n  scope:\n${scope}\n---\nBody.\n`;
  const source = memorySkillSource({
    'skills/a/SKILL.md': skill('a', '    limit:\n      type: number\n      value: "many"'),
    'skills/b/SKILL.md': skill('b', '    shared:\n      type: string\n      value: one'),
    'skills/c/SKILL.md': skill('c', '    shared:\n      type: string\n      value: two'),
    'skills/d/SKILL.md': skill('d', '    plan:\n      type: Neuralese<string>\n      nz: plan.nz#body'),
    'skills/d/plan.nz': new Uint8Array([1, 2, 3]),
  });
  const { set, diagnostics } = await loadSkills(source);
  assert.deepEqual(diagnostics.filter(d => d.severity === 'error'), []);
  const first = await scopeBindings(set);
  assert.deepEqual(first.diagnostics.map(d => d.code).sort(), ['nz-skill-unsupported', 'skill-scope-conflict', 'skill-scope-type']);
  assert.deepEqual(first.bindings.map(b => b.name), ['shared']);
  const seen = [];
  const resolved = await scopeBindings(set, { nz: async (skill, file, name) => { seen.push([skill.name, file, name]); return 'nz1_' + 'a'.repeat(52); } });
  assert.deepEqual(seen, [['d', 'plan.nz', 'body']]);
  const plan = resolved.bindings.find(b => b.name === 'plan');
  assert.ok(plan && isNeuraleseRef(plan.value));
  assert.equal(plan.value.$neuralese.type, 'Neuralese<string>');
});

test('.nz skills go through the metadata hook', async () => {
  const source = memorySkillSource({ 'skills/soft-rubric.nz': new Uint8Array([0]), 'skills/folded/folded.nz': new Uint8Array([1]) });
  const unsupported = await loadSkills(source);
  assert.equal(unsupported.set.size, 0);
  assert.deepEqual(unsupported.diagnostics.map(d => d.code), ['nz-skill-unsupported', 'nz-skill-unsupported']);
  const meta = { 'skills/soft-rubric.nz': { skill: { name: 'soft-rubric', description: 'A soft rubric.', body: 'instructions' } },
    'skills/folded/folded.nz': { skill: { name: 'folded', description: 'A soft skill in a folder.', body: 'body',
      natlang: { exports: { score: 'Neuralese<number>' } } } } };
  const { set, diagnostics } = await loadSkills(source, { nz: { async readMetadata(path) { return meta[path]; } } });
  assert.deepEqual(diagnostics, []);
  assert.deepEqual(set.names, ['folded', 'soft-rubric']);
  const doc = await readSkillDocument(set, 'skills.soft-rubric');
  assert.deepEqual([doc.kind, doc.export], ['neuralese', 'instructions']);
});

test('skill sets are immutable; the pool changes only by new revisions', async () => {
  const { set } = await program();
  const pool = new SkillPool();
  const before = pool.revision;
  pool.publish(set.select(['decision-finality', 'sql-join-keys']));
  const snapshot = pool.snapshot();
  assert.notEqual(pool.revision, before);
  const bound = bindSkills(set.select(['changelog-writer']), snapshot, ['sql-join-keys']);
  assert.deepEqual(bound.names, ['changelog-writer', 'sql-join-keys']);
  pool.retire(['sql-join-keys']);
  assert.ok(bound.has('sql-join-keys'), 'a bound set keeps its skills when the pool changes');
  assert.ok(snapshot.has('sql-join-keys'));
  assert.ok(!pool.snapshot().has('sql-join-keys'));
  assert.equal(pool.history.length, 2);
  assert.throws(() => bindSkills(set.select(['exact-file-edits'])), /exact-file-edits needs decision-finality/);
  assert.throws(() => set.select(['nope']), /unknown skills: nope/);
  const edited = parseMarkdownSkill('skills/sql-join-keys', { 'SKILL.md': '---\nname: sql-join-keys\ndescription: Edited.\n---\nNew body.\n' });
  assert.throws(() => set.union(new SkillSet([edited])), /conflicting revisions/);
  assert.equal(bound.revision, bindSkills(set.select(['changelog-writer']), snapshot, ['sql-join-keys']).revision);
});

test('skill snapshots pin supporting-file bytes and metadata across edits and rebinding', async () => {
  const files = {
    'skills/review/SKILL.md': '---\nname: review\ndescription: Reviews records.\nnatlang:\n  scope:\n    confidence:\n      type: number\n      value: 1\n---\nUse the supplied criteria.\n',
    'skills/review/references/rules.md': 'Rule set revision one.\n',
  };
  const source = { async list() { return Object.keys(files); }, async read(path) {
    if (!(path in files)) throw new Error(`missing ${path}`);
    return files[path];
  } };
  const pool = new SkillPool();
  const first = await loadSkills(source);
  pool.publish(first.set);
  const bound = pool.snapshot();
  const revision = bound.revision;
  files['skills/review/SKILL.md'] = files['skills/review/SKILL.md'].replace('Use the supplied criteria.', 'Prefer specific evidence.');
  files['skills/review/references/rules.md'] = 'Rule set revision two.\n';
  assert.throws(() => { bound.get('review').natlang.scope.confidence.value = 0.5; }, TypeError,
    'nested frontmatter is not mutable through the bound skill');
  assert.match((await readSkillDocument(bound, 'skills.review')).text, /Use the supplied criteria/);
  assert.equal(await readSkillFile(bound, 'review', 'references/rules.md'), 'Rule set revision one.\n');

  const second = await loadSkills(source);
  pool.publish(second.set);
  assert.notEqual(pool.revision, revision);
  assert.equal(await readSkillFile(bound, 'review', 'references/rules.md'), 'Rule set revision one.\n',
    'an existing snapshot keeps its disclosed file');
  assert.equal(await readSkillFile(pool.snapshot(), 'review', 'references/rules.md'), 'Rule set revision two.\n',
    'reloading and publishing makes the edit visible to newly bound calls');

  const input = new Uint8Array([1, 2, 3]);
  const memory = memorySkillSource({ 'skills/binary/data.bin': input });
  input[0] = 9;
  const returned = await memory.read('skills/binary/data.bin');
  assert.deepEqual([...returned], [1, 2, 3]);
  returned[1] = 8;
  assert.deepEqual([...await memory.read('skills/binary/data.bin')], [1, 2, 3]);
});

test('a selected topic skill becomes function-local context data', async () => {
  const text = '---\nname: review\ndescription: Reviews a payment dispute.\n---\nCheck the cited evidence before deciding.\n';
  const { set } = await loadSkills(memorySkillSource({ 'skills/review/SKILL.md': text,
    'skills/review/references/checklist.md': 'Check dates and amounts.\n' }));
  const pool = new SkillPool(); pool.publish(set);
  const files = await skillContextFiles(pool.snapshot().select(['review']));
  assert.equal(publicSkillContextFiles, skillContextFiles, 'the host helper is available from the Node package entry');
  assert.deepEqual(Object.keys(files).sort(), ['skills/review/SKILL.md', 'skills/review/references/checklist.md']);

  const root = mkdtempSync(join(tmpdir(), 'natlang-skill-context-'));
  mkdirSync(join(root, 'triage'), { recursive: true });
  writeFileSync(join(root, 'triage.nl'), '---\nargs:\n  issue: string\nreturns: string\n---\nUse the bound topic instructions.\n');
  const triage = loadNatlang(join(root, 'triage.nl'), root);
  const context = await Context.fromFolder(join(root, 'triage'), nodeSourceFiles(root));
  const seen = [];
  const runtime = createNatlangRuntime({ model: async request => {
    seen.push(request.messages);
    if (seen.length === 1) return { calls: [['read_code', { name: 'skills.review' }]] };
    return { calls: [['return_result', { status: 'success', value: 'reviewed' }]] };
  } });
  assert.equal(await runtime.run(() => triage.in(context.with(files))('payment dispute')), 'reviewed');
  assert.match(JSON.stringify(seen[0]), /Reviews a payment dispute/);
  assert.match(JSON.stringify(seen[1]), /Check the cited evidence/);
});

test('runtime openings report omitted skill bindings instead of hiding them', async () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-skill-diagnostic-'));
  mkdirSync(join(root, 'triage', 'skills', 'review'), { recursive: true });
  writeFileSync(join(root, 'triage.nl'), '---\nargs:\n  issue: string\nreturns: string\n---\nUse the bound topic instructions.\n');
  writeFileSync(join(root, 'triage', 'skills', 'review', 'SKILL.md'),
    '---\nname: review\ndescription: Reviews an issue.\nnatlang:\n  scope:\n    confidence: { type: number, value: high }\n---\nCheck the evidence.\n');
  const triage = loadNatlang(join(root, 'triage.nl'), root);
  const seen = [];
  const runtime = createNatlangRuntime({ model: async request => {
    seen.push(request.messages);
    return { calls: [['return_result', { status: 'success', value: 'reviewed' }]] };
  } });
  assert.equal(await runtime.run(() => triage('payment dispute')), 'reviewed');
  const opening = JSON.stringify(seen[0]);
  assert.match(opening, /Skill diagnostics/);
  assert.match(opening, /confidence \[skill-scope-type\]/);
});

test('JSON skill assets retain file semantics when a folder context is rebound', async () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-skill-json-context-'));
  mkdirSync(join(root, 'triage', 'skills', 'review'), { recursive: true });
  writeFileSync(join(root, 'triage.nl'), '---\nargs:\n  issue: string\nreturns: string\n---\nUse the assessment threshold.\n');
  writeFileSync(join(root, 'triage', 'skills', 'review', 'SKILL.md'),
    '---\nname: review\ndescription: Apply the review threshold.\nnatlang:\n  scope:\n    threshold: { type: number, file: criteria.json, pointer: /threshold }\n---\nCheck the evidence.\n');
  writeFileSync(join(root, 'triage', 'skills', 'review', 'criteria.json'), '{"threshold":0.82}\n');
  const triage = loadNatlang(join(root, 'triage.nl'), root);
  const context = await Context.fromFolder(join(root, 'triage'), nodeSourceFiles(root));
  const seen = [];
  const runtime = createNatlangRuntime({ model: async request => {
    seen.push(request.messages);
    return { calls: [['return_result', { status: 'success', value: 'reviewed' }]] };
  } });
  assert.equal(await runtime.run(() => triage.in(context)('payment dispute')), 'reviewed');
  const opening = JSON.stringify(seen[0]);
  assert.match(opening, /const threshold: number = 0\.82;  \/\/ from skill review/);
  assert.doesNotMatch(opening, /threshold \[skill-scope-load\]/);
});

test('contexts snapshot ordinary data and detach binary reads while preserving runtime handles', () => {
  const input = { rubric: { threshold: 0.8 }, examples: [{ result: 'review' }] };
  const bytes = new Uint8Array([1, 2, 3]);
  const liveValue = live(() => 'dynamic');
  const reference = { $neuralese: { type: 'Neuralese<string>', id: 'nz1_aaaaaaaaaaaaaaaaaaaa' } };
  const context = Context.of({}, { input, bytes, liveValue, reference });
  const id = context.id;
  input.rubric.threshold = 0.1;
  input.examples[0].result = 'changed';
  bytes[0] = 9;
  assert.equal(context.id, id);
  assert.equal(context.data.input.rubric.threshold, 0.8);
  assert.equal(context.data.input.examples[0].result, 'review');
  assert.deepEqual([...context.data.bytes], [1, 2, 3]);
  assert.equal(context.data.liveValue, liveValue, 'explicit live captures retain their accessor identity');
  assert.deepEqual(context.data.reference, reference, 'Neuralese references retain their type and block identity');
  assert.notEqual(context.data.reference, reference, 'the stored handle is detached from a mutable caller object');
  const exposed = context.data;
  exposed.bytes[1] = 7;
  assert.deepEqual([...context.data.bytes], [1, 2, 3], 'mutating a returned typed array cannot alter the stored context');
  assert.throws(() => { exposed.input.rubric.threshold = 0.4; }, TypeError, 'plain data snapshots are deeply frozen');
});

test('skill helpers are readable and callable under a hyphenated topic name', async () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-skill-helper-'));
  mkdirSync(join(root, 'triage', 'skills', 'exact-bookkeeping', 'helpers'), { recursive: true });
  mkdirSync(join(root, 'triage', 'skills', 'exact-bookkeeping', 'references'), { recursive: true });
  writeFileSync(join(root, 'triage.nl'), '---\nargs:\n  left: number\n  right: number\nreturns: number\n---\nUse exact-bookkeeping helpers for arithmetic.\n');
  writeFileSync(join(root, 'triage', 'skills', 'exact-bookkeeping', 'SKILL.md'),
    '---\nname: exact-bookkeeping\ndescription: Exact arithmetic checks.\n---\nUse the sum helper for addition.\n');
  writeFileSync(join(root, 'triage', 'skills', 'exact-bookkeeping', 'references', 'rules.md'),
    'Use integer minor units for exact financial arithmetic.\n');
  writeFileSync(join(root, 'triage', 'skills', 'exact-bookkeeping', 'helpers', 'calc.ts'),
    'export function sum(left: number, right: number): number { return left + right; }\n');
  const triage = loadNatlang(join(root, 'triage.nl'), root);
  const seen = [], traces = [];
  const runtime = createNatlangRuntime({ model: async request => {
    seen.push(request.messages);
    if (seen.length === 1) return { calls: [['read_code', { name: 'skills.exact-bookkeeping' }]] };
    if (seen.length === 2) return { calls: [['read_code', { name: 'skills.exact-bookkeeping/references/rules.md' }]] };
    if (seen.length === 3) return { calls: [['read_code', { name: 'skills.exact-bookkeeping.helpers.calc' }]] };
    return { calls: [['eval', { code: 'return skills["exact-bookkeeping"].helpers.calc.sum(left, right);', finish: true }]] };
  }, trace: trace => traces.push(trace) });
  assert.equal(await runtime.run(() => triage(2, 3)), 5);
  const firstEval = seen[0].flatMap(message => message.tool_calls ?? [])
    .find(call => call.function.name === 'eval');
  const scopeCode = JSON.parse(firstEval.function.arguments).code;
  assert.match(scopeCode, /declare const skills: \{[\s\S]*"exact-bookkeeping": \{/);
  assert.doesNotMatch(scopeCode, /namespace exact-bookkeeping/);
  const parsed = ts.createSourceFile('scope.ts', scopeCode, ts.ScriptTarget.ES2022, true, ts.ScriptKind.TS);
  assert.equal(parsed.parseDiagnostics.length, 0, 'the shown helper call tree is valid TypeScript syntax');
  assert.match(JSON.stringify(seen[1]), /Use the sum helper for addition/);
  assert.match(JSON.stringify(seen[2]), /integer minor units/);
  assert.match(JSON.stringify(seen[3]), /export function sum/);
  const events = traces.flatMap(trace => trace.events).filter(event => event.kind === 'skill_use');
  assert.deepEqual(events.map(({phase, path, helper_export}) => ({phase, path, helper_export})), [
    {phase:'offered', path:undefined, helper_export:undefined},
    {phase:'body_read', path:'SKILL.md', helper_export:undefined},
    {phase:'support_file_read', path:'references/rules.md', helper_export:undefined},
    {phase:'helper_invoked', path:'helpers/calc.ts', helper_export:'sum'},
  ]);
  assert.ok(events.every(event => event.skill_name === 'exact-bookkeeping' && typeof event.skill_revision === 'string' && event.skill_revision.length > 0));
  assert.equal(new Set(events.map(event => event.skill_revision)).size, 1);
  assert.ok(events.every(event => !('content' in event) && !('arguments' in event) && !('result' in event)));
});

test('helpers are executable nodes; adding one is detected', async () => {
  const { set } = await program();
  assert.deepEqual(skillExecutables(set.get('exact-file-edits')), ['skills/exact-file-edits/helpers/applyEdits.ts']);
  assert.deepEqual(skillExecutables(set.get('decision-finality')), []);
  const base = set.select(['decision-finality']);
  assert.deepEqual(addedExecutables(base, set.select(['decision-finality', 'sql-join-keys'])), [], 'adding data skills adds no nodes');
  assert.deepEqual(addedExecutables(base, set.select(['decision-finality', 'exact-file-edits'])), ['skills/exact-file-edits/helpers/applyEdits.ts']);
});

const kase = (id, group, expected) => ({ id, group, args: [], folder: { 'records.json': `[{"id":"${id}"}]`, 'README.md': 'Preserve this unrelated file.\n' },
  expectedFiles: { 'README.md': 'Preserve this unrelated file.\n', 'report.json': JSON.stringify(expected) } });
const episode = () => ({
  version: SKILL_EPISODE_SCHEMA, id: 'ep-1', family: 'active-urgency', split: 'train', source_groups: ['fam:active-urgency'], license: 'project-generated',
  target: { kind: 'improvement-case', files: { 'solve.nl': 'Run the workflow.' }, entry: 'solve.nl', exportName: 'default', source: { schema: 'natlang.improvement-case/1', id: 'case-1' } },
  library: { kind: 'empty', skills: {} },
  support: { cases: [kase('s1', 'g:support:1', { selected: ['s1-x'], total: 101 })] },
  query: { cases: [kase('q1', 'g:query:1', { selected: ['q1-secret-id'], total: 977 })] },
  transfer: { family: 'final-renewal', target: { kind: 'improvement-case', files: { 'solve.nl': 'Other.' }, entry: 'solve.nl', source: { schema: 'natlang.improvement-case/1', id: 'case-2' } },
    cases: [kase('t1', 'g:transfer:1', { selected: ['t1-other-secret'], total: 5 })] },
  operations: ['create', 'revise', 'test'], limits: { maxSteps: 8 }, provenance: { generator: 'fixture' },
});

test('a well-formed episode validates, and the author view seals query and transfer', () => {
  const ep = episode();
  assert.deepEqual(validateEpisode(ep), []);
  const view = authorView(ep);
  assert.equal(view.query, undefined);
  assert.equal(view.transfer, undefined);
  assert.deepEqual([view.evaluation.query_cases, view.evaluation.transfer_cases], [1, 1]);
  assert.doesNotMatch(JSON.stringify(view), /secret/);
  assert.ok(ticketMatches(ep, view.evaluation));
  const changed = episode(); changed.query.cases[0].expectedFiles['report.json'] = '{"selected":[],"total":0}';
  assert.ok(!ticketMatches(changed, view.evaluation), 'the ticket commits to the sealed content');
  assert.equal(evaluationTicket(ep).id, view.evaluation.id);
});

test('leakage rules: shared groups, visible answers and skill provenance', () => {
  const shared = episode(); shared.query.cases[0].group = 'g:support:1';
  assert.deepEqual(validateEpisode(shared).map(d => d.code), ['leak-group']);
  const visible = episode(); visible.support.cases[0].folder['hint.txt'] = 'answer: {"selected":["q1-secret-id"],"total":977}';
  assert.deepEqual(validateEpisode(visible).map(d => d.code), ['leak-answer']);
  const inSkill = episode();
  inSkill.library = { kind: 'existing', skills: { leaky: { 'SKILL.md': '---\nname: leaky\ndescription: d\n---\nThe ids are t1-other-secret.\n' } } };
  assert.ok(validateEpisode(inSkill).some(d => d.code === 'leak-answer' && d.path === 'transfer.t1'));
  const provenance = episode();
  provenance.library = { kind: 'existing', skills: { made: { 'SKILL.md': '---\nname: made\ndescription: d\nnatlang:\n  provenance:\n    support_groups: ["g:query:1"]\n---\nBody.\n' } } };
  assert.deepEqual(validateEpisode(provenance).map(d => d.code), ['leak-skill-provenance']);
  const broken = episode(); broken.operations = ['create', 'teleport']; broken.limits = { maxSteps: 0 }; broken.library = { kind: 'corrupted', skills: {} };
  assert.deepEqual(validateEpisode(broken).map(d => d.code).sort(), ['episode-library', 'episode-limits', 'episode-operations']);
  const sameFamily = episode(); sameFamily.transfer.family = 'active-urgency';
  assert.deepEqual(validateEpisode(sameFamily).map(d => d.code), ['episode-transfer']);
});

test('privilege rule: teacher-only knowledge may not appear in the student view', () => {
  const clean = episode(); clean.support.cases[0].privileged = { worked: 'Select records whose deadline is within 48 hours.' };
  assert.deepEqual(validateEpisode(clean), []);
  const inInputs = episode(); inInputs.support.cases[0].privileged = 'Select records whose deadline is within 48 hours.';
  inInputs.support.cases[0].folder['notes.txt'] = 'Hint: Select records whose deadline is within 48 hours.';
  assert.deepEqual(validateEpisode(inInputs).map(d => d.code), ['leak-privileged']);
  const inTarget = episode(); inTarget.query.cases[0].privileged = { doc: 'Run the workflow.' };
  assert.deepEqual(validateEpisode(inTarget).map(d => [d.code, d.path]), [['leak-privileged', 'query.q1.privileged']]);
});

test('episode builders: slate episodes are valid and repair episodes carry labelled defects', async () => {
  const { slateEpisodes, relatedFamily } = await import('../scripts/skills/build-episodes.mjs');
  const { corruptSkill, loadSeeds, repairEpisodes } = await import('../scripts/skills/build-repair-episodes.mjs');
  const { readFileSync } = await import('node:fs');
  const slate = readFileSync(fileURLToPath(new URL('../../data/teacher/self-improvement/task-slate-v1/cases.jsonl', import.meta.url)), 'utf8');
  const tasks = slate.trim().split('\n').map(line => JSON.parse(line)).filter(task => ['active-urgency', 'explicit-consent'].includes(task.family));
  const episodes = slateEpisodes(tasks);
  assert.equal(episodes.length, 2, 'only the family with an available related-family transfer is emitted for both splits');
  for (const ep of episodes) assert.deepEqual(validateEpisode(ep), [], ep.id);
  assert.equal(relatedFamily('active-urgency'), 'explicit-consent');
  assert.ok(episodes.every(ep => !JSON.stringify(ep).includes('"reference"')), 'gold references stay out of episodes');
  const first = episodes.find(ep => ep.family === 'active-urgency' && ep.split === 'train');
  assert.equal(first.transfer.family, 'explicit-consent');
  const { seeds } = await loadSeeds(fixture('program'));
  const { episodes: repairs } = repairEpisodes(first, seeds);
  assert.deepEqual(repairs.map(ep => ep.library.defect.kind), ['missing', 'irrelevant', 'incorrect']);
  for (const ep of repairs) assert.deepEqual(validateEpisode(ep), [], ep.id);
  const repairView = authorView(repairs[0]);
  assert.equal(repairView.library.kind, 'existing');
  assert.ok(!('defect' in repairView.library), 'repair kind and details are host-only');
  assert.ok(!('provenance' in repairView) && !('source_groups' in repairView), 'source lineage is host-only');
  assert.deepEqual(repairView.target.source, { schema: 'redacted', id: 'redacted' });
  assert.ok(!('decision-finality' in repairs[0].library.skills));
  assert.ok('decision-finality' in repairs[1].library.skills && Object.keys(repairs[1].library.skills).length === 2);
  assert.notDeepEqual(repairs[2].library.skills['decision-finality'], seeds.find(s => s.name === 'decision-finality').files);
  const steps = corruptSkill({ 'SKILL.md': '---\nname: x\ndescription: d\n---\n1. first\n2. second\n' }, 'seed');
  assert.match(steps.files['SKILL.md'], /1\. second\n2\. first/);
  assert.equal(corruptSkill({ 'SKILL.md': 'no steps' }, 'seed'), undefined);
});
