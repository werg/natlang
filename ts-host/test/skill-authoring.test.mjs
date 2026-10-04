import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { skillRoot, skillEpisodeFiles, supportSearchCases, authorSkillEpisode, checkTransferTarget } from '../dist/improvement/skill-authoring.js';
import { scriptedModel } from './support/natlang.mjs';
import { Folder, SourceEvaluator } from '../dist/index.js';
import { UsageGateway } from '../dist/evaluation/usage.js';

function episode() {
  return { version: 'natlang.skill-episode/1', id: 'skill-test', family: 'increment', split: 'train',
    license: 'project-generated', source_groups: ['support-a', 'support-b', 'query-private'],
    target: { kind: 'improvement-case', entry: 'solve.nl', source: { schema: 'test', id: 'increment' },
      files: { 'solve.nl': '---\nargs:\n  value: number\nreturns: number\n---\nIncrement the value, using bound skills when relevant.\n' } },
    library: { kind: 'empty', skills: {} },
    support: { cases: [{ id: 'support-a', group: 'support-a', args: [1], expected: 2 },
      { id: 'support-b', group: 'support-b', args: [2], expected: 3 }] },
    query: { cases: [{ id: 'query-private', group: 'query-private', args: [17], expected: 18 }] },
    operations: ['create', 'revise'], limits: { maxSteps: 2 }, provenance: {} };
}

test('skills bind in the callable companion, with pinned target and disjoint support search splits', () => {
  assert.equal(skillRoot('nested/solve.nl'), 'nested/solve/skills');
  const row = episode();
  const support = supportSearchCases(row);
  assert.deepEqual(support.map(x => x.split), ['train', 'validation']);
  assert.ok(!JSON.stringify(support).includes('query-private'));
  row.library = { kind: 'existing', skills: { increment: { 'SKILL.md': '---\nname: increment\ndescription: Increment.\n---\nAdd one.\n' } } };
  assert.ok(skillEpisodeFiles(row)['solve/skills/increment/SKILL.md']);
  row.library.skills.increment['../bad.txt'] = 'bad';
  assert.throws(() => skillEpisodeFiles(row), /unsafe skill file/);
});

test('support selection cannot reuse a single source group across train/validation', () => {
  const row = episode(); row.support.cases[1].group = 'support-a';
  assert.throws(() => supportSearchCases(row), /two disjoint/);
});

test('continuous heldout quality is independently scored and paired without a binary threshold', async () => {
  const before = Folder.fromFiles({ 'main.ts': 'export function solve(value: number): number {return value;}' }).snapshot();
  const after = Folder.fromFiles({ 'main.ts': 'export function solve(value: number): number {return value+1;}' }).snapshot();
  const evaluator = new SourceEvaluator({ entry: 'main.ts', exportName: 'solve', programId: 'partial-quality' },
    [{ id: 'held', group: 'held', split: 'test', args: [1], expected: 4 }],
    () => { throw Error('No inference for exact code'); },
    new UsageGateway({ maxModelCalls: 1, maxRollouts: 4, maxProposals: 0 }),
    { executorId: 'exact', scoring: { identity: 'computed-output/1',
      score: (row, result) => ({ quality: result.value / row.expected, gates: { feasible: result.value >= 0 } }) } });
  const paired = await evaluator.confirmQuality(before, after, 'fixed-partial-comparison');
  assert.equal(paired.effect, 0.25);
  assert.equal(paired.baseline.quality, 0.25);
  assert.equal(paired.selected.quality, 0.5);
  assert.equal(paired.wins, 1);
  await assert.rejects(() => evaluator.confirmQuality(before, after, 'adaptive-retry'), /repeated/);
});

test('authored skill edit improves a frozen target and query never enters author requests', async () => {
  const row = episode();
  const skill = '---\nname: task-procedure\ndescription: Increment numbers accurately.\n---\nReturn value plus one.\n';
  const model = scriptedModel(opening => opening.includes('Choose one coherent, evidenced hypothesis') ?
    `await folder.file("solve/skills/task-procedure/SKILL.md").writeText(${JSON.stringify(skill)}); return await bookkeeping.finish(folder,"teach increment",["target contract"]);` :
    'return await lifecycle.step(folder,evaluator,rewriteProgram,state,policy)');
  const author = async (request, signal) => {
    const turn = await model.driver(request, signal);
    for (const [name, args] of turn.calls ?? []) if (name === 'eval') args.finish = true;
    return turn;
  };
  const target = scriptedModel(opening => opening.includes('task-procedure') ? 'return value + 1' : 'return value');
  const executor = async (request, signal) => {
    const turn = await target.driver(request, signal);
    for (const [name, args] of turn.calls ?? []) if (name === 'eval') args.finish = true;
    return turn;
  };
  const result = await authorSkillEpisode({ episode: row, author, executor, executorId: 'scripted-frozen',
    directory: mkdtempSync(join(tmpdir(), 'natlang-skill-authoring-')), maxExperiments: 1,
    searchBudget: { maxModelCalls: 60, maxRollouts: 30, maxProposals: 3 },
    evaluationBudget: { maxModelCalls: 20, maxRollouts: 10, maxProposals: 0 } });
  assert.equal(result.disposition, 'evaluated', result.search.error);
  assert.equal(result.query.effect, 1);
  assert.equal(result.positive, true);
  assert.ok(model.openings.every(text => !text.includes('query-private')));
  assert.equal(result.selectedFiles['solve.nl'], row.target.files['solve.nl']);
  assert.equal(result.query.baseline.passed, 0);
  assert.equal(result.query.selected.passed, 1);
});

test('description-only tuning can improve discovery without changing target code or skill body', async () => {
  const row = episode();
  const body = '\nReturn value plus one.\n';
  const oldSkill = '---\nname: task-procedure\ndescription: Use for unrelated text formatting.\n---' + body;
  const newSkill = '---\nname: task-procedure\ndescription: Use when adding one to a number; not for text formatting.\n---' + body;
  row.provenance = {...row.provenance, selection_design:'metadata-tuning'};
  row.library = { kind: 'existing', skills: { 'task-procedure': { 'SKILL.md': oldSkill } } };
  const model = scriptedModel(opening => opening.includes('Choose one coherent, evidenced hypothesis') ?
    `await folder.file("solve/skills/task-procedure/SKILL.md").writeText(${JSON.stringify(newSkill)}); return await bookkeeping.finish(folder,"clarify applicability",["support discovery failure"]);` :
    'return await lifecycle.step(folder,evaluator,rewriteProgram,state,policy)');
  const author = async (request, signal) => {
    const turn = await model.driver(request, signal);
    for (const [name, args] of turn.calls ?? []) if (name === 'eval') args.finish = true;
    return turn;
  };
  const target = scriptedModel(opening => opening.includes('Use when adding one') ? 'return value + 1' : 'return value');
  const executor = async (request, signal) => {
    const turn = await target.driver(request, signal);
    for (const [name, args] of turn.calls ?? []) if (name === 'eval') args.finish = true;
    return turn;
  };
  const result = await authorSkillEpisode({ episode: row, author, executor, executorId: 'description-discovery-fixture',
    directory: mkdtempSync(join(tmpdir(), 'natlang-description-authoring-')), maxExperiments: 1, maxAblations: 2,
    searchBudget: { maxModelCalls: 60, maxRollouts: 30, maxProposals: 3 },
    evaluationBudget: { maxModelCalls: 40, maxRollouts: 20, maxProposals: 0 } });
  assert.equal(result.positive, true, result.search.error);
  assert.equal(result.query.effect, 1);
  assert.equal(result.selectedFiles['solve.nl'], row.target.files['solve.nl']);
  assert.equal(result.selectedFiles['solve/skills/task-procedure/SKILL.md'], newSkill);
  assert.ok(result.ablations.some(row => row.kind === 'baseline_description' && row.quality_effect_on_removal_or_restore === -1));
  assert.ok(result.ablations.every(row => row.paired));
  assert.ok(model.openings.some(text => text.includes('A description-only improvement is valid')));
  assert.ok(model.openings.every(text => !text.includes('query-private')));
});

test('a transfer target with its own skills is rejected before any search', () => {
  const row = episode();
  row.transfer = { target: { ...row.target, files: { ...row.target.files, 'solve/skills/x/SKILL.md': '---\nname: x\ndescription: X.\n---\n' } },
    cases: [{ id: 't', group: 't', args: [3], expected: 4 }] };
  assert.throws(() => checkTransferTarget(row), /already contains skills/);
  row.transfer.target.files = { ...row.target.files };
  checkTransferTarget(row);
});


test('retained baseline skips sealed query, transfer and ablation inference', async()=>{
 const row=episode();row.transfer={family:'related',target:structuredClone(row.target),cases:[{id:'transfer-private',group:'transfer-private',args:[99],expected:100}]};
 row.source_groups.push('transfer-private');
 const skill='---\nname: task-procedure\ndescription: Increment numbers.\n---\nAdd one.\n';
 const authorModel=scriptedModel(opening=>opening.includes('Choose one coherent, evidenced hypothesis')?
  `await folder.file("solve/skills/task-procedure/SKILL.md").writeText(${JSON.stringify(skill)}); return await bookkeeping.finish(folder,"redundant hint",["target contract"]);`:
  'return await lifecycle.step(folder,evaluator,rewriteProgram,state,policy)');
 const finish=driver=>async(request,signal)=>{const turn=await driver(request,signal);for(const [name,args] of turn.calls??[])if(name==='eval')args.finish=true;return turn;};
 const target=scriptedModel(()=>'return value+1');const scored=[];
 const result=await authorSkillEpisode({episode:row,author:finish(authorModel.driver),executor:finish(target.driver),executorId:'already-correct',
  directory:mkdtempSync(join(tmpdir(),'natlang-no-promotion-')),maxExperiments:1,maxAblations:2,
  scoring:{identity:'fixture-exact',score:(item,output)=>{scored.push(item.id);return {quality:output.value===item.expected?1:0,gates:{completed:!output.error}};}},
  searchBudget:{maxModelCalls:60,maxRollouts:30,maxProposals:3},evaluationBudget:{maxModelCalls:20,maxRollouts:10,maxProposals:0}});
 assert.equal(result.disposition,'not-promoted',result.search.error);assert.equal(result.positive,false);
 assert.equal(result.query,null);assert.equal(result.transfer,null);assert.deepEqual(result.ablations,[]);
 assert.ok(scored.length>0);assert.ok(scored.every(id=>id.startsWith('support-')));
});
