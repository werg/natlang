import test from 'node:test';
import assert from 'node:assert/strict';
import { arenaEpisodeExecutions, arenaTicket, ARENA_CODE_FILES } from '../dist/self-play/evaluation.js';
import { semanticGames, semanticScenarioCases } from '../dist/self-play/semantic-games.js';
import { GAME_POLICY_SOURCE } from '../dist/self-play/policy.js';
import { Folder } from '../dist/native/scoped-fs.js';
import { UsageGateway } from '../dist/evaluation/usage.js';
import { replayMatch } from '../dist/self-play/arena.js';
const pins = Object.fromEntries(ARENA_CODE_FILES.map(file => [file, 'fixture-pin']));
function fixture() {
  const game = semanticGames.find(g => g.id === 'evidence-bluff');
  const scenario = semanticScenarioCases().find(c => c.family === game.id).scenario;
  const files = {'play.nl': GAME_POLICY_SOURCE};
  const config = {game:game.id,scenario,seed:4,seat:'reviewer',opponents:{advocate:{files,entry:'play.nl',model:'fixture'}}};
  const row = {id:'support',group:'support',args:[arenaTicket('support',config)],expected:1};
  const profile = {schema:'natlang.adversarial-arena/1',maxDecisions:8,games:{[game.id]:{revision:game.revision}},cases:{support:config}};
  return {game, files, row, episode:{target:{entry:'play.nl'},support:{cases:[row]},query:{cases:[]},provenance:{arena:profile}}};
}
const gateway = () => new UsageGateway({maxModelCalls:20,maxRollouts:5,maxProposals:0});
const driver = async () => ({calls:[['eval',{code:'const view = JSON.parse(observation); return JSON.stringify(view.legalActions[0]);',finish:true}]]});
test('arena executes isolated Natlang players and accounts for every seat; feedback excludes host truth', async () => {
  const {game,files,row,episode} = fixture(), records = [], ledger = gateway();
  const executions = await arenaEpisodeExecutions(episode,driver,{pins,executorId:'fixture',onMatch:event=>records.push(event)});
  const result = await executions.executeCase(Folder.fromFiles(files).snapshot(),row,0,ledger);
  assert.equal(records.length,1); assert.equal(records[0].match.disposition,'completed');
  assert.equal(result.modelCalls,2); assert.equal(ledger.ledger.usage.modelCalls,2);
  assert.equal(result.value.decisions.length,1);
  assert.equal(result.value.decisions[0].view.seat,'reviewer');
  assert.equal('privateVerdict' in result.value.decisions[0].view.observation,false);
  assert.ok(!JSON.stringify(result.value).includes(episode.provenance.arena.cases.support.scenario.rationale));
  assert.deepEqual(await replayMatch(game,records[0].match),records[0].match);
});
test('runtime pins, exact ticket table and opponent model roster fail before inference', async () => {
  const {episode}=fixture(); let calls=0;
  const provider=async()=>{calls++;throw Error('must not call');};
  await assert.rejects(arenaEpisodeExecutions(episode,provider,{pins:{},executorId:'fixture'}),/code pin/);
  for(const mutate of [e=>{e.support.cases[0].args=['forged'];},
    e=>{e.provenance.arena.cases.extra=e.provenance.arena.cases.support;},
    e=>{e.provenance.arena.cases.support.opponents.advocate.model='other';}]) {
    const bad=structuredClone(episode);mutate(bad);
    await assert.rejects(arenaEpisodeExecutions(bad,provider,{pins,executorId:'fixture'}),/arena case|case table/);
  }
  assert.equal(calls,0);
});
test('opponent failure is an infrastructure/fixture failure and not candidate zero reward', async () => {
  const {files,row,episode}=fixture();
  const execution=await arenaEpisodeExecutions(episode,async()=>{throw Error('provider unavailable');},{pins,executorId:'fixture'});
  await assert.rejects(execution.executeCase(Folder.fromFiles(files).snapshot(),row,0,gateway()),/not a scored candidate loss/);
});

test('message-free bargaining observations survive persisted match replay', async () => {
  const game=semanticGames.find(g=>g.id==='meaning-bargain');
  const scenario=semanticScenarioCases().find(row=>row.family===game.id).scenario;
  const policies=Object.fromEntries(game.seats.map(seat=>[seat,{id:seat,decide:async view=>
    view.observation.currentOffer?{type:'accept'}:view.legalActions[0]}]));
  const {playMatch}=await import('../dist/self-play/arena.js');
  const match=await playMatch({game,scenario,policies,seed:17,maxDecisions:8});
  assert.equal(match.disposition,'completed',match.error);
  assert.deepEqual(await replayMatch(game,JSON.parse(JSON.stringify(match))),match);
});
