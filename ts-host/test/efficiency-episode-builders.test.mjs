import test from 'node:test';
import assert from 'node:assert/strict';
import {createEfficiencyEpisodes,EFFICIENCY_FAMILIES} from '../scripts/skills/build-efficiency-episodes.mjs';
import {createCodeGolfEpisodes} from '../scripts/skills/build-code-golf-episodes.mjs';
import {validateEpisode,authorView} from '../dist/skills/episode.js';
import {exactObjectiveBounds} from '../dist/skills/objective.js';
import {episodeScoring} from '../dist/skills/scoring.js';
const pins={'skills/objective.js':'wrapper','skills/extended-objective.js':'extended','skills/efficiency-objective.js':'efficiency','skills/code-objective.js':'code','skills/graded.js':'sandbox'};

test('expanded optimization packets validate, pin exact references, and expose metadata-only variants',()=>{
 const {episodes}=createEfficiencyEpisodes({replicas:1,variants:['empty','metadata']});
 assert.equal(episodes.length,EFFICIENCY_FAMILIES.length*2);
 for(const e of episodes){
  assert.deepEqual(validateEpisode(e),[]);
  const support=new Set(e.support.cases.map(c=>c.group));assert.ok(e.query.cases.every(c=>!support.has(c.group)));
  const scorer=episodeScoring(e.provenance.metric,{pins});assert.ok(scorer.identity.includes('extended')||scorer.identity.includes('efficiency'));
  for(const row of [...e.support.cases,...e.query.cases])assert.deepEqual(exactObjectiveBounds(e.provenance.metric.kind,row.args[0]),row.expected);
  if(e.provenance.library_variant==='metadata')assert.equal(e.provenance.selection_design,'metadata-tuning');
 }
});
test('code-golf queries hold out program templates and do not enter the author input',()=>{
 const episodes=createCodeGolfEpisodes({replicas:1,variants:['empty','metadata']});assert.equal(episodes.length,12);
 for(const e of episodes){
  assert.deepEqual(validateEpisode(e),[]);const support=new Set(e.support.cases.map(c=>c.group));
  assert.ok(e.query.cases.every(c=>!support.has(c.group)));
  const visible=JSON.stringify(authorView(e));
  for(const row of e.query.cases)assert.ok(!visible.includes(row.expected.id));
  assert.ok(episodeScoring(e.provenance.metric,{pins}).identity.includes('code:sandbox'));
 }
});
test('new scorer dependencies are mandatory identity pins',()=>{
 assert.throws(()=>episodeScoring({schema:'natlang.skill-objective/1',kind:'boolean-dnf'},{pins:{'skills/objective.js':'wrapper'}}),/pin required/);
 assert.throws(()=>episodeScoring({schema:'natlang.skill-code-objective/1',kind:'python-source-bytes'},{pins:{}}),/pins required/);
});
