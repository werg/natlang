import test from 'node:test';
import assert from 'node:assert/strict';
import {episodeScoring} from '../dist/skills/scoring.js';
import {createCrosswordEpisodes} from '../scripts/skills/build-crossword-episodes.mjs';
import {scoreCspProgress,cspProgressBound} from '../dist/skills/csp-objective.js';
test('crossword episode declares pinned collection scorer and accepts serialized valid fills',()=>{
 const e=createCrosswordEpisodes().episodes[0],row=e.support.cases[0];
 assert.throws(()=>episodeScoring(e.provenance.metric,{pins:{}}),/pin required/);
 const scorer=episodeScoring(e.provenance.metric,{pins:{'skills/crossword-objective.js':'fixture'}});
 assert.equal(scorer.score(row,{value:JSON.stringify({fills:row.expected.accepted[0]})}).quality,1);
});
test('CSP collection accepts serialized assignment and bound remains host controlled',()=>{
 const instance={schema:'natlang.finite-csp/1',template:'fixture',narrative:'Assign a to one.',variables:[{id:'a',domain:[1,2]}],constraints:[{kind:'equal',variable:'a',value:1}]};
 const row={args:[JSON.stringify(instance)],expected:cspProgressBound(instance)};
 const scorer=episodeScoring({schema:'natlang.skill-csp/1',kind:'csp-progress'},{pins:{'skills/csp-objective.js':'fixture'}});
 assert.equal(scorer.score(row,{value:'{"assignment":{"a":1}}'}).quality,1);
 assert.equal(scoreCspProgress(instance,'broken',row.expected).quality,0);
});
