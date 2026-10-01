/** Deterministic authoring fixtures; these are runtime-verified cases, not teacher transcripts. */
import {mkdir, writeFile} from 'node:fs/promises';
import {join} from 'node:path';
import {createNatlangRuntime, iterateOn} from '../../dist/runtime/node.js';
const output = process.argv[2];
if (!output) throw Error('usage: build-measure-curriculum.mjs OUTPUT_DIR');
const definitions = [
  {id:'remaining-work', initial:3, delta:-1, goal:0, judgment:'continue', expected:'done', instruction:'Finish three experiments using remaining work as the measure.'},
  {id:'quality-plateau', initial:12, delta:-1, goal:0, judgment:'continue', expected:'done', instruction:'Quality stays flat during exploration. Decrease remaining experiments, rather than using quality as the termination measure.'},
  {id:'stalled-measure', initial:3, delta:0, goal:0, judgment:'continue', expected:'IterationDivergedError', instruction:'Reject a step that fails to consume its declared remaining work; preserve the previous checkpoint.'},
  {id:'replenished-measure', initial:3, delta:1, goal:0, judgment:'continue', expected:'IterationDivergedError', instruction:'Reject replenishing remaining work, even when the semantic judge would continue.'},
  {id:'fractional-measure', initial:1.5, delta:-1, goal:0, judgment:'continue', expected:'IterationStepError', instruction:'Require a nonnegative safe integer work measure.'},
  {id:'semantic-early-stop', initial:100, delta:-1, goal:0, judgment:'divergent', expected:'IterationDivergedError', instruction:'Stop an unproductive search during semantic review despite remaining mechanical capacity.'},
  {id:'continue-at-exhaustion', initial:12, delta:-1, goal:-1, judgment:'continue', expected:'IterationLimitError', instruction:'A semantic continue verdict cannot extend exhausted work. Report exhaustion if the goal remains unmet.'},
  {id:'already-done', initial:0, delta:-1, goal:0, judgment:'continue', expected:'done', instruction:'Return the initial state without taking a step when its goal is already satisfied.'},
];
const rows=[];
for (const definition of definitions) {
  const {id,initial,delta,goal,judgment,expected,instruction}=definition;
  const events=[];
  const runtime=createNatlangRuntime();
  let outcome;
  try {
    const value=await runtime.run(()=>iterateOn(state=>({...state,remaining:state.remaining+delta}),{remaining:initial,quality:0.5})
      .withMeasure(state=>state.remaining)
      .checkProgress(async()=>({verdict:judgment,reason:'fixture judgment'}))
      .onStep(event=>events.push({kind:event.kind,iteration:event.iteration,state:event.state,review:event.review,error:event.error}))
      .until(state=>state.remaining===goal));
    outcome={disposition:'done',state:value};
  } catch(error) {outcome={disposition:error.name,state:error.lastState,reason:error.message};}
  if (outcome.disposition!==expected) throw Error(`${id}: expected ${expected}, got ${outcome.disposition}`);
  rows.push({version:'natlang.measure-curriculum/1',id:`measure/${id}`,task:{kind:'authoring',instruction},
    program:`await iterateOn(state => ({...state, remaining: state.remaining + (${delta})}), {remaining: ${initial}, quality: 0.5}).withMeasure(state => state.remaining).checkProgress(async () => ({verdict: ${JSON.stringify(judgment)}, reason: 'fixture judgment'})).until(state => state.remaining === ${goal});`,
    expected,observed:outcome,events,provenance:{kind:'synthetic-runtime-verified',teacherTranscript:false},admission:'authoring-fixture; collect interpreter decisions separately'});
}
await mkdir(output,{recursive:true});
await writeFile(join(output,'cases.jsonl'),rows.map(JSON.stringify).join('\n')+'\n');
await writeFile(join(output,'manifest.json'),JSON.stringify({version:'natlang.measure-curriculum/1',count:rows.length,verified:rows.length,api:['withMeasure','checkProgress'],source:'ts-host/scripts/self-improvement/build-measure-curriculum.mjs',trainingAdmission:'fixtures only; not admitted as teacher interpreter trajectories'},null,2)+'\n');
console.log(JSON.stringify({count:rows.length,verified:rows.length,output}));
