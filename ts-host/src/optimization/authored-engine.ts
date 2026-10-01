import {PROGRAM_PROMPT,PROGRAM_TOOLS,PROGRAM_CONTEXT_TOKENS} from '../improvement/prompt.js';
/** Host effects and exact GEPA primitives for the authored component-search reducer.
 * Selection math derives from Ax; see vendor/ax-gepa/UPSTREAM.json. No host search loop. */
import { Folder } from '../native/scoped-fs.js';
import { createNatlangRuntime } from '../runtime/node.js';
import { loadVirtualNatlang } from '../runtime/virtual-project.js';
import { EvidenceView } from '../improvement/evidence.js';
import { AUTHORED_IMPROVER } from '../improvement/authored-source.js';
import { canonical, fingerprint } from '../adaptation/identity.js';
import { validateCandidate } from '../adaptation/compatibility.js';
import { BudgetExhausted, type UsageGateway } from '../evaluation/usage.js';
import { EvaluationFeedbackError } from '../evaluation/feedback.js';
import type { Candidate, ComponentDescriptor } from '../adaptation/types.js';
import type { EvaluationBatch, PreparedSuite } from '../evaluation/types.js';
import { frontierParents, meanBetter, mergeCandidates } from './strategies/gepa.js';
import { ComponentSelector } from './vendor/ax-gepa/gepaSelection.js';
import { getUpdateGroup } from './vendor/ax-gepa/gepaDependencies.js';
import type { OptimizationOptions, OptimizationTarget, SearchCandidate, SearchState } from './types.js';

type Settings = {maxPopulation:number;maxRepairs:number;minibatchSize:number;dependencies:Readonly<Record<string,readonly string[]>>};
type LoopState = {iteration:number;incumbent:string;done:boolean;stopReason:string};
export async function runAuthoredSearch(context: {prepared:PreparedSuite;options:OptimizationOptions;settings:Settings;state:SearchState;
  target:OptimizationTarget;descriptors:readonly ComponentDescriptor[];train:string[];validation:string[];gateway:UsageGateway;
  batch:(candidate:Candidate,ids:readonly string[])=>Promise<EvaluationBatch>;trace?:(value:import('../runtime/runtime.js').InvocationTrace)=>void;checkpoint:()=>void;emit:(event:Record<string,unknown>)=>void}) {
  const {prepared,options,settings,state,target,descriptors,train,validation,gateway,batch,checkpoint,emit}=context;
  const targets=descriptors.map(component=>({key:component.key,dependsOn:settings.dependencies[component.key]??[]}));
  const values=new Map<string,Candidate>(state.population.map(member=>[member.id,member.value]));
  const evidence=new Map<string,{batch:EvaluationBatch;id:string;split:string}>();
  const merged=new Set<string>();
  let plan: {iteration:number;parent:SearchCandidate;selected:{key:string};keys:string[];mini:string[];partner:SearchCandidate}|undefined;
  let selector=new ComponentSelector(targets,state.selector);
  const random=()=>{let x=state.rng;x^=x<<13;x^=x>>>17;x^=x<<5;state.rng=x>>>0;return state.rng/4294967296;};
  const member=()=>state.population.find(item=>item.id===state.incumbent)!;
  const register=(value:Candidate)=>{const id=fingerprint(value);values.set(id,value);return id;};
  let committed=structuredClone(state);
  const pendingEvents:Record<string,unknown>[]=[];
  const event=(value:Record<string,unknown>)=>pendingEvents.push(value);
  const services={
    plan: async(iteration:number)=>{
      gateway.check(options.signal);
      if(iteration!==state.iteration)throw Error('stale authored iteration');
      const covered=new Set(state.population.flatMap(candidate=>candidate.train.results.flatMap(result=>[...result.coverage])));
      const eligible=targets.filter(item=>covered.has(item.key));
      if(!eligible.length)return {stopReason:'coverage-gap',incumbent:member().value};
      const winners=state.strategy==='gepa'?frontierParents(state.population):[state.incumbent];
      const parent=state.population.find(item=>item.id===(winners[Math.floor(random()*winners.length)]??state.incumbent))!;
      let selected=selector.pick(iteration,random);
      if(!eligible.some(item=>item.key===selected.key))selected=eligible[Math.floor(random()*eligible.length)]!;
      const keys=getUpdateGroup(selected,targets).map(item=>item.key);
      gateway.reserve('proposals',1,options.signal);selector.recordProposal(selected.key);
      const shuffled=[...train];for(let index=shuffled.length-1;index>0;index--){const chosen=Math.floor(random()*(index+1));[shuffled[index],shuffled[chosen]]=[shuffled[chosen]!,shuffled[index]!];}
      // The merge draw is made only on a scheduled composition experiment.
      const compose=state.strategy==='gepa'&&iteration%4===3&&state.population.length>1;
      const partner=compose?state.population[Math.floor(random()*state.population.length)]!:parent;
      plan={iteration,parent,selected,keys,mini:shuffled.slice(0,settings.minibatchSize).sort(),partner};
      const feedback=await target.feedback(parent.train,keys);
      return {stopReason:'',parent:parent.value,parentId:parent.id,keys,components:descriptors.filter(item=>keys.includes(item.key)),
        feedback:new EvidenceView(feedback),compose,duplicateIds:state.population.map(item=>item.id)};
    },
    merge:()=>{if(!plan)throw Error('experiment not opened');const value=mergeCandidates(state.baseline.value,plan.parent.value,plan.partner.value);if(value)merged.add(fingerprint(value));return value;},
    check:async(value:Candidate)=>{if(!plan)throw Error('experiment not opened');
      let candidate:Candidate;try {candidate=validateCandidate(value,descriptors);}catch(error){return {valid:false,id:'',feedback:String(error)};}
      for(const [key,base]of Object.entries(plan.parent.value))if(!merged.has(fingerprint(candidate))&&!plan.keys.includes(key)&&canonical(base)!==canonical(candidate[key]))throw Error('unselected component changed: '+key);
      const checked=await target.validate(candidate);return {...checked,id:checked.valid?register(candidate):''};},
    evaluate:async(id:string,split:'mini'|'train'|'validation',parent=false)=>{
      if(!plan)throw Error('experiment not opened');
      const value=parent?plan.parent.value:values.get(id);if(!value)throw Error('unregistered candidate');
      const measured=await batch(value,split==='mini'?plan.mini:split==='train'?train:validation);
      const reference=fingerprint({id:parent?plan.parent.id:id,split,digest:measured.digest});evidence.set(reference,{batch:measured,id:parent?plan.parent.id:id,split});
      return {evidence:reference,quality:measured.quality,gatesPassed:measured.gatesPassed};
    },
    finish:async(input:{candidate?:string;accepted:boolean;train?:string;validation?:string;error?:string;stopReason?:string})=>{
      if(!plan){if(input.stopReason!=='coverage-gap')throw Error('experiment not opened');state.stopReason='coverage-gap';return {state:{iteration:state.iteration,incumbent:state.incumbent,done:true,stopReason:'coverage-gap'},candidate:member().value};}
      const iteration=plan.iteration;
      if(input.accepted){const value=values.get(input.candidate??''),training=evidence.get(input.train??''),validated=evidence.get(input.validation??'');
        if(!value||!training||!validated||training.id!==input.candidate||validated.id!==input.candidate||training.split!=='train'||validated.split!=='validation')throw Error('acceptance requires complete independently executed evidence');
        // Exact acceptance is checked again; semantic code cannot fabricate promotion.
        const a=await batch(plan.parent.value,plan.mini),b=await batch(value,plan.mini);
        if(!b.gatesPassed||!(b.quality!>a.quality!))throw Error('acceptance violates paired improvement');
        if(state.population.some(item=>item.id===input.candidate))throw Error('duplicate accepted candidate');
        const candidate:SearchCandidate={id:input.candidate!,value,parents:[plan.parent.id],train:training.batch,validation:validated.batch};
        state.population.push(candidate);
        const guidanceChanged=descriptors.some(component=>component.kind==='program.guidance'&&canonical(candidate.value[component.key])!==canonical(component.baseline));
        const missing=guidanceChanged?prepared.program.components.filter(component=>component.kind==='lambda.instructions'&&!validated.batch.results.some(result=>result.coverage.includes(component.key))).map(item=>item.key):[];
        if(missing.length)event({iteration,type:'insufficient-guidance-coverage',components:missing});
        else if(meanBetter(candidate,member(),prepared.suite.selection))state.incumbent=candidate.id;
        if(state.population.length>settings.maxPopulation){const frontier=new Set(frontierParents(state.population));
          const remove=state.population.filter(item=>item.id!==state.incumbent&&item.id!==state.baseline.id).sort((a,b)=>Number(frontier.has(a.id))-Number(frontier.has(b.id))||(a.validation.quality??0)-(b.validation.quality??0)||a.id.localeCompare(b.id))[0];
          if(remove)state.population=state.population.filter(item=>item.id!==remove.id);}
      }
      selector.recordResult(plan.selected.key,input.accepted,iteration);state.selector=selector.snapshot();state.iteration++;
      const done=gateway.ledger.proposals>=gateway.limits.maxProposals;
      const type=input.error?'invalid-proposal':input.candidate?(input.accepted?'accepted':'rejected'):'duplicate-proposal';
      event({iteration,type,candidate:input.candidate??null,parent:plan.parent.id,keys:plan.keys,...(input.error?{error:input.error}:{})});
      plan=undefined;
      return {state:{iteration:state.iteration,incumbent:state.incumbent,done,stopReason:done?'completed':''},candidate:member().value};
    }
  };
  const declaration=`export function plan(iteration:number):Promise<unknown>;
export function merge():unknown;
export function check(candidate:unknown):Promise<{valid:boolean;id:string;feedback?:string}>;
export function evaluate(id:string,split:'mini'|'train'|'validation',parent?:boolean):Promise<{evidence:string;quality:number;gatesPassed:boolean}>;
export function finish(decision:unknown):Promise<{state:ComponentLoopState;candidate:Candidate}>;`;
  const runtime=createNatlangRuntime({model:{tools:PROGRAM_TOOLS,prompt:PROGRAM_PROMPT,contextTokens:PROGRAM_CONTEXT_TOKENS,driver:(request,signal)=>gateway.request(options.reflection,request,signal??options.signal,'reflection'),maxTurns:16,maxTokens:32000,turnTokens:4096,maxFailureRepairs:settings.maxRepairs},
    services:{search:services},serviceDeclarations:{search:declaration},serviceScopes:{search:['componentSearchStep.nl']},trace:context.trace,network:false,codeEdits:'deny',signal:options.signal});
  const folder=Folder.fromFiles({'components.json':JSON.stringify(member().value)});
  const initial:LoopState={iteration:state.iteration,incumbent:state.incumbent,done:gateway.ledger.proposals>=gateway.limits.maxProposals,stopReason:''};
  try {
    const result=await runtime.run(()=>folder.iterateOn<LoopState>(loadVirtualNatlang(AUTHORED_IMPROVER,'componentSearchStep.nl'),initial)
      .withMeasure(step=>step.done?0:gateway.limits.maxProposals-step.iteration)
      .onStep(async event=>{if(event.kind==='step'&&event.state){
        if(event.state.state.incumbent!==state.incumbent||canonical(JSON.parse(await event.state.folder.readText('components.json')))!==canonical(member().value))throw Error('authored selected source/state mismatch');
        checkpoint();committed=structuredClone(state);for(const pending of pendingEvents.splice(0))emit(pending);checkpoint();committed=structuredClone(state);}}).until(step=>step.done));
    state.stopReason=(result.state.stopReason||'completed') as SearchState['stopReason'];
  } catch(error){
    for(const key of Object.keys(state))delete (state as unknown as Record<string,unknown>)[key];
    Object.assign(state,committed,{ledger:gateway.snapshot()});pendingEvents.splice(0);
    gateway.check(options.signal);
    // Preserve the underlying accounting/infrastructure exception across interpreter wrappers.
    let cause:unknown=error;const seen=new Set<unknown>();
    while(cause instanceof Error&&!seen.has(cause)){seen.add(cause);if(cause instanceof BudgetExhausted||cause instanceof EvaluationFeedbackError)throw cause;cause=cause.cause;}
    throw error;
  }
}
