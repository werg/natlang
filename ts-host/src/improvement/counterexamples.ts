import {fingerprint,immutable} from '../adaptation/identity.js';
import type {ImprovementCase} from './types.js';
export type IndependentOracle={identity:string;expected:(args:readonly unknown[])=>Promise<unknown>};
export type CounterexampleAdmission={accepted:ImprovementCase[];rejected:{args:unknown[];reason:string}[];cases:readonly ImprovementCase[];suiteVersion:string;oracle:string};
/** A version is a value: admitting examples returns a new suite and consumes its finite oracle allowance. */
export class CounterexampleSuite {
 readonly cases:readonly ImprovementCase[];
 readonly version:string;
 constructor(cases:readonly ImprovementCase[],readonly oracle:IndependentOracle,readonly remainingChecks:number){
  if(!oracle.identity||!Number.isSafeInteger(remainingChecks)||remainingChecks<0)throw Error('pinned oracle and finite remaining checks required');
  this.cases=immutable(structuredClone(cases));this.version=fingerprint({cases:this.cases,oracle:oracle.identity});Object.freeze(this);
 }
 async admit(suggestions:readonly unknown[][]){
  const count=Math.min(suggestions.length,this.remainingChecks);
  const admission=await admitCounterexamples(this.cases,suggestions,this.oracle,count);
  return {suite:new CounterexampleSuite(admission.cases,this.oracle,this.remainingChecks-count),admission};
 }
}
/** Only an author-provided independent oracle may add gold. Locked cohorts stay unchanged. */
export async function admitCounterexamples(existing:readonly ImprovementCase[],suggestions:readonly unknown[][],oracle:IndependentOracle,maxChecks:number):Promise<CounterexampleAdmission>{
 if(!oracle.identity||!Number.isSafeInteger(maxChecks)||maxChecks<0)throw Error('a pinned independent oracle and finite check allowance are required');
 const accepted:ImprovementCase[]=[],rejected:{args:unknown[];reason:string}[]=[],seen=new Map<string,ImprovementCase>();
 for(const row of existing)seen.set(fingerprint(row.args),row);
 for(const args of suggestions.slice(0,maxChecks)){
  const input=fingerprint(args),prior=seen.get(input);
  // Never query the oracle for a suggestion overlapping any protected cohort.
  if(prior&&prior.split!=='train'){rejected.push({args:structuredClone(args),reason:'protected input overlap'});continue;}
  try{
   const expected=await oracle.expected(structuredClone(args));fingerprint(expected);
   if(prior){rejected.push({args:structuredClone(args),reason:fingerprint(expected)===fingerprint(prior.expected)?'duplicate training input':'independent oracle contradicts existing gold'});continue;}
   const row:ImprovementCase={id:'counterexample-'+input,group:'counterexample-'+input,split:'train',args:structuredClone(args),expected:structuredClone(expected)};
   accepted.push(row);seen.set(input,row);
  }catch(error){rejected.push({args:structuredClone(args),reason:'independent oracle failed: '+String(error)});}
 }
 const cases=immutable([...structuredClone(existing),...accepted]);
 return {accepted,rejected,cases,suiteVersion:fingerprint({cases,oracle:oracle.identity}),oracle:oracle.identity};
}
