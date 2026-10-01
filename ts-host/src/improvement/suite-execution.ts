/** Reuse native evaluation fixtures and independent metrics for actual edited source. */
import {mkdirSync,writeFileSync} from 'node:fs';
import {join} from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {buildProject} from '../compiler/node-project.js';
import {formatDiagnostics} from '../compiler/project.js';
import {fingerprint} from '../adaptation/identity.js';
import {runWorker} from '../evaluation/worker.js';
import type {PreparedSuite} from '../evaluation/types.js';
import type {ModelDriver} from '../runtime/runtime.js';
import {sourceFiles,type SourceCaseExecution} from './host.js';
/** Only the top-level host owns this adapter; target code never receives it. */
export function suiteExecution(prepared:PreparedSuite,directory:string,driver:ModelDriver,options:{signal?:AbortSignal;judge?:ModelDriver;timeoutMs?:number;judgeIdentity?:string}={}):SourceCaseExecution {
  if(options.judge && !options.judgeIdentity)throw new Error('declare the frozen judge identity');
  const builds=new Map<string,{program:PreparedSuite['program'];compiledEntry:string}>();
  const execute: SourceCaseExecution = Object.assign(async(folder:import('../native/scoped-fs.js').FolderSnapshot,row:import('./types.js').ImprovementCase,seed:number,gateway:import('../evaluation/usage.js').UsageGateway)=>{
    const testCase=prepared.cases.find(testCase=>testCase.id===row.id);
    if(!testCase)throw new Error('foreign fixture case');
    let built=builds.get(folder.digest);
    if(!built){
      const root=join(directory,folder.digest),outDir=join(root,'build');mkdirSync(root,{recursive:true});
      for(const [path,source] of Object.entries(sourceFiles(folder))){const destination=join(root,path);mkdirSync(join(destination,'..'),{recursive:true});writeFileSync(destination,source);}
      writeFileSync(join(root,'package.json'),JSON.stringify({type:'module'}));
      const result=buildProject({project:root,programId:prepared.program.id,guidance:prepared.suite.program.guidance,services:prepared.suite.services,outDir,write:true,writeDeclarations:false,constrained:true,
        runtimeModule:{specifiers:['@natlang/node','@natlang/browser'],url:new URL('./target-runtime.js',import.meta.url).href,types:fileURLToPath(new URL('./target-runtime.d.ts',import.meta.url))}});
      if(!result.ok)throw new Error(formatDiagnostics(result.diagnostics));
      built={program:result.manifest.adaptation!,compiledEntry:pathToFileURL(join(outDir,prepared.suite.program.entry.replace(/\.m?ts$/,'.js'))).href};builds.set(folder.digest,built);
    }
    const result=await runWorker({moduleURL:prepared.moduleURL,compiledEntry:built.compiledEntry,program:built.program,testCase,artifact:null,seed,replicate:0,
      id:fingerprint({source:folder.digest,suite:prepared.suiteHash,case:row.id,seed})},driver,gateway,options.signal,options.timeoutMs,options.judge);
    if(result.status!=='scored' || result.quality===null)throw new Error('fixture execution did not produce an independent score');
    return {value:result.outcome.value,...(result.outcome.kind==='threw'?{error:result.outcome.error?.message}:{}),score:{quality:result.quality,gates:result.gates}};
  },{evaluationLevel:1 as const,identity:fingerprint({suite:prepared.suiteHash,judge:options.judgeIdentity??null,timeout:options.timeoutMs??120000})});
  return execute;
}
