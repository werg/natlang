import {readFileSync,writeFileSync,existsSync,mkdirSync,renameSync,unlinkSync} from 'node:fs';
import {resolve,dirname,relative,join} from 'node:path';
import {randomUUID} from 'node:crypto';
import {Folder} from '../native/scoped-fs.js';
import type {CheckReport} from './types.js';
import type {FolderSnapshot} from '../native/scoped-fs.js';
export type SourceManifest={schema:'natlang.program-source/1';source:string;base:string;files:Record<string,string>;baseFiles:Record<string,string>;contract:unknown};
export type AdoptionRecord={schema:'natlang.source-adoption/1';directory:string;before:Record<string,string>;after:Record<string,string>;beforeDigest:string;afterDigest:string;changed:string[]};
const digest=(files:Record<string,string>)=>Folder.fromFiles(files).snapshot().digest;
function pathIn(directory:string,path:string){const target=resolve(directory,path),rel=relative(directory,target);if(!rel||rel.startsWith('..')||rel.startsWith('/')||path.includes('\\'))throw Error('source path outside adoption root: '+path);return target;}
function current(directory:string,paths:string[]){return Object.fromEntries(paths.filter(path=>existsSync(pathIn(directory,path))).map(path=>[path,readFileSync(pathIn(directory,path),'utf8')]));}
function install(directory:string,before:Record<string,string>,after:Record<string,string>){
 const paths=[...new Set([...Object.keys(before),...Object.keys(after)])].sort(),token=randomUUID(),staged=new Map<string,string>();
 try {
  for(const [path,text]of Object.entries(after)){const target=pathIn(directory,path);mkdirSync(dirname(target),{recursive:true});const temp=target+'.natlang-'+token;writeFileSync(temp,text);staged.set(target,temp);}
  for(const path of paths){const target=pathIn(directory,path);if(path in after)renameSync(staged.get(target)!,target);else if(existsSync(target))unlinkSync(target);}
 } catch(error){
  for(const path of paths){const target=pathIn(directory,path);if(path in before){mkdirSync(dirname(target),{recursive:true});writeFileSync(target,before[path]!);}else if(existsSync(target))unlinkSync(target);}
  throw error;
 } finally {for(const temp of staged.values())if(existsSync(temp))unlinkSync(temp);}
}
/** Verify the complete source before publication and reject stale checkout contents. */
export async function adoptSource(directory:string,manifest:SourceManifest,verify:(source:FolderSnapshot)=>Promise<CheckReport>,recordPath?:string):Promise<AdoptionRecord>{
 directory=resolve(directory);
 if(digest(manifest.files)!==manifest.source||digest(manifest.baseFiles)!==manifest.base)throw Error('source manifest digest mismatch');
 const paths=[...new Set([...Object.keys(manifest.baseFiles),...Object.keys(manifest.files)])].sort();
 const before=current(directory,paths);
 if(digest(before)!==manifest.base)throw Error('stale checkout: source differs from expected base');
 const checked=await verify(Folder.fromFiles(manifest.files).snapshot());if(!checked.valid)throw Error('adoption build failed: '+checked.diagnostics.join('\n'));
 // Verification can be asynchronous; check again immediately before writes.
 if(digest(current(directory,paths))!==manifest.base)throw Error('stale checkout during build verification');
 const changed=paths.filter(path=>manifest.baseFiles[path]!==manifest.files[path]);
 const record:AdoptionRecord={schema:'natlang.source-adoption/1',directory,before:manifest.baseFiles,after:manifest.files,beforeDigest:manifest.base,afterDigest:manifest.source,changed};
 if(recordPath){mkdirSync(dirname(resolve(recordPath)),{recursive:true});writeFileSync(recordPath,JSON.stringify({...record,status:'prepared'},null,2));}
 install(directory,manifest.baseFiles,manifest.files);
 if(recordPath)writeFileSync(recordPath,JSON.stringify({...record,status:'installed'},null,2));
 return record;
}
export function rollbackSource(record:AdoptionRecord):void{
 const paths=[...new Set([...Object.keys(record.before),...Object.keys(record.after)])];
 if(digest(current(record.directory,paths))!==record.afterDigest)throw Error('stale checkout: cannot roll back changed source');
 install(record.directory,record.after,record.before);
}
/** Finish a recorded interrupted publication only when every tracked file matches either side. */
export function recoverAdoption(recordPath:string):AdoptionRecord{
 const saved=JSON.parse(readFileSync(recordPath,'utf8')) as AdoptionRecord&{status:string};
 if(saved.status!=='prepared'&&saved.status!=='installed')throw Error('adoption record is not recoverable');
 if(digest(saved.before)!==saved.beforeDigest||digest(saved.after)!==saved.afterDigest)throw Error('adoption record digest mismatch');
 const paths=[...new Set([...Object.keys(saved.before),...Object.keys(saved.after)])],observed=current(saved.directory,paths);
 for(const path of paths)if(observed[path]!==saved.before[path]&&observed[path]!==saved.after[path])throw Error('stale checkout during adoption recovery: '+path);
 install(saved.directory,observed,saved.after);
 const temporary=recordPath+'.tmp';writeFileSync(temporary,JSON.stringify({...saved,status:'installed'},null,2));renameSync(temporary,recordPath);
 return saved;
}
