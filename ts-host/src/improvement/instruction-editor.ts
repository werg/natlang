import {FolderSnapshot} from '../native/scoped-fs.js';
import {ProgramView} from '../adaptation/program-view.js';
import {bindAdaptation} from '../adaptation/compatibility.js';
import {candidateArtifact} from '../evaluation/runner.js';
import type {PreparedSuite} from '../evaluation/types.js';
import type {Candidate} from '../adaptation/types.js';
export type InstructionEdit={site:string;segments:string[]};
/** Compiler-resolved site editing preserves interpolation expressions and public signatures. */
export class InstructionEditor {
 #prepared:PreparedSuite;
 constructor(prepared:PreparedSuite){this.#prepared=prepared;}
 sites(){return this.#prepared.program.components.filter(item=>this.#prepared.components.includes(item.key)&&item.kind==='lambda.instructions').map(item=>({site:item.key,name:item.definitionId??item.key,path:item.source?.path,value:item.baseline}));}
 edit(source:FolderSnapshot,edits:readonly InstructionEdit[]):FolderSnapshot{
  const prepared=this.#prepared;
  for(const [path,text]of Object.entries(prepared.program.sources))if(!path.startsWith('@')&&new TextDecoder().decode(source.readBytesSync(path))!==text)throw Error('instruction sites belong to a different source revision; rebuild inventory');
  const components=prepared.program.components.filter(item=>prepared.components.includes(item.key));
  const candidate:Record<string,Candidate[string]>=Object.fromEntries(components.map(item=>[item.key,item.baseline]));
  const seen=new Set<string>();
  for(const edit of edits){const component=components.find(item=>item.key===edit.site);
   if(!component||component.baseline.kind!=='lambda.instructions'||seen.has(edit.site))throw Error('unknown or duplicate instruction site: '+edit.site);
   seen.add(edit.site);candidate[edit.site]={kind:'lambda.instructions',template:{segments:[...edit.segments],slotIds:[...component.baseline.template.slotIds]}};
  }
  const artifact=candidateArtifact(prepared,candidate);
  const view=new ProgramView(prepared.program,bindAdaptation(artifact,prepared.program,prepared.suite.executorIdentity));
  const branch=source.branch();
  for(const [path,text]of Object.entries(prepared.program.sources))if(!path.startsWith('@'))branch.writeText(path,view.source(path,text));
  return branch.snapshot();
 }
}
