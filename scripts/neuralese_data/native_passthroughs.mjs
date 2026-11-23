#!/usr/bin/env node
/** Replace exact native-service passthrough leaves with typed immutable TS, retaining NL policy. */
import {readFile,writeFile,readdir,stat,mkdir,unlink} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {createRequire} from 'node:module';
import {resolve,join,dirname,relative,basename} from 'node:path';
import {pathToFileURL,fileURLToPath} from 'node:url';
const hash=value=>createHash('sha256').update(value).digest('hex');
const builtins=new Set('string number boolean bigint symbol undefined null unknown any void never object this Array ReadonlyArray Promise PromiseLike Record Readonly Partial Required Pick Omit Exclude Extract NonNullable ReturnType Parameters InstanceType ConstructorParameters Awaited Uint8Array Uint16Array Uint32Array Int8Array Int16Array Int32Array Float32Array Float64Array BigInt64Array BigUint64Array ArrayBuffer ArrayBufferLike SharedArrayBuffer DataView Map ReadonlyMap Set ReadonlySet WeakMap WeakSet Date RegExp Error JSON Function Object String Number Boolean'.split(' '));
async function exists(path){try{await stat(path);return true;}catch(error){if(error.code==='ENOENT')return false;throw error;}}
async function walk(root){const out=[];for(const item of await readdir(root,{withFileTypes:true})){if(['node_modules','.git','dist','vendor'].includes(item.name))continue;const path=join(root,item.name);if(item.isDirectory())out.push(...await walk(path));else if(item.isFile())out.push(path);}return out;}
export async function inventoryNativePassthroughs(root,reader,bindings={}){
 root=resolve(root);const require=createRequire(join(reader,'ts-host/package.json')),ts=require('typescript');
 const {readNatlangFrontmatter}=await import(pathToFileURL(join(reader,'ts-host/dist/runtime/loader.js')));
 const files=await walk(root),rows=[];
 const configCache=new Map();
 async function packageAt(path){for(let dir=dirname(path);dir.startsWith(root);dir=dirname(dir)){if(await exists(join(dir,'natlang.json')))return dir;if(dir===root)break;}return null;}
 async function configuration(dir){if(!configCache.has(dir))configCache.set(dir,JSON.parse(await readFile(join(dir,'natlang.json'),'utf8')));return configCache.get(dir);}
 function references(text){const file=ts.createSourceFile('types.ts',`type __Result=${text};`,ts.ScriptTarget.Latest,true);const names=new Set();const visit=node=>{if(ts.isTypeReferenceNode(node)){let name=node.typeName;while(ts.isQualifiedName(name))name=name.left;if(ts.isIdentifier(name)&&!builtins.has(name.text))names.add(name.text);}ts.forEachChild(node,visit);};visit(file);return names;}
 const serviceCache=new Map();
 async function serviceNames(dir){if(serviceCache.has(dir))return serviceCache.get(dir);const names=new Set();
  for(const file of files.filter(path=>path.startsWith(dir+'/')&&path.endsWith('.d.ts'))){
   const source=ts.createSourceFile(file,await readFile(file,'utf8'),ts.ScriptTarget.Latest,true);
   const visit=node=>{if(ts.isModuleDeclaration(node)&&ts.isStringLiteral(node.name)&&node.name.text==='natlang:services'){
    const inspect=child=>{if(ts.isVariableDeclaration(child)&&ts.isIdentifier(child.name))names.add(child.name.text);if(ts.isFunctionDeclaration(child)&&child.name)names.add(child.name.text);if(ts.isModuleDeclaration(child)&&ts.isIdentifier(child.name))names.add(child.name.text);if(ts.isExportSpecifier(child))names.add(child.name.text);ts.forEachChild(child,inspect);};if(node.body)inspect(node.body);
   }else ts.forEachChild(node,visit);};visit(source);
  }serviceCache.set(dir,names);return names;
 }
 for(const file of files.filter(path=>path.endsWith('.nl'))){
  const source=await readFile(file,'utf8'),parts=/^---\r?\n([\s\S]*?)\r?\n---\r?\n([\s\S]*)$/u.exec(source);if(!parts)continue;
  const body=parts[2].trim();
  let call=/^Return\s+(?:the\s+(?:(?:exact|actual)\s+)?results? of\s+)?(?:await\s+)?(?:services\.)?([A-Za-z_$][\w$]*)\.([A-Za-z_$][\w$]*)\(([^()]*)\)(?:\s+(?:unchanged|verbatim|exactly|for the original goal))?\.$/u.exec(body);
  if(call){const rest=body.slice(call[0].indexOf(')')+1).replace(/^[^.]*(?:\.|$)/u,'');if(/\b(?:[Cc]all|[Pp]ass|[Gg]ive|[Ii]f|[Tt]hen|[Cc]hoose|[Ii]nfer|[Cc]lassify|[Ii]nterpret|[Cc]onstruct|[Rr]ewrite)\b/u.test(rest))call=null;}
  let inferred=false;
  if(/\b(?:match|quote|omit|once per|for each|ignore the archive|preferring|classify|infer|choose|select|summarize|rewrite|transform)\b/iu.test(body))continue;
  if(!call){
   const prefix=/^Call\s+(?:services\.)?([A-Za-z_$][\w$]*)\.([A-Za-z_$][\w$]*)(?:\(([^()]*)\))?/u.exec(body);
   if(!prefix)continue;
   const rest=body.slice(prefix[0].length);
   if(body.length>330||/\b(?:if|when|then|otherwise|unless|only|require|identify|compare|choose|select|classify|interpret|infer|derive|construct|check|verify|validate|pass|give|expect|for each)\b/iu.test(rest)||/\b[Cc]all\b/u.test(rest))continue;
   call=prefix;inferred=prefix[3]===undefined||!/^\s*(?:and return (?:its |the )?(?:actual |exact )?(?:result|receipt|observation)(?: unchanged)?)?\.$/u.test(rest);
  }
  const row={path:relative(root,file),line:1,body_line:source.slice(0,source.indexOf(parts[2])).split('\n').length,source_sha256:hash(source),service:call[1],method:call[2],arguments:call[3],status:'kept',reason:''};rows.push(row);
  const owner=await packageAt(file);if(!owner){row.reason='No enclosing Natlang package declares the service binding';continue;}
  const meta=readNatlangFrontmatter(parts[1]),config=await configuration(owner),own=relative(owner,file);
  row.package=relative(root,owner);row.metadata=meta;
  if(bindings[row.path]){const binding=bindings[row.path];if(binding.source_sha256!==row.source_sha256)throw Error('Manual binding source changed: '+row.path);call[3]=binding.arguments;row.arguments=call[3];row.manual_binding=binding.rationale;inferred=false;}
  if(inferred){
   row.reason='Native call argument projection needs explicit review';continue;
  }

  if(!(await serviceNames(owner)).has(call[1])){row.reason='Receiver is not a declared native service; this may compose semantic functions';continue;}
  const policy=[...new Set([...Object.keys(meta),...Array.from(parts[1].matchAll(/^([A-Za-z][\w-]*):/gmu),match=>match[1])])].filter(key=>!['description','args','returns','types','budget','model','withhold','protect'].includes(key));
  if(policy.length){row.reason=`Preserve additional execution/postcondition semantics: ${policy.join(', ')}`;continue;}
  if(await exists(file.slice(0,-3))){row.reason='Owns a callable subtree; preserve hierarchy until TS owner support is qualified';continue;}
  if(config.workspaceBranches?.[own.slice(0,-3)]){row.reason='Owns a mounted callable subtree; workspaceModules currently requires an NL owner';continue;}
  if(!meta.returns||typeof meta.returns!=='string'||Object.values(meta.args??{}).some(value=>typeof value!=='string')){row.reason='Requires an explicit typed signature';continue;}
  const inline=meta.types??{},typeNames=new Set();for(const type of [...Object.values(meta.args??{}),meta.returns,...Object.values(inline)])for(const name of references(type))if(!Object.hasOwn(inline,name))typeNames.add(name);
  let typesFile=null;for(let dir=dirname(file);dir.startsWith(owner);dir=dirname(dir)){if(await exists(join(dir,'types.ts'))){typesFile=join(dir,'types.ts');break;}if(dir===owner)break;}
  if(typeNames.size&&!typesFile){row.reason=`Native signature needs explicit type imports: ${[...typeNames].join(', ')}`;continue;}
  const params=Object.entries(meta.args??{}).map(([name,type])=>`${name}: ${type}`).join(', ');
  const imports=[`import {${call[1]}} from "natlang:services";`];
  if(typeNames.size){let target=relative(dirname(file),typesFile).replace(/\.ts$/u,'.js');if(!target.startsWith('.'))target='./'+target;imports.push(`import type {${[...typeNames].sort().join(', ')}} from ${JSON.stringify(target)};`);}
  const aliases=Object.entries(inline).map(([name,type])=>`type ${name}=${type};`);
  const name=basename(file,'.nl');const functionName=/^[A-Za-z_$][\w$]*$/u.test(name)&&!['delete','new','default','import','export','return','class','function'].includes(name)?' '+name:'';
  row.status='convert';row.reason='Exact native service call, typed signature, no semantic judgment or child subtree';row.output=relative(root,file.slice(0,-3)+'.ts');
  row.generated=[...imports,...aliases,`/** ${String(meta.description??'Exact native service value.').replaceAll('*/','* /')} */`,`export default async function${functionName}(${params}):Promise<${meta.returns}>{`,` return ${call[1]}.${call[2]}(${call[3]});`,'}',''].join('\n');
 }
 const previousPath=join(root,'meta/NATIVE_PASSTHROUGH_INVENTORY.json');
 if(await exists(previousPath)){
  const previous=JSON.parse(await readFile(previousPath,'utf8'));
  for(const old of previous.rows??[])if(!rows.some(row=>row.path===old.path)&&old.generated&&old.output&&!await exists(join(root,old.path))&&await exists(join(root,old.output))){
   const actual=await readFile(join(root,old.output),'utf8');if(actual!==old.generated)throw Error('Converted leaf changed outside recorded transformation: '+old.output);
   rows.push({...old,status:'converted'});
  }
 }
 return {schema:'natlang.native-passthrough-inventory/1',root,rows};
}
export async function applyNativePassthroughs(report){
 const changed=new Map();
 for(const row of report.rows.filter(row=>row.status==='convert')){
  const old=join(report.root,row.path),source=await readFile(old,'utf8');if(hash(source)!==row.source_sha256)throw Error(`source changed: ${row.path}`);
  const target=join(report.root,row.output);if(await exists(target))throw Error(`existing TS function: ${row.output}`);
  await writeFile(target,row.generated);await unlink(old);
  const declaration=old.slice(0,-3)+'.d.nl.ts';if(await exists(declaration))await unlink(declaration);
  const configPath=join(report.root,row.package,'natlang.json');const config=changed.get(configPath)??JSON.parse(await readFile(configPath,'utf8'));
  const own=relative(dirname(configPath),old),next=own.slice(0,-3)+'.ts';
  if(config.serviceScopes)for(const [name,paths]of Object.entries(config.serviceScopes))config.serviceScopes[name]=paths.filter(path=>path!==own&&path!==own+'/**');
  const rewrite=value=>{if(Array.isArray(value))return value.map(rewrite);if(value&&typeof value==='object')return Object.fromEntries(Object.entries(value).map(([key,item])=>[key,rewrite(item)]));return value===own?next:value;};
  changed.set(configPath,rewrite(config));row.status='converted';
 }
 for(const [path,config]of changed)await writeFile(path,JSON.stringify(config,null,2)+'\n');
}
async function main(){const args=process.argv.slice(2),root=resolve(args[0]),reader=resolve(args[1]),apply=args.includes('--apply');const at=args.indexOf('--bindings'),bindings=at<0?{}:JSON.parse(await readFile(args[at+1],'utf8'));const report=await inventoryNativePassthroughs(root,reader,bindings);if(apply)await applyNativePassthroughs(report);const out=join(root,'meta/NATIVE_PASSTHROUGH_INVENTORY.json');await mkdir(dirname(out),{recursive:true});await writeFile(out,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({root,converted:report.rows.filter(row=>['convert','converted'].includes(row.status)).length,kept:report.rows.filter(row=>row.status==='kept').length,applied:apply,inventory:out}));}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url))await main();
