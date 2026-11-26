import {hexDigest} from '../native/hash.js';

export type SourceTypeExports = Record<string, Record<string,string>>;
/** Pure frontend data, never executable records, service values or module state. */
export type SourceTypeContext = {
  schema:'natlang.source-types/1';
  root:string;
  sourceDigest:string;
  compilerOptions:Record<string,unknown>;
  compilerDigest:string;
  modules:SourceTypeExports;
  modulesDigest:string;
};
const canonical=(value:unknown):unknown=>Array.isArray(value)?value.map(canonical):value&&typeof value==='object'
  ? Object.fromEntries(Object.entries(value).filter(([,v])=>v!==undefined).sort(([a],[b])=>a<b?-1:a>b?1:0).map(([k,v])=>[k,canonical(v)])):value;
const digest=(value:unknown)=>hexDigest(JSON.stringify(canonical(value)));
// Runtime realms/modules are not frontend inputs and must never enter serialized metadata.
const optionNames=['target','inlineRecords','constrained','programId','guidance','importedGuidancePrograms','services','runtimeSpecifiers','emit','declarations','adaptation','reuseSourceTypes'] as const;
const frontendOptions=(value:Record<string,unknown>)=>Object.fromEntries(optionNames.filter(key=>value[key]!==undefined).map(key=>[key,value[key]]));
export function sourceTypeReuseEnabled():boolean {
  return (globalThis as {process?:{env?:Record<string,string|undefined>}}).process?.env?.NATLANG_SOURCE_TYPE_REUSE!=='0';
}
export function sourceTypeContext(files:Record<string,string>,root:string,compilerOptions:Record<string,unknown>,modules:SourceTypeExports):SourceTypeContext {
  const options=structuredClone(frontendOptions(compilerOptions)),aliases=structuredClone(modules);
  return {schema:'natlang.source-types/1',root,sourceDigest:digest(files),compilerOptions:options,compilerDigest:digest(options),modules:aliases,modulesDigest:digest(aliases)};
}
/** Match exact immutable source/mount/SKILL bytes, lexical root, compiler options and alias data. */
export function sourceTypeContextMatches(context:SourceTypeContext|undefined,files:Record<string,string>,root:string,compilerOptions:Record<string,unknown>=context?.compilerOptions??{}):boolean {
  return !!context&&context.schema==='natlang.source-types/1'&&context.root===root&&context.sourceDigest===digest(files)&&
    context.compilerDigest===digest(frontendOptions(compilerOptions))&&context.compilerDigest===digest(context.compilerOptions)&&
    !!context.modules&&Object.values(context.modules).every(aliases=>!!aliases&&typeof aliases==='object'&&!Array.isArray(aliases)&&Object.values(aliases).every(value=>typeof value==='string'))&&
    context.modulesDigest===digest(context.modules);
}
