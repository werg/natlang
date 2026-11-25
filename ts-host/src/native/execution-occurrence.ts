/** Canonical source-graph occurrence identity, shared by interpreted and crisp calls. */
import {hexDigest} from './hash.js';
import type {NativeExecutionOccurrence} from '../contracts.js';
export function stableOccurrenceBinding(value:unknown):string {
  const normalize=(item:unknown):unknown=>{
    if(Array.isArray(item))return item.map(normalize);
    if(!item||typeof item!=='object')return item;
    const record=item as Record<string,unknown>;
    if(record.$live&&typeof record.$live==='object')return {$live:{type:(record.$live as Record<string,unknown>).type}};
    if(record.$host&&typeof record.$host==='object')return {$host:normalize(record.$host)};
    return Object.fromEntries(Object.keys(record).sort().map(key=>[key,normalize(record[key])]));
  };
  return JSON.stringify(normalize(value));
}
export const occurrenceArgumentHash=(value:unknown):string=>hexDigest(stableOccurrenceBinding(value));
export function executionOccurrence(input:{definitionId:string;revision:string;context:string;signature:string;parent?:Partial<NativeExecutionOccurrence>;ordinal:number;argumentHash:string}):NativeExecutionOccurrence {
  const base={schema:'natlang.execution-occurrence/1' as const,definition_id:input.definitionId,revision:input.revision,context:input.context,signature:input.signature,
    parent_occurrence_sha256:typeof input.parent?.occurrence_sha256==='string'?input.parent.occurrence_sha256:null,sibling_ordinal:input.ordinal,argument_binding_sha256:input.argumentHash};
  return {...base,occurrence_sha256:hexDigest(stableOccurrenceBinding(base))};
}
