/**
 * Recording model turns for gradient replay (spec/NEURALESE_GRAPH.md, "Replay"; `natlang:learning`).
 *
 * A Neuralese-capable driver reports every exchange it makes to the active recorder: the request it sent (messages
 * with block parts, tools) and the assistant message the server returned (with block parts and the write records of
 * the blocks it wrote). The learning module installs the source of the active recorder (Node: `AsyncLocalStorage`),
 * so this module stays platform-neutral.
 */
export type NativeWriteOccurrence={index:number;id:string;node:string};
export type NativeReadOccurrence={index:number;id:string;source_state_id:string;node:string;source:string;stop_gradient?:true};
export type NativeTurnGraphBinding={graph_turn:string;written_nodes:Record<string,string>;read_bindings:Record<string,string>;
  written_occurrences?:NativeWriteOccurrence[];read_occurrences?:NativeReadOccurrence[]};
export type NativeRecordedTurn={schema:'natlang.native-recorded-turn/1';id:string;
  invocation_id?:string;retry:number;graph?:NativeTurnGraphBinding} & (
    {kind:'generation';sampling:Record<string,unknown>} | {kind:'decision';decision:Record<string,unknown>});
export type RecordedTurn = {
  readonly messages: unknown[];
  readonly tools?: unknown[];
  /** The assistant message as the server returned it: content and tool-call arguments may hold block parts. */
  readonly reply: Record<string, unknown>;
  /** Write records of the blocks this turn wrote. */
  readonly blocks: readonly Record<string, unknown>[];
  /** A decision readout instead of a generated reply: the options it scored (native/decision.ts). */
  readonly decision?: { readonly options: readonly string[] };
  /** Exact named native coefficient actors and scales active for this recorded turn. */
  readonly adapters?: readonly NativeAdapterBinding[];
  /** Explicit transport contract retained for exact native replay. */
  readonly native_state_modes?: Readonly<Record<string, 'same-base-adapter-state'>>;
  /** Exact native receipt and actual execution-graph occurrence binding. */
  readonly native?: NativeRecordedTurn;
};

/** A named native coefficient actor bound to a call, with its explicit scale. */
export type NativeAdapterBinding = { readonly id: string; readonly scale: number; readonly stop_gradient?:true };
/** A stored positional KV/hybrid-state value supplied as context to a model call. */
export type NativeStateBinding = { readonly id: string; readonly stop_gradient?: true; readonly source_occurrence?: string };

export interface TurnRecorder {
  record(turn: RecordedTurn): void;
  linkNativeTurn?(id:string,graphTurn:string):void;
  recordGraphNode?(event:Record<string,unknown>):void;
  nativeStateWriter?(id:string):string|undefined;
  /** Exact in-episode native writer nodes, including nested/child invocation traces. */
  nativeStateWriters?(id:string):readonly string[];
}

let source: () => TurnRecorder | undefined = () => undefined;

/** Install where the active recorder comes from (the learning module does this). */
export function setRecorderSource(next: () => TurnRecorder | undefined): void { source = next; }

/** The recorder of the current asynchronous context, if any. */
export function activeRecorder(): TurnRecorder | undefined { return source(); }

let nativeAdapterSource: () => readonly NativeAdapterBinding[] | undefined = () => undefined;
let nativeStateSource: () => readonly NativeStateBinding[] | undefined = () => undefined;

/** Install where the active nnt1 actor bindings come from. */
export function setNativeAdapterSource(next: () => readonly NativeAdapterBinding[] | undefined): void { nativeAdapterSource = next; }

/** Install where active stored native-state context comes from. */
export function setNativeStateSource(next: () => readonly NativeStateBinding[] | undefined): void { nativeStateSource = next; }

let lexicalAdapterSource: () => readonly NativeAdapterBinding[] | undefined = () => undefined;
let lexicalSoftStateSource: () => readonly string[] | undefined = () => undefined;

/** Install where the current invocation's tuned soft instruction/skill state IDs come from (the runtime does this). */
export function setLexicalSoftStateSource(next: () => readonly string[] | undefined): void { lexicalSoftStateSource = next; }

/** Soft states the current invocation consumes: written by the exact base and read by its tuned stack, so their
 * consumption is the explicit same-base adapter-state exchange (plan §3.3). */
export function activeSoftStates(): readonly string[] { return lexicalSoftStateSource() ?? []; }

/** Install where the current invocation's lexical (tuned) weight stack comes from (the runtime does this). */
export function setLexicalNativeAdapterSource(next: () => readonly NativeAdapterBinding[] | undefined): void { lexicalAdapterSource = next; }

/**
 * Native coefficient actors active in the current asynchronous context: the invocation's lexical stack resolved from
 * its own binding, then explicit `withNativeAdapters` scopes in nesting order (plans/HIERARCHICAL_LORA_TUNING.md §6).
 */
export function activeNativeAdapters(): readonly NativeAdapterBinding[] {
  const lexical = lexicalAdapterSource(), explicit = nativeAdapterSource();
  return lexical?.length ? explicit?.length ? [...lexical, ...explicit] : lexical : explicit ?? [];
}

/** Stored positional state active in the current asynchronous context. */
export function activeNativeStates(): readonly NativeStateBinding[] { return nativeStateSource() ?? []; }

export class ModelTurnRecorder implements TurnRecorder {
  readonly turns: RecordedTurn[] = [];
  /** Turns of outputs captured under their own recorder (function outputs): never claimed by an objective, but
   * producers of the gradient session like any other turn, so a downstream loss reaches their writes. */
  readonly nested: RecordedTurn[] = [];
  private claimed = 0;
  /** Objectives awaiting a promise output right now, and whether two of them overlapped (see `capture`). */
  active = 0;
  overlapped = false;
  readonly graphNodes=new Map<string,Record<string,unknown>>();
  readonly nativeLinks=new Map<string,string>();
  private readonly graphConflicts=new Set<string>();
  record(turn: RecordedTurn): void { this.turns.push(turn); }
  recordGraphNode(event:Record<string,unknown>):void{
    if(typeof event.node==='string'){
      const previous=this.graphNodes.get(event.node);
      if(previous&&JSON.stringify(previous)!==JSON.stringify(event))this.graphConflicts.add(event.node);
      else this.graphNodes.set(event.node,structuredClone(event));
    }
  }
  nativeStateWriters(id:string):readonly string[]{
    const writers=[...this.graphNodes.values()].filter(event=>event.kind==='block_write'&&event.block===id);
    const nodes=writers.map(event=>event.node).filter((node):node is string=>typeof node==='string');
    if(nodes.some(node=>this.graphConflicts.has(node)))throw Object.assign(
      new Error(`neuralese-native-grad-invalid: native state ${id} has a conflicting in-trajectory writer occurrence`),
      {code:'neuralese-native-grad-invalid'});
    return nodes;
  }
  nativeStateWriter(id:string):string|undefined{
    const writers=this.nativeStateWriters(id);
    if(writers.length>1)throw Object.assign(new Error(`neuralese-native-grad-invalid: native state ${id} has multiple possible in-trajectory producers`),{code:'neuralese-native-grad-invalid'});
    return writers[0];
  }
  linkNativeTurn(id:string,graphTurn:string):void{
    if(!this.turns.some(turn=>turn.native?.id===id))throw Object.assign(new Error('neuralese-native-grad-invalid: native graph link has no exact recorded model turn'),{code:'neuralese-native-grad-invalid'});
    this.bindNativeTurn(id,graphTurn);
  }
  private bindNativeTurn(id:string,graphTurn:string):void {
    const previous=this.nativeLinks.get(id);
    if(previous&&previous!==graphTurn)throw Object.assign(new Error('neuralese-native-grad-invalid: native receipt was linked to two model-turn occurrences'),{code:'neuralese-native-grad-invalid'});
    this.nativeLinks.set(id,graphTurn);
  }
  absorb(other:ModelTurnRecorder):void{
    for(const node of other.graphNodes.values())this.recordGraphNode(node);
    for(const id of other.graphConflicts)this.graphConflicts.add(id);
    for(const [id,turn] of other.nativeLinks)this.bindNativeTurn(id,turn);
  }
  nativeTurns():RecordedTurn[]{
    if([...this.turns,...this.nested].some(turn=>turn.native)&&this.graphConflicts.size)
      throw Object.assign(new Error('neuralese-native-grad-invalid: native execution graph reuses conflicting source node identities'),{code:'neuralese-native-grad-invalid'});
    return [...this.turns,...this.nested].filter(turn=>turn.native).map(turn=>{
      const graphTurn=this.nativeLinks.get(turn.native!.id);
      if(!graphTurn)return structuredClone(turn);
      const modelNode=this.graphNodes.get(graphTurn);
      if(modelNode?.kind!=='model_turn'||turn.native!.invocation_id&&modelNode.call_id!==turn.native!.invocation_id||
          turn.native!.kind==='decision'&&modelNode.readout!=='decision'||
          turn.native!.kind==='generation'&&modelNode.readout!==undefined)
        throw Object.assign(new Error('neuralese-native-grad-invalid: native receipt does not match its actual model-turn occurrence'),{code:'neuralese-native-grad-invalid'});
      const capture=turn.native!.kind==='generation'?turn.native!.sampling:turn.native!.decision;
      const binding=nativeCaptureGraphBinding(graphTurn,capture,this.graphNodes);
      return {...structuredClone(turn),native:{...structuredClone(turn.native!),graph:binding}};
    });
  }
  /** Turns recorded since the previous claim. */
  claim(): RecordedTurn[] { const out = this.turns.slice(this.claimed); this.claimed = this.turns.length; return out; }
}

/** Exact capture/event matching; equal IDs do not select a latest writer. */
export function nativeCaptureGraphBinding(name:string,capture:Record<string,unknown>,nodes:ReadonlyMap<string,Record<string,unknown>>):NativeTurnGraphBinding {
  const writes=[...nodes.values()].filter(event=>event.kind==='block_write'&&event.turn===name);
  const reads=[...nodes.values()].filter(event=>event.kind==='block_read'&&event.turn===name);
  const capturedWrites=(capture.writes??[]) as {state_id:string}[];
  const capturedReads=((capture.prompt??[]) as Record<string,unknown>[]).filter(segment=>segment.kind==='native_state');
  const fail=(message:string):never=>{throw Object.assign(new Error(`neuralese-native-grad-invalid: ${message}`),{code:'neuralese-native-grad-invalid'});};
  const source=(event:Record<string,unknown>):string=>{
    const edges=Array.isArray(event.inputs)?event.inputs as Record<string,unknown>[]:[];
    const sources=edges.filter(edge=>edge.block===event.block).map(edge=>edge.node);
    return sources.length===1&&typeof sources[0]==='string'&&sources[0]?sources[0]:fail('native read needs one exact writer or external input');
  };
  if(writes.length!==capturedWrites.length||reads.length!==capturedReads.length)fail('native capture differs from actual state occurrence inventory');
  const boundWrites:NativeWriteOccurrence[]=capturedWrites.map((write,index)=>{
    const at=writes.findIndex(event=>event.block===write.state_id);
    if(at<0)fail('native write changed its actual artifact');
    const event=writes.splice(at,1)[0]!;
    return {index,id:String(event.block),node:String(event.node)};
  });
  const boundReads:NativeReadOccurrence[]=capturedReads.map((segment,index)=>{
    if(segment.source_state_id!==undefined&&segment.source_state_id!==segment.id)fail('semantic input requires differentiable codec lowering');
    const candidates=reads.filter(event=>event.block===segment.id&&event.stop_gradient===segment.stop_gradient&&
      (segment.source_occurrence===undefined||source(event)===segment.source_occurrence));
    if(!candidates.length||new Set(candidates.map(source)).size!==1)fail('native input has ambiguous or changed source occurrence');
    const event=candidates[0]!;reads.splice(reads.indexOf(event),1);
    const occurrenceSource=source(event);
    if(occurrenceSource!=='external'){
      const writer=nodes.get(occurrenceSource);
      if(writer?.kind!=='block_write'||writer.block!==segment.id)
        fail('native read references a writer occurrence absent from the exact episode graph');
    }
    return {index,id:String(segment.id),source_state_id:String(segment.id),node:String(event.node),source:occurrenceSource,
      ...(segment.stop_gradient===true?{stop_gradient:true}:{})};
  });
  const count=(items:string[])=>{const counts=new Map<string,number>();for(const id of items)counts.set(id,(counts.get(id)??0)+1);return counts;};
  const wc=count(boundWrites.map(item=>item.id)),rc=count(boundReads.map(item=>item.source_state_id));
  return {graph_turn:name,written_nodes:Object.fromEntries(boundWrites.filter(item=>wc.get(item.id)===1).map(item=>[item.id,item.node])),
    read_bindings:Object.fromEntries(boundReads.filter(item=>rc.get(item.source_state_id)===1&&item.source!=='external').map(item=>[item.source_state_id,item.source])),
    ...([...wc.values()].some(value=>value>1)?{written_occurrences:boundWrites}:{}),
    // External roots and repeated imports have no unique producer keyed by ID.
    // Keep every actual occurrence so native-state objectives can bind exact
    // targets and gradients without guessing from an ID-only map.
    ...([...rc.values()].some(value=>value>1)||boundReads.some(item=>item.source==='external')?
      {read_occurrences:boundReads}:{})};
}
