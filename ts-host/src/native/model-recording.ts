/**
 * Recording model turns for gradient replay (spec/NEURALESE_GRAPH.md, "Replay"; `natlang:learning`).
 *
 * A Neuralese-capable driver reports every exchange it makes to the active recorder: the request it sent (messages
 * with block parts, tools) and the assistant message the server returned (with block parts and the write records of
 * the blocks it wrote). The learning module installs the source of the active recorder (Node: `AsyncLocalStorage`),
 * so this module stays platform-neutral.
 */
export type NativeTurnGraphBinding={graph_turn:string;written_nodes:Record<string,string>;read_bindings:Record<string,string>};
export type NativeRecordedTurn={schema:'natlang.native-recorded-turn/1';id:string;sampling:Record<string,unknown>;
  invocation_id?:string;retry:number;graph?:NativeTurnGraphBinding};
export type RecordedTurn = {
  readonly messages: unknown[];
  readonly tools?: unknown[];
  /** The assistant message as the server returned it: content and tool-call arguments may hold block parts. */
  readonly reply: Record<string, unknown>;
  /** Write records of the blocks this turn wrote. */
  readonly blocks: readonly Record<string, unknown>[];
  /** A decision readout instead of a generated reply: the options it scored (native/decision.ts). */
  readonly decision?: { readonly options: readonly string[] };
  /** Weight adapters that were active for this turn (`withAdapters`); replay scores the same adapted model. */
  readonly adapters?: readonly AdapterBinding[];
  /** Explicit transport contract retained for exact native replay. */
  readonly native_state_modes?: Readonly<Record<string, 'same-base-adapter-state'>>;
  /** Exact native receipt and actual execution-graph occurrence binding. */
  readonly native?: NativeRecordedTurn;
};

/** An adapter block bound for a call, with its scale (serve/grad.py, model/tiny_adapters.py). */
export type AdapterBinding = { readonly id: string; readonly scale: number };

export interface TurnRecorder {
  record(turn: RecordedTurn): void;
  linkNativeTurn?(id:string,graphTurn:string):void;
  recordGraphNode?(event:Record<string,unknown>):void;
}

let source: () => TurnRecorder | undefined = () => undefined;

/** Install where the active recorder comes from (the learning module does this). */
export function setRecorderSource(next: () => TurnRecorder | undefined): void { source = next; }

/** The recorder of the current asynchronous context, if any. */
export function activeRecorder(): TurnRecorder | undefined { return source(); }

let adapterSource: () => readonly AdapterBinding[] | undefined = () => undefined;

/** Install where the active adapters come from (the learning module's `withAdapters`). */
export function setAdapterSource(next: () => readonly AdapterBinding[] | undefined): void { adapterSource = next; }

/** Adapters active in the current asynchronous context; a Neuralese driver sends them with every request. */
export function activeAdapters(): readonly AdapterBinding[] { return adapterSource() ?? []; }

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
  linkNativeTurn(id:string,graphTurn:string):void{
    if(!this.turns.some(turn=>turn.native?.id===id))throw Object.assign(new Error('neuralese-native-grad-invalid: native graph link has no exact recorded model turn'),{code:'neuralese-native-grad-invalid'});
    const previous=this.nativeLinks.get(id);
    if(previous&&previous!==graphTurn)throw Object.assign(new Error('neuralese-native-grad-invalid: native receipt was linked to two model-turn occurrences'),{code:'neuralese-native-grad-invalid'});
    this.nativeLinks.set(id,graphTurn);
  }
  absorb(other:ModelTurnRecorder):void{
    for(const node of other.graphNodes.values())this.recordGraphNode(node);
    for(const id of other.graphConflicts)this.graphConflicts.add(id);
    for(const [id,turn] of other.nativeLinks)this.nativeLinks.set(id,turn);
  }
  nativeTurns():RecordedTurn[]{
    if([...this.turns,...this.nested].some(turn=>turn.native)&&this.graphConflicts.size)
      throw Object.assign(new Error('neuralese-native-grad-invalid: native execution graph reuses conflicting source node identities'),{code:'neuralese-native-grad-invalid'});
    return [...this.turns,...this.nested].filter(turn=>turn.native).map(turn=>{
      const graphTurn=this.nativeLinks.get(turn.native!.id);
      if(!graphTurn)return structuredClone(turn);
      const modelNode=this.graphNodes.get(graphTurn);
      if(modelNode?.kind!=='model_turn'||turn.native!.invocation_id&&modelNode.call_id!==turn.native!.invocation_id)
        throw Object.assign(new Error('neuralese-native-grad-invalid: native receipt does not match its actual model-turn occurrence'),{code:'neuralese-native-grad-invalid'});
      const writes:Record<string,string>={},reads:Record<string,string>={};
      for(const event of this.graphNodes.values()){
        if(event.turn!==graphTurn||typeof event.block!=='string'||typeof event.node!=='string')continue;
        if(event.kind==='block_write')writes[event.block]=event.node;
        if(event.kind==='block_read'){
          const inputs=(event.inputs??[]) as {node:string;block?:string}[];
          const sources=inputs.filter(input=>input.block===event.block&&input.node!=='external');
          if(sources.length>1)throw Object.assign(new Error('neuralese-native-grad-invalid: native read has ambiguous actual execution-graph producers'),{code:'neuralese-native-grad-invalid'});
          if(sources[0])reads[event.block]=sources[0].node;
        }
      }
      const binding:NativeTurnGraphBinding={graph_turn:graphTurn,written_nodes:writes,read_bindings:reads};
      return {...structuredClone(turn),native:{...structuredClone(turn.native!),graph:binding}};
    });
  }
  /** Turns recorded since the previous claim. */
  claim(): RecordedTurn[] { const out = this.turns.slice(this.claimed); this.claimed = this.turns.length; return out; }
}
