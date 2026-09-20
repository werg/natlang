import { randomUUID } from 'node:crypto';
import { TypeScriptEnvironment } from '../environment.js';
import type { ModelTurnRequest, ModelTurn } from '../runtime.js';
import { NativeToolAgent, type NativeReviewOptions } from './agent.js';
import { checkedDefinitions, type NativeDefinition } from './codebase.js';
import { NativeRuntime } from './runtime.js';
import { fitsType, formatType, parseType, TypeEnv } from './types.js';
import { dump } from './values.js';

export type NativeChildResult = { source_revision: string; parent_call_id: string | null;
  outcome: string; value: unknown; trace: Record<string, unknown>[] };

/** Versioned, immutable source graph for explicit child invocations from a TypeScript application. */
export class NativeSourceWorkspace {
  private readonly definitions: Record<string, NativeDefinition>;
  readonly root: string;
  readonly revision: string;
  private active = 0;

  constructor(definitions: Record<string, NativeDefinition>, root: string) {
    this.definitions = structuredClone(definitions);
    this.root = root;
    this.revision = checkedDefinitions(this.definitions, root).revision;
  }

  edited(name: string, definition: NativeDefinition): NativeSourceWorkspace {
    return new NativeSourceWorkspace({ ...this.definitions, [name]: structuredClone(definition) }, this.root);
  }

  describe(name = this.root) {
    const definition = this.definitions[name];
    if (!definition) throw new RangeError(`unknown source function ${name}`);
    const args = Object.entries(definition.args ?? {}).map(([key, type]) => `${key}: ${type}`).join(', ');
    return { name, signature: `${name}(${args}) -> ${definition.returns}`,
      kind: definition.code === undefined ? 'instructions' : 'code',
      engine: definition.code === undefined ? null : definition.engine ?? 'quickjs-isolated',
      functions: Object.keys(definition.uses ?? {}).sort(), revision: this.revision };
  }

  typeCheck(actual: string, expected: string) {
    const a = parseType(actual), b = parseType(expected);
    return { actual: formatType(a), expected: formatType(b), fits: fitsType(a, b, new TypeEnv()) };
  }

  async invoke(name: string, inputs: Record<string, unknown> = {}, options: {
    modelTurn?: (request: ModelTurnRequest) => Promise<ModelTurn> | ModelTurn;
    review?: NativeReviewOptions;
    parentCallId?: string; maxEpisodes?: number; maxDepth?: number;
    seedPolicy?: { mode: 'compatibility' | 'derived' | 'backend'; root?: number };
    capabilities?: Record<string, (args: unknown[]) => unknown>;
    environment?: TypeScriptEnvironment;
    signal?: AbortSignal; timeoutMs?: number;
  } = {}): Promise<NativeChildResult> {
    if (this.active >= 8) throw new RangeError('nested source invocation depth exceeded');
    const maxEpisodes = options.maxEpisodes ?? 32;
    if (!Number.isInteger(maxEpisodes) || maxEpisodes < 1 || maxEpisodes > 32)
      throw new RangeError('child episode budget must be between 1 and 32');
    const graph = checkedDefinitions(this.definitions, name);
    const agent = options.modelTurn ? new NativeToolAgent(options.modelTurn, { review: options.review }) : undefined;
    const environment = options.environment ?? new TypeScriptEnvironment();
    const runtime = new NativeRuntime({ environment, agent: agent ? session => agent.run(session) : undefined,
      capabilities: options.capabilities, maxEpisodes, maxDepth: options.maxDepth,
      seedPolicy: options.seedPolicy, sourceRevision: graph.revision, parentCallId: options.parentCallId,
      runId: randomUUID(), signal: options.signal, timeoutMs: options.timeoutMs });
    this.active++;
    try {
      const result = await runtime.runRoot(graph.instantiate(inputs));
      return { source_revision: graph.revision, parent_call_id: options.parentCallId ?? null,
        outcome: result.outcome.kind, value: result.outcome.kind === 'done' ? dump(result.value) : null,
        trace: runtime.trace.events as Record<string, unknown>[] };
    } finally {
      this.active--;
      runtime.close();
      if (!options.environment) environment.close();
    }
  }
}
