/**
 * The Neuralese instance of the builtin `view(value, instructions?)` on a server (plans/neuralese/DECISIONS.md
 * 2026-10-09: "one summarizer family" and "representation chosen by use"). A view is a template write of view's body:
 * the model reads the value at view's write site, with the body as its system text, and its reply is forced to
 * `return_result` with the value written as a block; the stop head decides the length. Without instructions the view
 * is faithful compression, with them it keeps what their purpose needs.
 *
 * The server owns the plan (training/neuralese/natlang_neuralese/view.py, `POST /v1/neuralese/view`): a value that fits
 * the write site's window (by default the model's whole context) is viewed in one write; a longer one in token chunks
 * whose views a final write combines. The trainer writes through the same site and plan, so what is trained is what
 * runs. The runtime's opening listing shows such a view for each argument it would cut off, with the receiving call's
 * instructions as the view's (prompt.ts `listingViewInstructions`).
 *
 * The body is the prompt piece `view`; under a system-prompt bank its soft form is sent instead.
 */
import { VIEW_PROMPT } from '../builtin/index.js';
import { neuraleseRef, textToParts, type NeuraleseRef } from '../native/neuralese.js';
import type { NeuraleseStore } from '../native/neuralese-store.js';
import { activeSystemPrompts, softenText, type SystemPromptBank } from '../native/system-prompts.js';
import { fetchModel } from '../model/chat-completion.js';
import { neuraleseServerInfo, serves } from '../model/neuralese-info.js';
import { requestNativeStateIds } from '../model/neuralese-server.js';
import { currentFrame } from '../runtime/context.js';
import { graphNode, traceFor, blockInput } from '../native/graph.js';
import { archiveNativeState, restoreNativeInputs } from '../model/native-state-client.js';
import { HttpNeuraleseStore, withRestoredBlocks } from '../model/neuralese-server.js';

/** The type of a written view: the Neuralese instance of view's result. */
export const VIEW_TYPE = 'Neuralese<string>';

/** The arguments of one view call: the value, and what the view is for (absent: faithful). */
export type ViewSite = { value: string; instructions?: string };
/** Runs view's Neuralese instance on one value, or answers undefined to keep the crisp preview. */
export type NativeViewTurn = { messages: unknown[]; tools: unknown[]; template: Record<string, unknown>;
  response: { x_natlang_rl_capture: { schema: string; policy: Record<string, string>; prompt: { kind: string; token_ids?: number[] }[]; generated_inputs: { token_id: number; input_role: string }[]; writes: { state_id: string; native_identity: Record<string, string>; payload_own_begin: number; payload_own_end: number }[] }; native_state: { blocks: { id: string }[] } } };
export type Viewer = (site: ViewSite) => Promise<NeuraleseRef | undefined>;

/** View's Neuralese instance on a Neuralese server, with the soft body from `bank` when it has it. `window` caps the
 * value tokens of one write site (default: the server's, from the model's context). */
export function serverViewer(options: { endpoint: string; headers?: Record<string, string>; store?: NeuraleseStore;
  bank?: SystemPromptBank; window?: number; onSourceTurns?: (turns: NativeViewTurn[]) => void | Promise<void> }): Viewer {
  const remote = new HttpNeuraleseStore(options.endpoint, options.headers);
  const base = options.endpoint.replace(/\/$/, '');
  const info = neuraleseServerInfo(options.endpoint, { headers: options.headers });
  return async site => {
    const bank = activeSystemPrompts(options.bank);
    const softened = bank?.size ? softenText(VIEW_PROMPT, bank) : VIEW_PROMPT;
    const system = softened === VIEW_PROMPT ? VIEW_PROMPT : textToParts(softened);
    const server = await info;
    if ((server.native_state as { enabled?: boolean } | undefined)?.enabled) {
      if (!serves(server, 'view')) throw Object.assign(new Error('neuralese-view-unavailable: use a native server with the shared view stage'), { code: 'neuralese-view-unavailable' });
      if (!options.store?.putNativeState || !options.store.getNativeState)
        throw new Error('neuralese-native-archive-unavailable: native views need a native-capable archive');
      const client = { endpoint: options.endpoint, headers: options.headers, store: options.store, info: server };
      await restoreNativeInputs(client, system);
      const response = await fetchModel(`${base}/v1/neuralese/view`, { method: 'POST',
        headers: { 'content-type': 'application/json', ...options.headers },
        body: JSON.stringify({ value: site.value, ...(site.instructions ? { instructions: site.instructions } : {}), system,
          ...(options.window ? { window: options.window } : {}) }) });
      if (!response.ok) throw new Error(`native view failed (${response.status}): ${(await response.text()).slice(0, 1000)}`);
      const result = await response.json() as { id: string; source_turns?: NativeViewTurn[] };
      if (!Array.isArray(result.source_turns) || !result.source_turns.length)
        throw new Error('neuralese-view-source-integrity: native views must preserve every actual chunk and combine turn');
      const trace = traceFor(currentFrame()?.parentCallId);
      const written = new Set<string>();
      for (const [index, turn] of result.source_turns.entries()) {
        const receipt = turn.response?.x_natlang_rl_capture;
        if (receipt?.schema !== 'natlang.native-rl-sampling/2' || !Array.isArray(receipt.writes))
          throw new Error('neuralese-view-source-integrity: native view turn lacks exact sampled producer provenance');
        const imports = requestNativeStateIds(turn.messages);
        for (const id of imports) await archiveNativeState(client, id);
        const node = graphNode(trace, 'model_turn', { turn: index + 1, calls: ['return_result'],
          temperature: 1, seed: null, source_kind: 'native-view', source_turn: turn },
          imports.map(id => blockInput(id, 'native-state')));
        for (const write of receipt.writes) {
          const artifact = await archiveNativeState(client, write.state_id);
          for (const key of ['checkpoint_sha256', 'tokenizer_fingerprint', 'embedding_weight_sha256', 'architecture'] as const)
            if (artifact.metadata[key] !== write.native_identity[key] || artifact.metadata[key] !== receipt.policy[key])
              throw new Error('neuralese-view-source-integrity: view output identity differs from its actual producer');
          const own = receipt.prompt.flatMap(segment => segment.kind === 'tokens' ? segment.token_ids ?? [] : []);
          own.push(...receipt.generated_inputs.map(input => input.token_id));
          const start = write.payload_own_begin, stop = write.payload_own_end;
          const payload = artifact.tensors.token_ids!;
          const values = new DataView(payload.data.buffer, payload.data.byteOffset, payload.data.byteLength);
          const markers = (server.native_state as { role_marker_ids: number[] }).role_marker_ids;
          if (!Number.isSafeInteger(start) || !Number.isSafeInteger(stop) || start < 1 || stop < start ||
              stop >= own.length || stop - start !== payload.shape[0] || own[start - 1] !== markers[0] || own[stop] !== markers[1] ||
              own.slice(start, stop).some((token, index) => BigInt(token) !== values.getBigInt64(index * 8, true)))
            throw new Error('neuralese-view-source-integrity: actual producer payload differs from the archived view');
          written.add(write.state_id);
          graphNode(trace, 'block_write', { block: write.state_id, format: 'native-state',
            length: artifact.tensors.token_ids!.shape[0], width: artifact.tensors.embeddings!.shape[1],
            truncated: false, stops: [], native_identity: write.native_identity, source_turn: turn },
            node ? [{ node, port: 'turn' }] : []);
        }
      }
      if (!written.has(result.id)) throw new Error('neuralese-view-source-integrity: final view is not an actual output of the preserved turns');
      await options.onSourceTurns?.(result.source_turns);
      return neuraleseRef(VIEW_TYPE, result.id);
    }
    const ids = Array.isArray(system) ? system.flatMap(part => part.type === 'neuralese' ? [part.id] : []) : [];
    const meta = await withRestoredBlocks(remote, options.store, ids, async () => {
      const response = await fetchModel(`${base}/v1/neuralese/view`, { method: 'POST',
        headers: { 'content-type': 'application/json', ...options.headers },
        body: JSON.stringify({ value: site.value, ...(site.instructions ? { instructions: site.instructions } : {}), system,
          ...(options.window ? { window: options.window } : {}) }) });
      if (!response.ok) throw new Error(`view failed (${response.status}): ${(await response.text()).slice(0, 300)}`);
      return await response.json() as { id: string };
    });
    if (options.store && !(await options.store.has(meta.id))) {
      const block = await remote.get(meta.id);
      if (!block) throw new Error(`neuralese-unknown-block: the server wrote ${meta.id} but does not have it`);
      const { id: _, ...rest } = block.meta;
      await options.store.put({ ...rest, data: block.data });
    }
    return neuraleseRef(VIEW_TYPE, meta.id);
  };
}
