/**
 * Per-request LoRA scales for llama.cpp servers (managed llama-server, in-page wllama): every loaded adapter is listed,
 * those outside the calling invocation's stack at scale 0, so a request never inherits another call's adapters and
 * the server compares adapter sets exactly when it reuses a cached prefix.
 */
import { activeNativeAdapters } from '../native/model-recording.js';

/** The `lora` request field for the current invocation's stack; `loraIndex` maps artifact IDs to loaded adapter
 * indices. Read it in the calling invocation's context, before any queueing. */
export function perRequestLora(loraIndex: ReadonlyMap<string, number>, server: string): { lora: { id: number; scale: number }[] } | undefined {
  const active = activeNativeAdapters();
  if (!loraIndex.size) {
    if (active.length) throw new Error(`weights-unavailable: ${server} loaded no tuned adapters, but this call requests ${active.map(item => item.id).join(', ')}`);
    return undefined;
  }
  const scales = new Map<number, number>();
  for (const { id, scale } of active) {
    const index = loraIndex.get(id);
    if (index === undefined) throw new Error(`weights-unavailable: adapter ${id} is not loaded by ${server}; rerun natlang tune or restart`);
    scales.set(index, (scales.get(index) ?? 0) + scale);
  }
  return { lora: [...loraIndex.values()].map(index => ({ id: index, scale: scales.get(index) ?? 0 })) };
}
