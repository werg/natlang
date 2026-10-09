/**
 * pi-durable's public module with `Harness.open` replaced by the port's (host/harness.ts portOptions): the built-in
 * task names resolve to the natural-language task kinds, and PI_CONTEXT / PI_SCHEDULER / PI_ADMISSION
 * ("natural-language") select the pluggable implementations. Aliased as "@earendil-works/pi-durable" in
 * vitest.config.ts, so pi-durable's own harness suites test the port.
 *
 * Executor: PI_EXECUTOR_ENDPOINT (default http://127.0.0.1:8083), PI_EXECUTOR_MODEL, PI_EXECUTOR_CONCURRENCY;
 * PI_TRACE_DIR keeps the natlang traces; PI_PHASE_LOG prints each phase.
 */
import { readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as durable from '../../vendor/durable/src/index.ts';
import type { HarnessOptions, Storage } from '../../vendor/durable/src/index.ts';
import type { Context } from '@earendil-works/chord';
// @ts-ignore: the runtime's build has no types for this relative import
import { createNatlangRuntime, fileTraceSink, loadNatlang, openAICompatibleModelTurn } from '../../../../ts-host/dist/index.js';
import type { Entry } from '../../host/tasks.ts';
import { portOptions, type Implementation } from '../../host/harness.ts';

export * from '../../vendor/durable/src/index.ts';

const root = fileURLToPath(new URL('../../', import.meta.url));
const natlang = createNatlangRuntime({
  model: { driver: openAICompatibleModelTurn({ endpoint: process.env.PI_EXECUTOR_ENDPOINT ?? 'http://127.0.0.1:8083',
    model: process.env.PI_EXECUTOR_MODEL ?? 'nvidia/Qwen3.6-35B-A3B-NVFP4', concurrency: Number(process.env.PI_EXECUTOR_CONCURRENCY ?? 4) }),
    // The context budget follows the window the executor's server reports unless PI_EXECUTOR_CONTEXT sets it.
    ...(process.env.PI_EXECUTOR_CONTEXT ? { contextTokens: Number(process.env.PI_EXECUTOR_CONTEXT) } : {}) },
  codeEdits: 'deny',
  // Calls are recorded in the machine's call store under the port's directory, where offline work reloads them.
  programRoot: root.replace(/\/$/, ''),
  ...(process.env.PI_TRACE_DIR ? { trace: fileTraceSink(process.env.PI_TRACE_DIR) } : {}),
});
/** The newest modification time of the port's own .nl and .ts files (not vendor/, node_modules/ or build output). */
function sourceStamp(dir = root): number {
  let newest = 0;
  for (const item of readdirSync(dir, { withFileTypes: true })) {
    if (item.name.startsWith('.') || item.name === 'vendor' || item.name === 'node_modules' || item.name === 'test') continue;
    const path = join(dir, item.name);
    if (item.isDirectory()) newest = Math.max(newest, sourceStamp(path));
    else if (/\.(nl|ts)$/.test(item.name) && !item.name.endsWith('.d.nl.ts')) newest = Math.max(newest, statSync(path).mtimeMs);
  }
  return newest;
}

/**
 * A phase entry that reloads its function (and the functions of its folder) when the port's sources changed, so a
 * long run uses the newest instructions from its next phase on (owner rule: newest code always).
 */
const entry = (name: string): Entry => {
  let loaded = loadNatlang(`${root}${name}.nl`, root) as unknown as Entry, stamp = sourceStamp();
  return ((...args: Parameters<Entry>) => {
    const now = sourceStamp();
    if (now !== stamp) {
      try { loaded = loadNatlang(`${root}${name}.nl`, root) as unknown as Entry; stamp = now; }
      catch (error) { console.error(`pi conformance: keeping the loaded ${name}.nl, the changed sources do not load: ${error instanceof Error ? error.message : String(error)}`); }
    }
    return loaded(...args);
  }) as Entry;
};
const entries = { generation: entry('generation'), tool: entry('tool'), compaction: entry('compaction') };
const pick = (name: string): Implementation => process.env[name] === 'natural-language' ? 'natural-language' : 'crisp';
export const phaseLog: string[] = [];

export const Harness = {
  async open(storage: Storage, options: HarnessOptions, context: Context) {
    const port = portOptions(options, {
      natlang, entries, attempts: 2,
      implementations: { context: pick('PI_CONTEXT'), scheduler: pick('PI_SCHEDULER'), admission: pick('PI_ADMISSION'), planning: pick('PI_PLANNING') },
      onPhase: event => {
        phaseLog.push(`${event.kind}#${event.taskId} ${event.phase}: ${event.error ?? event.summary}`);
        if (process.env.PI_PHASE_LOG) console.error(phaseLog.at(-1));
      },
    });
    const harness = await durable.Harness.open(storage, port.options, context);
    port.bind(harness);
    return harness;
  },
};
