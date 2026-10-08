/**
 * pi-durable's public module with `Harness.open` replaced by the port's (host/harness.ts portOptions): the built-in
 * task names resolve to the natural-language task kinds, and PI_CONTEXT / PI_SCHEDULER / PI_ADMISSION
 * ("natural-language") select the pluggable implementations. Aliased as "@earendil-works/pi-durable" in
 * vitest.config.ts, so pi-durable's own harness suites test the port.
 *
 * Executor: PI_EXECUTOR_ENDPOINT (default http://127.0.0.1:8083), PI_EXECUTOR_MODEL, PI_EXECUTOR_CONCURRENCY;
 * PI_TRACE_DIR keeps the natlang traces; PI_PHASE_LOG prints each phase.
 */
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
    contextTokens: Number(process.env.PI_EXECUTOR_CONTEXT ?? 57344) },
  codeEdits: 'deny',
  ...(process.env.PI_TRACE_DIR ? { trace: fileTraceSink(process.env.PI_TRACE_DIR) } : {}),
});
const entry = (name: string) => loadNatlang(`${root}${name}.nl`, root) as unknown as Entry;
const entries = { generation: entry('generation'), tool: entry('tool'), compaction: entry('compaction') };
const pick = (name: string): Implementation => process.env[name] === 'natural-language' ? 'natural-language' : 'crisp';
export const phaseLog: string[] = [];

export const Harness = {
  async open(storage: Storage, options: HarnessOptions, context: Context) {
    const port = portOptions(options, {
      natlang, entries, attempts: 2,
      implementations: { context: pick('PI_CONTEXT'), scheduler: pick('PI_SCHEDULER'), admission: pick('PI_ADMISSION') },
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
