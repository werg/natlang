/**
 * pi-durable's public module with `Harness.open` replaced: the registry it is given resolves the built-in task names to
 * the port's natural-language task kinds, run by a real executor. Aliased as "@earendil-works/pi-durable" in
 * vitest.config.ts, so pi-durable's own harness suites test the port.
 *
 * Executor: PI_EXECUTOR_ENDPOINT (default http://127.0.0.1:8083) and PI_EXECUTOR_MODEL.
 */
import { fileURLToPath } from 'node:url';
import * as durable from '../../vendor/durable/src/index.ts';
import type { HarnessOptions, Storage } from '../../vendor/durable/src/index.ts';
import type { Context } from '@earendil-works/chord';
// @ts-ignore: the runtime's build has no types for this relative import
import { createNatlangRuntime, fileTraceSink, loadNatlang, openAICompatibleModelTurn } from '../../../../ts-host/dist/index.js';
import { naturalLanguageTask, type Entry, type TaskHost } from '../../host/tasks.ts';
import { substituteTasks } from '../../host/registry.ts';

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
export const phaseLog: string[] = [];

export const Harness = {
  async open(storage: Storage, options: HarnessOptions, context: Context) {
    let harness: durable.Harness | undefined;
    const host: TaskHost = {
      natlang, attempts: 2,
      onPhase: event => { phaseLog.push(`${event.kind}#${event.taskId} ${event.phase}: ${event.error ?? event.summary}`); if (process.env.PI_PHASE_LOG) console.error(phaseLog.at(-1)); },
      implementation: point => (process.env[`PI_${point.toUpperCase()}`] as 'crisp' | 'natural-language' | undefined) ?? 'crisp',
      async submit(conversationId, draft, submitContext) {
        const conversation = await harness!.conversation(conversationId, submitContext);
        return (await conversation!.submit(draft as never, submitContext)).id;
      },
      async submission(id, submissionContext) {
        const handle = await harness!.submission(id as never, submissionContext);
        return handle ? JSON.parse(JSON.stringify(await handle.status(submissionContext))) : undefined;
      },
      async scan() { throw new Error('scan is not wired in the conformance shim'); },
    };
    const tasks = [
      naturalLanguageTask(host, 'pi.generation', 1, () => ({ phase: 'prepare', attempt: 1 }), ['prepare', 'request', 'retry', 'poll', 'tools'], entry('generation')),
      naturalLanguageTask(host, 'pi.tool', 1, () => ({ phase: 'call' }), ['call', 'execute'], entry('tool')),
      naturalLanguageTask(host, 'pi.compaction', 1, () => ({ phase: 'select' }), ['select', 'summarize', 'retry'], entry('compaction')),
    ];
    harness = await durable.Harness.open(storage, { ...options, registry: substituteTasks(options.registry, tasks) }, context);
    return harness;
  },
};
