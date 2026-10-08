import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';

const shim = fileURLToPath(new URL('./harness-shim.ts', import.meta.url));
const testing = fileURLToPath(new URL('../../vendor/durable/src/testing/index.ts', import.meta.url));

/** pi-durable's harness suites against the port: `npx vitest --run --config test/conformance/vitest.config.ts`. */
export default defineConfig({
  test: {
    environment: 'node',
    // pi's waitFor helper allows 5 s, sized for crisp tasks; the port's phases are model calls.
    env: { PI_WAIT_MS: process.env.PI_WAIT_MS ?? '3600000' },
    root: fileURLToPath(new URL('../../vendor/durable', import.meta.url)),
    include: ['test/harness-{generation,generation-recovery,compaction,context,prompt,inbox,submissions,tools,tools-recovery,structured,tasks}.test.ts'],
    testTimeout: 7_200_000,
    hookTimeout: 600_000,
    fileParallelism: process.env.PI_FILE_PARALLEL === "1",
    maxWorkers: Number(process.env.PI_WORKERS ?? 1),
  },
  resolve: {
    conditions: ['source'],
    alias: [
      { find: /^@earendil-works\/pi-durable$/, replacement: shim },
      { find: /^@earendil-works\/pi-durable\/testing$/, replacement: testing },
    ],
  },
  ssr: { resolve: { conditions: ['source'] } },
});
