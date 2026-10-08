import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';

const shim = fileURLToPath(new URL('./harness-shim.ts', import.meta.url));
const testing = fileURLToPath(new URL('../../vendor/durable/src/testing/index.ts', import.meta.url));

/** pi-durable's harness suites against the port: `npx vitest --run --config test/conformance/vitest.config.ts`. */
export default defineConfig({
  test: {
    environment: 'node',
    root: fileURLToPath(new URL('../../vendor/durable', import.meta.url)),
    include: ['test/harness-{generation,generation-recovery,compaction,context,prompt,inbox,submissions,tools,tools-recovery,structured,tasks}.test.ts'],
    testTimeout: 1_800_000,
    hookTimeout: 600_000,
    fileParallelism: false,
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
