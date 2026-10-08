import assert from 'node:assert/strict';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { test } from 'node:test';

const distRoot = process.env.NATLANG_TS_HOST_TEST_DIST;
const moduleUrl = distRoot
  ? pathToFileURL(resolve(distRoot, 'teacher/request-telemetry.js')).href
  : new URL('../dist/teacher/request-telemetry.js', import.meta.url).href;
const { CollectorRequestTelemetryAccumulator } = await import(moduleUrl);

const start = (attempt_id, purpose, extra = {}) => ({ attempt_id, purpose, status: 'completed',
  chat_transport_starts: 1, chat_transport_retry_starts: 0, provider_sdk_turn_starts: 1, ...extra });

test('batch telemetry separates fresh/resumed rows, reused lineage, and missing telemetry', () => {
  const accumulator = new CollectorRequestTelemetryAccumulator();
  accumulator.add({ request_telemetry: { attempt_ids: ['fresh-a', 'fresh-b'], starts: [
    start('fresh-a', 'planner', { planner_attempt: 1 }),
    start('fresh-b', 'action', { plan_status: 'planned', chat_transport_retry_starts: 1 }),
    start('fresh-b', 'judge', { role: 'judge' }),
  ], authored_synthetic_root_actions: 1 } });
  accumulator.add({ provenance: { reused_from: { path: '/prior/run.jsonl' } }, request_telemetry: {
    attempt_ids: ['old-a'], starts: [start('old-a', 'action', { plan_status: 'fallback' })],
  } });
  accumulator.add({ task: { program_ir: { id: 'legacy-without-telemetry' } } });

  assert.deepEqual(accumulator.totals, {
    rows_with_telemetry: 2, rows_without_telemetry: 1,
    rows_reused_with_telemetry: 1, rows_fresh_or_resumed_with_telemetry: 1,
    collector_attempts: 3, collector_retry_attempts: 1,
    collector_sender_starts: 4, fresh_or_resumed_sender_starts: 3, reused_lineage_sender_starts: 1,
    planner_starts: 1, planner_retry_starts: 0, action_starts: 2, judge_starts: 1,
    chat_transport_starts: 4, chat_transport_retry_starts: 1, provider_sdk_turn_starts: 4,
    planner_chat_transport_starts: 1, action_chat_transport_starts: 2, judge_chat_transport_starts: 1,
    planner_provider_sdk_turn_starts: 1, action_provider_sdk_turn_starts: 2, judge_provider_sdk_turn_starts: 1,
    planned_action_turns: 1, planner_fallback_turns: 1,
    sender_completed: 4, sender_failed: 0, authored_synthetic_root_actions: 1,
  });
});
