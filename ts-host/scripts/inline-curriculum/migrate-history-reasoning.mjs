#!/usr/bin/env node
/**
 * Give the assistant turns in recorded contexts the reasoning they were made with, as the agent now sends them.
 *
 *   node scripts/inline-curriculum/migrate-history-reasoning.mjs IN.results.jsonl OUT.results.jsonl
 *
 * Contexts used to go back without reasoning, which a template that keeps earlier thinking shows as empty
 * <think></think> blocks; trained on such contexts, a model learns to skip thinking. An earlier decision is found by
 * its tool call ids (a text reply by its text) among the row's own decisions; the runtime's opening turns get the
 * agent's opening thoughts. request_sha256 still names the request as it was sent. Any row can be passed again; each
 * rewritten row is marked in provenance.history_reasoning_migration.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { OPENING_THOUGHT, FOLDER_THOUGHT } from '../../dist/native/agent.js';

const MIGRATION = 'reasoning/1';
const OPENING = { scope_0: OPENING_THOUGHT, scope_1: FOLDER_THOUGHT };

/** The row with reasoning restored in its contexts, or the same row when there is nothing to restore. */
export function migrateRow(row) {
  const byCall = new Map(), byText = new Map();
  for (const turn of row.trajectory ?? []) {
    const reasoning = turn.assistant?.reasoning ?? turn.model_response?.reasoning;
    if (!reasoning) continue;
    const ids = (turn.model_response?.raw_calls ?? []).map(call => call.id).filter(Boolean);
    for (const id of ids) byCall.set(id, reasoning);
    if (!ids.length && typeof turn.model_response?.text === 'string' && !byText.has(turn.model_response.text))
      byText.set(turn.model_response.text, reasoning);
  }
  let changed = false;
  const trajectory = (row.trajectory ?? []).map(turn => {
    if (!Array.isArray(turn.context)) return turn;
    const context = turn.context.map(message => {
      if (message.role !== 'assistant' || message.reasoning_content) return message;
      const id = message.tool_calls?.[0]?.id;
      const reasoning = id ? OPENING[id] ?? byCall.get(id) : byText.get(message.content);
      if (!reasoning) return message;
      changed = true;
      return { ...message, reasoning_content: reasoning };
    });
    return changed ? { ...turn, context } : turn;
  });
  if (!changed) return row;
  return { ...row, trajectory, provenance: { ...row.provenance, history_reasoning_migration: MIGRATION } };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const [input, output] = process.argv.slice(2);
  if (!input || !output) throw new Error('usage: migrate-history-reasoning.mjs IN.results.jsonl OUT.results.jsonl');
  const lines = readFileSync(input, 'utf8').split('\n').filter(Boolean);
  let changed = 0;
  const out = lines.map(line => {
    const row = JSON.parse(line), next = migrateRow(row);
    if (next === row) return line;
    changed++;
    return JSON.stringify(next);
  });
  writeFileSync(output, out.join('\n') + '\n');
  process.stdout.write(`${changed}/${lines.length} rows given history reasoning -> ${output}\n`);
}
