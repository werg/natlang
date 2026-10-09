import assert from 'node:assert/strict';
import { test } from 'node:test';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { spawnSync } from 'node:child_process';
import { loadSourceDerivedRepairCandidates, sourceDerivedRepairReviewPair } from '../scripts/source-derived-repair-pairs.mjs';
import { renderPreferencePair } from '../scripts/export-preference-pairs.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const proposalPath = path.join(root, 'runs/luna-authored-root-return-guidance-fivecase-20261009-v1/evidence/leaf-eligibility-repair-audit-v1/causal-boundary-repairs-v5.json');

const cpuTemplate = async (messages, tools) => {
  const rendered = [`<|tools|>${JSON.stringify(tools)}<|messages|>`];
  for (const message of messages) {
    const calls = (message.tool_calls ?? []).map(call =>
      `<|call|>${call.function.name}\n${call.function.arguments}<|call_end|>`).join('');
    rendered.push(`<|${message.role}|>${message.content ?? ''}${calls}<|end|>`);
  }
  return rendered.join('') + '<|assistant|>';
};

test('shared preference exporter renders held diagnostics with exact program/group and assistant target boundaries', async () => {
  const { items } = await loadSourceDerivedRepairCandidates(proposalPath);
  const reviews = items.map(sourceDerivedRepairReviewPair);
  for (const review of reviews) {
    assert.equal(review.status, 'held');
    const pair = review.pair;
    const rendered = await renderPreferencePair(pair, cpuTemplate, '<|end|>');
    const sourceItem = items.find(item => item.candidate_id === pair.evidence.proposal_item_id);
    assert.equal(rendered.program_id, sourceItem.source.program_id);
    assert.equal(rendered.split, 'train');
    assert.deepEqual(rendered.source_groups, sourceItem.source.source_groups);
    assert.equal(rendered.evidence.terminal_tool_call_id, sourceItem.provider_request.terminal_tool_call_id);
    assert.ok(rendered.chosen.startsWith('<|call|>'));
    assert.ok(rendered.rejected.startsWith('<|call|>'));
    assert.ok(rendered.chosen.endsWith('<|end|>'));
    assert.ok(rendered.rejected.endsWith('<|end|>'));
    const targetCalls = sourceItem.preference_pair_candidate.chosen.target.tool_calls;
    assert.equal(rendered.chosen.includes(targetCalls[0].function.name), true);
    assert.equal(rendered.prompt, (await renderPreferencePair(pair, cpuTemplate, '<|end|>')).prompt);
    const jsonlRoundTrip = JSON.parse(JSON.stringify(rendered));
    assert.equal(jsonlRoundTrip.chosen, rendered.chosen);
    assert.equal(jsonlRoundTrip.rejected, rendered.rejected);
    if (sourceItem.candidate_id !== 'BIR-2C') {
      assert.ok(rendered.chosen.includes('\\n'), 'embedded eval source newline stays JSON escaped in the target');
      assert.equal(rendered.chosen.includes('\nreturn '), false, 'eval source newline is not emitted as a raw JSON-line break');
    }
  }
});

test('trainable exporter CLI continues to reject the held diagnostic pair', async () => {
  const { items } = await loadSourceDerivedRepairCandidates(proposalPath);
  const review = sourceDerivedRepairReviewPair(items[0]);
  const directory = await mkdtemp(path.join(tmpdir(), 'held-preference-export-'));
  try {
    const input = path.join(directory, 'held.jsonl');
    const output = path.join(directory, 'must-not-exist.jsonl');
    await writeFile(input, `${JSON.stringify({ ...review.pair, status: 'held', training_admission: false })}\n`);
    const result = spawnSync(process.execPath, ['scripts/export-preference-pairs.mjs', input, output],
      { cwd: path.resolve(root, 'ts-host'), encoding: 'utf8' });
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /held or unsupported preference records cannot be exported/);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
