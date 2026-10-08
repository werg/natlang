/**
 * The compaction task's crisp formats equal pi-durable's own: the summarized messages, the serialized conversation,
 * the summarizer's prompt text, and the summary entry. Run: node --test applications/pi/test/compaction-transcript.test.mjs
 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { serializeConversation as piSerialize, summarizedMessages as piSummarized } from '../vendor/durable/src/harness/compaction.ts';
import { serializeConversation, summarizedMessages, summaryMessages, SUMMARIZATION_PROMPT, SUMMARIZATION_SYSTEM_PROMPT } from '../compaction/summarize/transcript.ts';
import { summarizedMessages as selectSummarized } from '../compaction/select/messages.ts';
import summaryEntry from '../compaction/summaryEntry.ts';

let seed = 11;
const random = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
const pick = items => items[Math.floor(random() * items.length)];
let calls = 0;

function message(pending) {
  const kind = pick(['user', 'user-string', 'assistant', 'tools', 'result', 'system']);
  const t = Math.floor(random() * 1000);
  if (kind === 'user') return { role: 'user', content: [{ type: 'text', text: pick(['hi', '', 'fix it\nplease']) }, { type: 'image', data: 'AA', mimeType: 'image/png' }], timestamp: t };
  if (kind === 'user-string') return { role: 'user', content: pick(['plain', '']), timestamp: t };
  if (kind === 'system') return { role: 'system', content: '', sections: { a: 'x' }, timestamp: t };
  if (kind === 'result' && pending.length) {
    const call = pending.splice(Math.floor(random() * pending.length), 1)[0];
    return { role: 'toolResult', toolCallId: call.id, toolName: call.name, content: [{ type: 'text', text: pick(['ok', '', 'y'.repeat(2500)]) }], isError: false, timestamp: t };
  }
  const content = [];
  if (random() < 0.4) content.push({ type: 'thinking', thinking: 'hmm' });
  if (random() < 0.7) content.push({ type: 'text', text: pick(['done', 'a\nb']) });
  if (kind === 'tools') for (let i = 0; i < 1 + Math.floor(random() * 3); i++) {
    const call = { type: 'toolCall', id: `c${calls++}`, name: pick(['read', 'bash']), arguments: { path: 'x.js', n: 3, deep: { a: [1] } } };
    content.push(call); pending.push(call);
  }
  return { role: 'assistant', content, api: 'faux', provider: 'faux', model: 'm', usage: {}, stopReason: kind === 'tools' ? 'toolUse' : 'stop', timestamp: t };
}

function view() {
  const pending = [];
  const contributions = Array.from({ length: 1 + Math.floor(random() * 8) }, () => Array.from({ length: Math.floor(random() * 3) }, () => message(pending)));
  return { contributions };
}

test('summarized messages and serialization equal pi', () => {
  for (let i = 0; i < 1000; i++) {
    const v = view();
    const cut = Math.floor(random() * (v.contributions.length + 1));
    const expected = piSummarized(v, cut);
    assert.deepEqual(summarizedMessages(v, cut), expected);
    assert.deepEqual(selectSummarized(v, cut), expected);
    assert.equal(serializeConversation(expected), piSerialize(expected));
  }
});

test('the prompts are pi\'s verbatim, and the request and entry have pi\'s shape', () => {
  const source = readFileSync(new URL('../vendor/durable/src/harness/compaction.ts', import.meta.url), 'utf8');
  assert.ok(source.includes('`' + SUMMARIZATION_SYSTEM_PROMPT + '`'));
  assert.ok(source.includes('`' + SUMMARIZATION_PROMPT + '`'));
  const v = view();
  const [system, user] = summaryMessages(v, v.contributions.length, 'keep paths', 5);
  assert.deepEqual(system, { role: 'system', content: SUMMARIZATION_SYSTEM_PROMPT, timestamp: 5 });
  assert.equal(user.content[0].text, `<conversation>\n${piSerialize(piSummarized(v, v.contributions.length))}\n</conversation>\n\n${SUMMARIZATION_PROMPT}\n\nAdditional focus: keep paths`);
  assert.ok(!summaryMessages(v, 0, null, 5)[1].content[0].text.includes('Additional focus'));
  assert.deepEqual(summaryEntry('S', 4, 'threshold', 9), { kind: 'pi.compaction', head: 4, data: { reason: 'threshold' },
    model: [{ role: 'user', content: [{ type: 'text', text: 'The conversation history before this point was compacted into the following summary:\n\n<summary>\nS\n</summary>' }], timestamp: 9 }] });
});
