import assert from 'node:assert/strict';
import { visibleTurnText, visibleValue } from './lib/decision-child-visibility.mjs';

const options = ['order_status', 'oos'];
const priorInput = {
  messages: [{ role: 'tool', content: `input: ${JSON.stringify({ state: 'check my order', options })}` }],
  target: { tool_calls: [{ function: { arguments: '"options": ["target_only"]' } }] },
};
assert.ok(visibleTurnText(priorInput).includes(JSON.stringify(options)),
  'a prior tool response rendered into the conversation is visible input');

const targetOnly = {
  messages: [{ role: 'user', content: 'Classify the request.' }],
  target: { tool_calls: [{ function: { arguments: JSON.stringify({
    code: `const options: unknown = ${JSON.stringify(options)}; return 'order_status';`,
  }) } }] },
};
assert.equal(visibleTurnText(targetOnly).includes(JSON.stringify(options)), false,
  'current target arguments cannot establish input visibility');
const assistantCallOnly = { messages: [{ role: 'assistant', tool_calls: [{ function: {
  arguments: JSON.stringify({ options, code: 'return options;' }),
} }] }] };
assert.equal(visibleValue(options, visibleTurnText(assistantCallOnly)), true,
  'a prior assistant tool call present in the prompt history is model-visible context');
assert.equal(visibleValue(options, 'options: ["order_status", "oos"]'), true,
  'the complete structured field is visible when its serialized values are present');
assert.equal(visibleValue(options, '[<<fields: order_status, oos; values not shown>>]'), false,
  'summaries cannot stand in for the complete structured field');
assert.equal(visibleValue(options, 'options: ["order_status"] plus a separate oos mention'), false,
  'a subset of the structured values is not complete input visibility');
