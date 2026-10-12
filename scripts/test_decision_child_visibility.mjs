import assert from 'node:assert/strict';
import { visibleTurnText } from './lib/decision-child-visibility.mjs';

const options = ['order_status', 'oos'];
const priorInput = {
  messages: [{ role: 'assistant', tool_calls: [{ function: { arguments: JSON.stringify({
    code: `const options: unknown = ${JSON.stringify(options)};`,
  }) } }] }],
  target: { tool_calls: [{ function: { arguments: '"options": ["target_only"]' } }] },
};
assert.ok(visibleTurnText(priorInput).includes(JSON.stringify(options)),
  'decoded prior tool arguments are visible input');

const targetOnly = {
  messages: [{ role: 'user', content: 'Classify the request.' }],
  target: { tool_calls: [{ function: { arguments: JSON.stringify({
    code: `const options: unknown = ${JSON.stringify(options)}; return 'order_status';`,
  }) } }] },
};
assert.equal(visibleTurnText(targetOnly).includes(JSON.stringify(options)), false,
  'current target arguments cannot establish input visibility');
