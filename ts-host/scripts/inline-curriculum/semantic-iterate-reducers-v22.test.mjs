import test from 'node:test';
import assert from 'node:assert/strict';
import { escapeTaggedTemplateText } from './semantic-iterate-worlds-v15-soft-guided-builder.mjs';

test('guided source prose is safely quoted into a JavaScript tagged template', () => {
  const prose = ['Use ', '$', '{current}', ', a backtick ', '`here`', ', and a slash ', '\\', '.'].join('');
  const source = `return capture\`${escapeTaggedTemplateText(prose)}\`;`;
  const capture = strings => strings[0];
  const rendered = new Function('capture', source)(capture);
  assert.equal(rendered, prose);
});

