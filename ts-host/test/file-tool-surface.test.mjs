import assert from 'node:assert/strict';
import { test } from 'node:test';
import { NativeToolAgent } from '../dist/native/agent.js';
import { directoryReducerPrompt, DIRECTORY_REDUCER_PROMPT } from '../dist/native/prompt.js';

const names = surface => new NativeToolAgent(async () => ({ calls: [] }), surface ? { fileTools: surface } : {})
  .tools({ runtime: { frame: { adHocDepth: 0 } }, lam: { codebase: {}, projectTransaction: {}, type: { kind: 'prim', name: 'unknown' } } })
  .map(item => item.function.name);

test('a directory reducer offers the file tools of its surface, and its prompt names only those', () => {
  assert.deepEqual(names().filter(name => !['eval', 'read_page', 'read_code', 'compact_history', 'return_result'].includes(name)),
    ['list_files', 'search_files', 'read_file', 'write_file', 'edit_file', 'diff_files', 'bash', 'python', 'delegate', 'editor']);
  assert.deepEqual(names('editor').filter(name => ['editor', 'bash', 'read_file', 'list_files'].includes(name)), ['editor', 'bash']);
  assert.deepEqual(names('files').filter(name => ['editor', 'bash', 'read_file'].includes(name)), ['read_file']);
  assert.equal(DIRECTORY_REDUCER_PROMPT, directoryReducerPrompt('all'));
  assert.doesNotMatch(directoryReducerPrompt('editor'), /read_file|list_files/);
  assert.match(directoryReducerPrompt('editor'), /- editor\(command/);
  assert.doesNotMatch(directoryReducerPrompt('files'), /- bash|- editor/);
});
