import assert from 'node:assert/strict';
import { test } from 'node:test';
import { Folder } from '../dist/index.js';
import { NativeToolAgent } from '../dist/native/agent.js';
import { directoryReducerPrompt, DIRECTORY_REDUCER_PROMPT, scopedFileToolNames, TOOLS_PROMPT } from '../dist/native/prompt.js';
import { BUILT_IN_DOCS } from '../dist/native/runtime.js';

const names = surface => new NativeToolAgent(async () => ({ calls: [] }), surface ? { fileTools: surface } : {})
  .tools({ runtime: { frame: { adHocDepth: 0 } }, rememberOfferedTools() {}, lam: { codebase: {}, projectTransaction: { folder: { access: 'write' } }, type: { kind: 'prim', name: 'unknown' } } })
  .map(item => item.function.name);

test('a directory reducer offers the file tools of its surface, and its prompt names only those', () => {
  const tools = new NativeToolAgent(async () => ({ calls: [] }))
    .tools({ runtime: { frame: { adHocDepth: 0 } }, rememberOfferedTools() {}, lam: { codebase: {}, projectTransaction: { folder: { access: 'write' } }, type: { kind: 'prim', name: 'unknown' } } });
  const delegate = tools.find(item => item.function.name === 'delegate').function;
  assert.match(delegate.parameters.properties.returns.description, /Natlang result type expression/);
  assert.match(delegate.parameters.properties.returns.description, /boolean, string, number/);
  assert.match(delegate.parameters.properties.returns.description, /not a prose description/);
  assert.deepEqual(names().filter(name => !['eval', 'read_page', 'read_code', 'compact_history', 'return_result'].includes(name)),
    ['list_files', 'search_files', 'read_file', 'write_file', 'edit_file', 'diff_files', 'bash', 'python', 'delegate', 'editor']);
  assert.deepEqual(names('editor').filter(name => ['editor', 'bash', 'read_file', 'list_files'].includes(name)), ['editor', 'bash']);
  assert.deepEqual(names('files').filter(name => ['editor', 'bash', 'read_file'].includes(name)), ['read_file']);
  assert.equal(DIRECTORY_REDUCER_PROMPT, directoryReducerPrompt('all'));
  assert.doesNotMatch(directoryReducerPrompt('editor'), /read_file|list_files|write_file/);
  assert.match(directoryReducerPrompt('editor'), /- editor\(command/);
  assert.doesNotMatch(directoryReducerPrompt('files'), /- bash|- editor/);
});

test('read-only scoped folders omit mutation tools while writable file and folder scopes retain them', async () => {
  const makeAgent = transaction => new NativeToolAgent(async () => ({ calls: [] })).tools({
    runtime: { frame: { adHocDepth: 0 } }, rememberOfferedTools() {},
    lam: { codebase: {}, projectTransaction: transaction, type: { kind: 'prim', name: 'unknown' } },
  }).map(item => item.function.name);
  const readBase = Folder.fromFiles({ 'input.txt': 'evidence' }, 'read');
  const readonlyTx = await readBase.beginFileTransaction('input.txt');
  const readonlyNames = makeAgent(readonlyTx);
  for (const name of ['write_file', 'edit_file', 'editor', 'bash', 'python']) assert.ok(!readonlyNames.includes(name), `${name} offered for read-only scope`);
  for (const name of ['list_files', 'search_files', 'read_file', 'diff_files', 'delegate']) assert.ok(readonlyNames.includes(name), `${name} missing from read-only scope`);

  const writableBase = Folder.fromFiles({ 'input.txt': 'evidence' }, 'write');
  const writableFileTx = await writableBase.beginFileTransaction('input.txt');
  const writableNames = makeAgent(writableFileTx);
  for (const name of ['write_file', 'edit_file', 'editor', 'bash', 'python']) assert.ok(writableNames.includes(name), `${name} missing from writable file scope`);
  const writableFolderTx = await Folder.fromFiles({ 'input.txt': 'evidence' }, 'write').beginTransaction();
  assert.ok(makeAgent(writableFolderTx).includes('write_file'), 'writable Folder retains file mutation tools');

  assert.deepEqual(scopedFileToolNames('all', false), ['list_files', 'search_files', 'read_file', 'diff_files', 'delegate']);
  const readonlyPrompt = directoryReducerPrompt('all', true, false);
  assert.match(readonlyPrompt, /- read_file\(/);
  assert.match(readonlyPrompt, /- delegate\(/);
  for (const name of ['write_file', 'edit_file', 'editor', 'bash(', 'python(']) assert.doesNotMatch(readonlyPrompt, new RegExp(name.replace(/[()]/g, '\\$&')));
  assert.match(directoryReducerPrompt('all', true, true), /- write_file\(/);
});

test('directory reducers promote semantic per-file lambdas only while ad hoc calls are available', () => {
  assert.match(directoryReducerPrompt('all', true), /Use inline nl lambdas for semantic per-file judgments/);
  assert.doesNotMatch(directoryReducerPrompt('all', true), /make the judgments here or delegate/);
  assert.doesNotMatch(directoryReducerPrompt('all', false), /inline nl lambdas|nl<|nl`/);
});

test('file guidance preserves each supplied handle view and explicit output authority', () => {
  for (const surface of ['all', 'files', 'editor']) {
    const prompt = directoryReducerPrompt(surface);
    assert.match(prompt, /Read a supplied FileHandle directly/);
    assert.match(prompt, /folder\.file\(sourceFile\.path\) may refer to a different view/);
    assert.match(prompt, /can access only that file through its folder/);
    assert.match(prompt, /Pass an output FileHandle explicitly/);
    assert.match(prompt, /structured object or array, verify saved JSON with readJson<T>\(\) and return that parsed value directly/);
    assert.match(prompt, /Use readText\(\) as the result only when the declared return type is string/);
    assert.match(prompt, /do not wrap JSON text in an object or array/);
  }
});

test('inline lambda guidance exposes explicit snapshots and latest state arguments', () => {
  assert.match(TOOLS_PROMPT, /nl\.with<boolean>\(\{ policy \}\)/);
  assert.match(TOOLS_PROMPT, /nl\.with<ResultType> declares the child result type; its object argument supplies fixed context/);
  assert.match(TOOLS_PROMPT, /Keep input FileHandles in the arguments/);
  assert.match(TOOLS_PROMPT, /snapshots taken when the function is created/);
  assert.match(TOOLS_PROMPT, /Pass changing iteration state as an argument/);
});


test('partial semantic checks retain explicit evidence scope without inheriting parent context', () => {
  assert.match(TOOLS_PROMPT, /A child does not inherit the parent conversation or sibling evidence/);
  assert.match(TOOLS_PROMPT, /as arguments or named captures/);
  assert.match(TOOLS_PROMPT, /Evaluate each assigned condition independently/);
  for (const surface of ['all','files','editor']) {
    assert.match(directoryReducerPrompt(surface), /Include the record's entity or group context/);
  }
});


test('ordered repairs keep the current pass and the declared carried-value shape', () => {
  assert.match(TOOLS_PROMPT, /apply only the current pass's correction/);
  assert.match(TOOLS_PROMPT, /evidence for later passes/);
  assert.match(TOOLS_PROMPT, /Return the declared draft value itself/);
});


test('nl help distinguishes result annotations, capture snapshots and changing inputs', () => {
  assert.match(BUILT_IN_DOCS.nl, /nl\.with<T>\(\{ policy \}\)/);
  assert.match(BUILT_IN_DOCS.nl, /nl<T>`instructions`\.with\(\{ policy \}\)\(item\)/);
  assert.match(BUILT_IN_DOCS.nl, /can apply directly to an inline\s+nl template/);
  assert.match(BUILT_IN_DOCS.nl, /T describes the result, not the capture object/);
  assert.match(BUILT_IN_DOCS.nl, /parameter shadows the capture inside the child/);
  assert.match(BUILT_IN_DOCS.nl, /Pass changing iteration state as an argument/);
});
