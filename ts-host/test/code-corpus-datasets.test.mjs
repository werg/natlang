import assert from 'node:assert/strict';
import { mkdtemp, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { promisify } from 'node:util';
import { execFile } from 'node:child_process';
import { normalizeCase2Code, normalizeTinyCodes, normalizeXlam, normalizeCodeSearchNet, normalizeMagicoder, normalizeMcEvalInstruct, importDataset } from '../scripts/code-corpus/datasets.mjs';
const execFileAsync = promisify(execFile);

test('Case2Code retains Python source and repr values without interpreting them', () => {
  const row = { id: 'ex-1', prompt: 'Sort the numbers', code: 'def solve(x): return sorted(x)', input: "__import__('os').system('false')", output: '[1, 2]' };
  const result = normalizeCase2Code(row, { revision: 'abc', license: 'MIT' });
  assert.equal(result.language, 'python');
  assert.equal(result.raw.python_repr_input, row.input);
  assert.equal(result.raw.expected_output_repr, row.output);
  assert.equal(result.function.body, row.code);
  assert.equal(result.verification.status, 'unverified');
  assert.equal(result.cases.length, 0);
  assert.deepEqual(result.source.license, 'MIT');
});

test('Case2Code example fields remain raw io synthesis data with upstream status and index', () => {
  const row = { id: 7, func_name: 'choose', instruction: 'Choose a value', code: 'def choose(x): return x', example_inputs: "{'x': 1}", example_outputs: '1', parsed_inputs: "{'x': 1}", exec_status: 'passed', choosed_example_idx: 2 };
  const result = normalizeCase2Code(row);
  assert.equal(result.kind, 'io_synthesis');
  assert.equal(result.function.name, 'choose');
  assert.equal(result.raw.example_inputs, row.example_inputs);
  assert.equal(result.raw.parsed_inputs, row.parsed_inputs);
  assert.equal(result.raw.choosed_example_idx, 2);
  assert.ok(result.verification.reasons.includes('upstream_exec_status:passed'));
  assert.equal(result.cases.length, 0);
});

test('xLAM parses serialized fields and follows schema signature order', () => {
  const row = { id: 2, query: 'Find a place', tools: JSON.stringify([{ name: 'lookup-place', parameters: { type: 'object', properties: { city: { type: 'string' }, limit: { type: 'integer' } }, required: ['city'] } }]), answers: JSON.stringify([{ name: 'lookup-place', arguments: '{"limit":3,"city":"Oslo"}' }]) };
  const result = normalizeXlam(row);
  assert.equal(result.kind, 'tool_calls');
  assert.equal(result.raw.typescript_call_view, 'await tools["lookup-place"]("Oslo", 3);');
  assert.match(result.verification.reasons.join(' '), /not_executed/);
  assert.equal(result.cases.length, 0);
});

test('xLAM accepts Salesforce direct parameter descriptors and preserves omitted optional positions', () => {
  const row = { query: 'Call it', tools: [{ name: 'lookup', parameters: { first: { type: 'string', required: true }, optional: { type: 'string', required: false }, last: { type: 'integer', required: true } } }], answers: [{ name: 'lookup', arguments: { first: 'a', last: 4 } }] };
  const result = normalizeXlam(row);
  assert.equal(result.raw.typescript_call_view, 'await tools["lookup"]("a", undefined, 4);');
  assert.deepEqual(result.raw.signatures[0].parameters.map(p => [p.name, p.required]), [['first', true], ['optional', false], ['last', true]]);
  const odd = normalizeXlam({ query: 'x', tools: [{ name: 'bad-tool', parameters: { 'not-valid': { type: 'string' } } }], answers: [{ name: 'bad-tool', arguments: { 'not-valid': 'x' } }] });
  assert.ok(odd.verification.reasons.includes('invalid_signature_identifier:bad-tool'));
});

test('xLAM flags malformed, unknown, and schema-mismatched calls', () => {
  const bad = normalizeXlam({ query: 'Do it', tools: '[{"name":"ok","parameters":{"properties":{"x":{}},"required":["x"]}}]', answers: '[{"name":"missing","arguments":{}},{"name":"ok","arguments":"not json"},{"name":"ok","arguments":{"extra":1}}]' });
  assert.equal(bad.verification.status, 'candidate');
  assert.ok(bad.verification.reasons.some(x => x.startsWith('unknown_tool:')));
  assert.ok(bad.verification.reasons.some(x => x.startsWith('malformed_arguments:')));
  assert.ok(bad.verification.reasons.some(x => x.startsWith('unknown_arguments:')));
  assert.equal(bad.raw.typescript_call_view, undefined);
});

test('Tiny-Codes requires explicit JS or TS metadata and accepts common aliases', () => {
  const js = normalizeTinyCodes({ prompt: 'Return the input', completion: 'export const f = x => x;', language: 'typescript' });
  assert.equal(js.language, 'typescript');
  assert.equal(js.function.body, 'export const f = x => x;');
  assert.equal(normalizeTinyCodes({ prompt: 'x', completion: 'int main(){}', language: 'Java' }).verification.status, 'rejected');
  assert.equal(normalizeTinyCodes({ prompt: 'x', completion: 'const x = 1;' }).verification.reasons[0], 'language_not_explicitly_javascript_or_typescript');
});

test('CodeSearchNet maps function documentation and code with stable repository provenance', () => {
  const row = { func_documentation_string: 'Return the input', func_code_string: 'function echo(x) { return x; }', repo: 'acme/lib', path: 'src/echo.js', url: 'https://example.test/echo', split: 'validation' };
  const a = normalizeCodeSearchNet(row, { input_sha256: 'shard-a' });
  const b = normalizeCodeSearchNet(row, { input_sha256: 'shard-b' });
  assert.equal(a.instruction, 'Return the input');
  assert.equal(a.function.source, row.func_code_string);
  assert.equal(a.source.repo, row.repo);
  assert.equal(a.source.path, row.path);
  assert.equal(a.source.url, row.url);
  assert.equal(a.source.upstream_split, 'validation');
  assert.equal(normalizeCodeSearchNet({ ...row, split: undefined }, { split: 'train' }).source.split, 'train');
  assert.equal(a.group_id, b.group_id);
  assert.equal(a.id, b.id);
  assert.equal(a.cases.length, 0);
});

test('Tiny-Codes extracts one code fence while preserving raw prose and ambiguous responses', () => {
  const row = {language:'typescript',prompt:'Double a number',response:'Here is code:\n```typescript \nfunction double(x: number) { return x * 2; }\n```\nExplanation.'};
  const result = normalizeTinyCodes(row);
  assert.equal(result.function.source, 'function double(x: number) { return x * 2; }');
  assert.equal(result.raw.source_row.response, row.response);
  const ambiguous = normalizeTinyCodes({...row,response:row.response+'\n```ts\nconst y = 2;\n```'});
  assert.ok(ambiguous.verification.reasons.includes('ambiguous_solution_fences:2'));
});

test('Magicoder preserves source fields and rejects non JavaScript languages', () => {
  const row = { lang: 'TypeScript', problem: 'Add two numbers', solution: 'Here is one solution:\n```typescript\nfunction add(a: number, b: number) { return a + b; }\n```', split: 'train' };
  const result = normalizeMagicoder(row);
  assert.equal(result.language, 'typescript');
  assert.equal(result.function.source, 'function add(a: number, b: number) { return a + b; }');
  assert.equal(result.source.upstream_language, 'TypeScript');
  assert.equal(result.source.transformation_label, 'single_javascript_typescript_fence');
  assert.equal(result.source.upstream_split, 'train');
  assert.equal(normalizeMagicoder({ lang: 'Python', problem: 'x', solution: 'print(1)' }).verification.status, 'rejected');
});

test('McEval-Instruct strips one outer solution fence while retaining raw row', () => {
  const row = { language: 'js', instruction: 'Make a greeting', output: 'Solution:\n```javascript\nfunction greet() { return "hi"; }\n```', split: 'test' };
  const result = normalizeMcEvalInstruct(row);
  assert.equal(result.function.source, 'function greet() { return "hi"; }');
  assert.equal(result.raw.source_row.output, row.output);
  assert.equal(result.source.upstream_split, 'test');
  const ambiguous = normalizeMcEvalInstruct({ language: 'js', instruction: 'x', output: '```js\nconst a = 1;\n```\n```js\nconst b = 2;\n```' });
  assert.equal(ambiguous.function.source, ambiguous.raw.source_row.output);
  assert.ok(ambiguous.verification.reasons.includes('ambiguous_solution_fences:2'));
  assert.equal(ambiguous.verification.status, 'unverified');
  assert.equal(normalizeMcEvalInstruct({ language: 'java', instruction: 'x', output: 'y' }).verification.status, 'rejected');
});

test('bounded importer reads JSON arrays, hashes the input, and refuses existing outputs', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'code-corpus-'));
  const input = join(dir, 'rows.json'), output = join(dir, 'out.jsonl');
  await writeFile(input, JSON.stringify([{ prompt: 'one', completion: 'const x=1', language: 'js' }, { prompt: 'two', completion: 'const y=2', language: 'js' }]));
  const result = await importDataset({ input, output, adapter: 'tiny-codes', limit: 1, source: { revision: 'r1', license: 'MIT' } });
  assert.equal(result.rows, 1);
  assert.match(result.input_sha256, /^[0-9a-f]{64}$/);
  assert.equal((await readFile(output, 'utf8')).trim().split('\n').length, 1);
  await assert.rejects(() => importDataset({ input, output, adapter: 'tiny-codes', limit: 1, source: {} }), /exist|overwrite/i);
});

test('importer unwraps Hugging Face rows and rejects truncated cells', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'code-corpus-hf-'));
  const input = join(dir, 'rows.jsonl'), output = join(dir, 'new', 'out.jsonl');
  await writeFile(input, [
    { row_idx: 11, row: { prompt: 'one', completion: 'const x=1', language: 'js' }, truncated_cells: [] },
    { row_idx: 12, row: { prompt: 'two', completion: 'const y=2', language: 'js' }, truncated_cells: ['completion'] },
  ].map(JSON.stringify).join('\n') + '\n');
  const result = await importDataset({ input, output, adapter: 'tiny-codes', source: {} });
  assert.equal(result.rows, 2);
  const saved = (await readFile(output, 'utf8')).trim().split('\n').map(JSON.parse);
  assert.equal(saved[0].source.upstream_id, 11);
  assert.equal(saved[0].raw.hf_wrapper.row_idx, 11);
  assert.equal(saved[1].verification.status, 'rejected');
  assert.ok(saved[1].verification.reasons.some(r => r.startsWith('truncated_cells:')));
  assert.equal(JSON.parse(await readFile(`${output}.manifest.json`, 'utf8')).limit, 100);
});

test('CLI --replace is a boolean flag and does not consume the following argument', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'code-corpus-cli-'));
  const input = join(dir, 'rows.json'), output = join(dir, 'nested', 'out.jsonl');
  await writeFile(input, JSON.stringify([{ prompt: 'one', completion: 'const x=1', language: 'js' }]));
  const script = new URL('../scripts/code-corpus/datasets.mjs', import.meta.url).pathname;
  const { stdout } = await execFileAsync(process.execPath, [script, '--adapter', 'tiny-codes', '--input', input, '--output', output, '--replace', '--revision', 'r1']);
  assert.equal(JSON.parse(stdout).limit, 100);
  assert.equal(JSON.parse((await readFile(output, 'utf8')).trim()).source.revision, 'r1');
});
