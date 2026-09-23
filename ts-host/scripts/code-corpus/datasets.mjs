#!/usr/bin/env node
import { readFile, open, link, rename, unlink, mkdir } from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import { createHash } from 'node:crypto';
import { basename, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { digest, readJsonl, writeJsonl } from './common.mjs';

const own = (o, k) => Object.hasOwn(o ?? {}, k);
const first = (o, keys) => keys.find(k => own(o, k));
const text = v => typeof v === 'string' ? v : null;
const metadata = (name, row, src) => ({ name, revision: src.revision ?? 'unknown', path: src.path ?? '', license: src.license ?? null, upstream_id: row?.id ?? row?.uid ?? src.upstream_id ?? null });
const identity = (name, row, src) => `${name}:${src.input_sha256 ?? digest(JSON.stringify(row))}:${row?.id ?? row?.uid ?? digest(JSON.stringify(row)).slice(0, 16)}`;
const rejected = (name, row, source, reasons, instruction = '') => ({ version: 'natlang.code_task/1', id: identity(name, row, source), group_id: identity(name, row, source), kind: 'instruction', language: 'unknown', instruction, source: metadata(name, row, source), function: null, cases: [], verification: { status: 'rejected', reasons }, raw: { source_row: row } });

function jsonField(value) {
  if (typeof value !== 'string') return { value, error: null };
  try { return { value: JSON.parse(value), error: null }; }
  catch (error) { return { value: null, error: `invalid JSON: ${error.message}` }; }
}

async function readJsonArray(path, limit) {
  const rows = [];
  if (limit <= 0) return rows;
  let started = false, inString = false, escaped = false, depth = 0, token = '', collecting = false;
  for await (const chunk of createReadStream(path, { encoding: 'utf8' })) {
    for (const ch of chunk) {
      if (!started) { if (ch === '[') started = true; else if (!/\s/.test(ch)) throw new Error('JSON input must be an array'); continue; }
      if (!collecting && !inString && ch === ']') return rows;
      if (!collecting && !inString && (ch === ',' || /\s/.test(ch))) continue;
      if (collecting && !inString && depth === 0 && ch === ']') {
        const raw = token.trim(); if (raw) rows.push(JSON.parse(raw)); return rows.slice(0, limit);
      }
      if (!collecting) { collecting = true; token = ''; depth = 0; inString = false; escaped = false; }
      token += ch;
      if (inString) {
        if (escaped) escaped = false;
        else if (ch === '\\') escaped = true;
        else if (ch === '"') inString = false;
      } else if (ch === '"') inString = true;
      else if (ch === '{' || ch === '[') depth++;
      else if (ch === '}' || ch === ']') {
        if (depth > 0) depth--;
      }
      if (!inString && depth === 0 && (ch === ',' || ch === ']')) {
        const raw = token.slice(0, -1).trim();
        if (raw) rows.push(JSON.parse(raw));
        collecting = false; token = '';
        if (rows.length >= limit) return rows;
        if (ch === ']') return rows;
      }
    }
  }
  if (collecting && token.trim()) rows.push(JSON.parse(token));
  if (!started) throw new Error('JSON input must be an array');
  return rows.slice(0, limit);
}

export function normalizeCase2Code(row, source = {}) {
  const p = first(row, ['prompt', 'instruction', 'question', 'description']);
  const c = first(row, ['code', 'solution', 'answer', 'completion', 'function_code', 'func_code']);
  const prompt = p ? text(row[p]) : null, code = c ? text(row[c]) : null;
  const pyreprKey = first(row, ['example_inputs', 'parsed_inputs', 'input', 'inputs', 'test_input', 'args']);
  const outputKey = first(row, ['example_outputs', 'parsed_outputs', 'output', 'outputs', 'expected', 'test_output']);
  const hasExamples = own(row, 'example_inputs') || own(row, 'example_outputs') || own(row, 'parsed_inputs') || own(row, 'parsed_outputs');
  if (!prompt || !code) return rejected('case2code', row, source, [!prompt ? 'missing_instruction' : null, !code ? 'missing_code' : null].filter(Boolean), prompt ?? '');
  const id = identity('case2code', row, source);
  return { version: 'natlang.code_task/1', id, group_id: id, kind: hasExamples ? 'io_synthesis' : 'instruction', language: 'python', instruction: prompt,
    source: metadata('case2code', row, { ...source, upstream_id: row?.id ?? row?.func_name ?? row?.uid }), function: { name: row.func_name ?? null, parameters: [], body: code, source: code }, cases: [],
    verification: { status: 'unverified', reasons: ['python_translation_request', 'dataset_values_not_executed', ...(row.exec_status ? [`upstream_exec_status:${row.exec_status}`] : [])] },
    raw: { source_row: row, python_code: code, func_name: row.func_name, example_inputs: row.example_inputs,
      example_outputs: row.example_outputs, parsed_inputs: row.parsed_inputs, parsed_outputs: row.parsed_outputs,
      python_repr_input: pyreprKey ? row[pyreprKey] : undefined,
      expected_output_repr: outputKey ? row[outputKey] : undefined, exec_status: row.exec_status,
      choosed_example_idx: row.choosed_example_idx } };
}

export function normalizeXlam(row, source = {}) {
  const qKey = first(row, ['query', 'instruction', 'prompt']);
  const toolsKey = first(row, ['tools', 'functions']);
  const callsKey = first(row, ['answers', 'calls', 'tool_calls']);
  const query = qKey ? text(row[qKey]) : null;
  const parsedTools = toolsKey ? jsonField(row[toolsKey]) : { value: null, error: 'missing tools' };
  const parsedCalls = callsKey ? jsonField(row[callsKey]) : { value: null, error: 'missing calls' };
  const reasons = [];
  if (!query) reasons.push('missing_instruction');
  if (parsedTools.error) reasons.push(`tools_${parsedTools.error}`);
  if (parsedCalls.error) reasons.push(`calls_${parsedCalls.error}`);
  if (!Array.isArray(parsedTools.value)) reasons.push('tools_not_array');
  if (!Array.isArray(parsedCalls.value)) reasons.push('calls_not_array');
  const tools = Array.isArray(parsedTools.value) ? parsedTools.value : [];
  const calls = Array.isArray(parsedCalls.value) ? parsedCalls.value : [];
  const byName = new Map();
  for (const tool of tools) {
    if (!tool || typeof tool.name !== 'string' || !tool.name) reasons.push('unknown_tool_schema');
    else if (byName.has(tool.name)) reasons.push(`duplicate_tool:${tool.name}`);
    else byName.set(tool.name, tool);
  }
  const views = [], signatures = [];
  const signatureByTool = new Map();
  for (const tool of tools) {
    if (!tool || typeof tool.name !== 'string' || !tool.name) continue;
    const parameters = tool.parameters ?? {};
    const props = parameters.properties && typeof parameters.properties === 'object' && !Array.isArray(parameters.properties)
      ? parameters.properties : parameters;
    const reserved = new Set(['type', 'required', 'description', 'additionalProperties', '$schema']);
    const names = Object.keys(props).filter(k => !reserved.has(k) && props[k] && typeof props[k] === 'object');
    const requiredList = Array.isArray(parameters.required) ? parameters.required : [];
    const params = names.map(name => ({ name, type: props[name].type ?? 'unknown', required: typeof props[name].required === 'boolean' ? props[name].required : requiredList.includes(name) }));
    const valid = /^[$A-Z_a-z][$\w]*$/.test(tool.name) && params.every(p => /^[$A-Z_a-z][$\w]*$/.test(p.name));
    signatureByTool.set(tool.name, { parameters: params, valid });
    signatures.push({ name: tool.name, parameters: params, identifiers_valid: valid });
    if (!valid) reasons.push(`invalid_signature_identifier:${tool.name}`);
  }
  for (const call of calls) {
    const tool = call && byName.get(call.name);
    if (!tool) { reasons.push(`unknown_tool:${call?.name ?? 'missing_name'}`); continue; }
    const argsParsed = jsonField(call.arguments);
    if (argsParsed.error || !argsParsed.value || Array.isArray(argsParsed.value) || typeof argsParsed.value !== 'object') {
      reasons.push(`malformed_arguments:${call.name}`); continue;
    }
    const args = argsParsed.value;
    const signature = signatureByTool.get(call.name);
    if (!signature || !signature.parameters.length) { reasons.push(`missing_signature:${call.name}`); continue; }
    const ordered = signature.parameters.map(p => p.name);
    const required = new Set(signature.parameters.filter(p => p.required).map(p => p.name));
    const extra = Object.keys(args).filter(k => !ordered.includes(k));
    const missing = ordered.filter(k => !own(args, k) && required.has(k));
    if (extra.length) reasons.push(`unknown_arguments:${call.name}:${extra.join(',')}`);
    if (missing.length) reasons.push(`missing_arguments:${call.name}:${missing.join(',')}`);
    if (!extra.length && !missing.length) views.push(`await tools[${JSON.stringify(call.name)}](${ordered.map(k => own(args, k) ? JSON.stringify(args[k]) : 'undefined').join(', ')});`);
  }
  if (!calls.length && Array.isArray(parsedCalls.value)) reasons.push('empty_calls');
  const id = identity('xlam', row, source);
  const record = { version: 'natlang.code_task/1', id, group_id: id, kind: 'tool_calls', language: 'typescript', instruction: query ?? '', source: metadata('xlam-function-calling-60k', row, source), function: null, cases: [], verification: { status: reasons.length ? 'candidate' : 'unverified', reasons: [...reasons, 'call_view_is_schema_ordered_not_executed', 'no_runtime_outputs_claimed'] }, raw: { source_row: row, tools, calls, signatures, explicit_signatures: signatures } };
  if (views.length === calls.length && views.length) record.raw.typescript_call_view = views.join('\n');
  return record;
}

export function normalizeTinyCodes(row, source = {}) {
  const instructionKey = first(row, ['instruction', 'prompt', 'question', 'description', 'text']);
  const codeKey = first(row, ['code', 'completion', 'solution', 'response', 'output']);
  const langKey = first(row, ['language', 'lang', 'programming_language', 'language_name']);
  const instruction = instructionKey ? text(row[instructionKey]) : null;
  const code = codeKey ? text(row[codeKey]) : null;
  const lang = langKey ? text(row[langKey]) : null;
  const reasons = [];
  if (!instruction) reasons.push('missing_instruction');
  if (!code) reasons.push('missing_code');
  if (!lang || !/^(?:javascript|js|typescript|ts)(?:\b|[-_])/i.test(lang.trim())) reasons.push('language_not_explicitly_javascript_or_typescript');
  if (reasons.length) return rejected('tiny-codes', row, source, reasons, instruction ?? '');
  const normalizedLanguage = /^(?:typescript|ts)/i.test(lang.trim()) ? 'typescript' : 'javascript';
  const extracted = extractSolution(code);
  const id = identity('tiny-codes', row, source);
  return { version: 'natlang.code_task/1', id, group_id: id, kind: 'instruction', language: normalizedLanguage, instruction,
    source: { ...metadata('tiny-codes', row, source), transformation_label: extracted.label },
    function: { name: null, parameters: [], body: extracted.code, source: extracted.code }, cases: [],
    verification: { status: 'unverified', reasons: ['dataset_code_candidate', 'not_executed', ...(extracted.diagnostic ? [extracted.diagnostic] : [])] }, raw: { source_row: row } };
}

const stableHash = value => createHash('sha256').update(String(value)).digest('hex');
function jsTsLanguage(value) {
  if (typeof value !== 'string') return null;
  const lang = value.trim().toLowerCase();
  if (/^(?:javascript|js)(?:\b|[-_])/.test(lang)) return 'javascript';
  if (/^(?:typescript|ts)(?:\b|[-_])/.test(lang)) return 'typescript';
  return null;
}
function extractSolution(code) {
  const blocks = [...code.matchAll(/```(javascript|js|typescript|ts)\s*\r?\n([\s\S]*?)\r?\n```/gi)];
  if (blocks.length === 1) return { code: blocks[0][2], label: 'single_javascript_typescript_fence', diagnostic: null };
  if (blocks.length > 1) return { code, label: 'ambiguous_multiple_fenced_blocks', diagnostic: `ambiguous_solution_fences:${blocks.length}` };
  return { code, label: 'raw_solution', diagnostic: null };
}
function directRecord(adapter, row, source, { instruction, code, language = 'javascript', name = null, group, provenance = {}, split, transform = null }) {
  const validationReasons = [];
  if (!instruction) validationReasons.push('missing_instruction');
  if (!code) validationReasons.push('missing_code');
  if (!language) validationReasons.push('language_not_explicitly_javascript_or_typescript');
  if (validationReasons.length) return rejected(adapter, row, source, validationReasons, instruction ?? '');
  // Dataset identities are derived from row content and stable upstream location, never the shard hash.
  const location = group || row.id || row.uid || stableHash(JSON.stringify(row));
  const id = `${adapter}:${stableHash(`${location}\n${instruction}\n${code}`).slice(0, 32)}`;
  const src = { ...metadata(adapter, row, { ...source, upstream_id: row.id ?? row.uid ?? location }), ...provenance };
  if (split !== undefined) { src.split = split; src.upstream_split = split; }
  if (transform) src.transformation_label = transform.label;
  const reasons = ['dataset_code_candidate', 'not_executed'];
  if (transform?.diagnostic) reasons.push(transform.diagnostic);
  return { version: 'natlang.code_task/1', id, group_id: `${adapter}:${stableHash(location).slice(0, 32)}`, kind: 'instruction', language, instruction,
    source: src, function: { name, parameters: [], body: code, source: code }, cases: [],
    verification: { status: 'unverified', reasons }, raw: { source_row: row } };
}

export function normalizeCodeSearchNet(row, source = {}) {
  const instruction = text(row.func_documentation_string ?? row.docstring);
  const code = text(row.func_code_string ?? row.code);
  const language = row.language == null && row.lang == null ? 'javascript' : jsTsLanguage(row.language ?? row.lang);
  const repo = text(row.repo), path = text(row.path), url = text(row.url);
  const split = row.split ?? source.split ?? source.upstream_split;
  const group = repo || row.repo_name ? `${repo ?? row.repo_name}:${path ?? row.func_name ?? ''}` : undefined;
  return directRecord('codesearchnet', row, source, { instruction, code, language, name: row.func_name ?? null, group, split,
    provenance: { repo, path, url, upstream_language: row.language ?? row.lang ?? 'javascript' } });
}

export function normalizeMagicoder(row, source = {}) {
  const language = jsTsLanguage(row.lang ?? row.language);
  const original = text(row.solution);
  const extracted = original ? extractSolution(original) : null;
  return directRecord('magicoder', row, source, { instruction: text(row.problem), code: extracted?.code ?? null, language, transform: extracted,
    group: row.id ?? row.uid ?? stableHash(`${row.problem ?? ''}\n${row.solution ?? ''}`), split: row.split ?? source.split ?? source.upstream_split,
    provenance: { upstream_language: row.lang ?? row.language ?? null } });
}

export function normalizeMcEvalInstruct(row, source = {}) {
  const language = jsTsLanguage(row.language ?? row.lang);
  const original = text(row.output);
  const extracted = original ? extractSolution(original) : null;
  return directRecord('mceval', row, source, { instruction: text(row.instruction), code: extracted?.code ?? null, language, transform: extracted,
    group: row.id ?? row.uid ?? stableHash(`${row.instruction ?? ''}\n${row.output ?? ''}`), split: row.split ?? source.split ?? source.upstream_split,
    provenance: { upstream_language: row.language ?? row.lang ?? null } });
}

async function atomicManifest(path, value, replace) {
  const target = resolve(path), temp = `${target}.${process.pid}.tmp`;
  const handle = await open(temp, 'wx');
  try { await handle.writeFile(`${JSON.stringify(value)}\n`); await handle.sync(); }
  finally { await handle.close(); }
  try { if (replace) await rename(temp, target); else { await link(temp, target); await unlink(temp); } }
  catch (error) { await unlink(temp).catch(() => {}); throw error; }
}

export async function importDataset({ input, output, source, adapter, limit = 100, replace = false }) {
  const inputPath = resolve(input), outputPath = resolve(output);
  if (!Number.isSafeInteger(limit) || limit < 1) throw new Error('limit must be a positive integer');
  const hash = createHash('sha256');
  let prefix = '';
  for await (const chunk of createReadStream(inputPath)) { hash.update(chunk); if (prefix.length < 32) prefix += chunk.toString('utf8', 0, Math.min(chunk.length, 32 - prefix.length)); }
  const inputSha256 = hash.digest('hex');
  const isArray = prefix.trimStart().startsWith('[');
  let rows;
  if (isArray) rows = await readJsonArray(inputPath, limit);
  else rows = await readJsonl(inputPath, { limit });
  const fn = ({ case2code: normalizeCase2Code, xlam: normalizeXlam, 'tiny-codes': normalizeTinyCodes, codesearchnet: normalizeCodeSearchNet,
    magicoder: normalizeMagicoder, mceval: normalizeMcEvalInstruct })[adapter];
  if (!fn) throw new Error(`unknown adapter: ${adapter}`);
  const records = rows.map(item => {
    const wrapped = item && typeof item === 'object' && !Array.isArray(item) && item.row && typeof item.row === 'object' && !Array.isArray(item.row) && Number.isInteger(item.row_idx);
    const row = wrapped ? { ...item.row, id: item.row.id ?? item.row_idx } : item;
    const record = fn(row, { ...source, input_sha256: inputSha256, path: source.path ?? basename(inputPath) });
    if (wrapped) {
      record.raw.hf_wrapper = item;
      if (Array.isArray(item.truncated_cells) && item.truncated_cells.length) {
        record.verification = { status: 'rejected', reasons: [...record.verification.reasons, `truncated_cells:${item.truncated_cells.join(',')}`] };
      }
    }
    return record;
  });
  const statuses = Object.groupBy ? Object.groupBy(records, r => r.verification.status) : records.reduce((o, r) => ((o[r.verification.status] ??= []).push(r), o), {});
  const manifest = { version: 'natlang.code_corpus_import/1', adapter, input: inputPath, input_sha256: inputSha256, limit: Number.isFinite(limit) ? limit : null, rows: rows.length, emitted: records.length, verification: Object.fromEntries(Object.entries(statuses).map(([k, v]) => [k, v.length])), output: outputPath };
  await mkdir(resolve(outputPath, '..'), { recursive: true });
  const manifestPath = `${outputPath}.manifest.json`;
  await atomicManifest(manifestPath, manifest, replace);
  try { await writeJsonl(outputPath, records, { replace }); }
  catch (error) { if (!replace) await unlink(manifestPath).catch(() => {}); throw error; }
  return manifest;
}

function argsOf(argv) {
  const o = {};
  const flags = new Set(['replace']);
  for (let i = 0; i < argv.length; i++) {
    const m = argv[i].match(/^--([^=]+)(?:=(.*))?$/);
    if (!m) throw new Error(`unexpected argument ${argv[i]}`);
    if (flags.has(m[1])) { if (m[2] !== undefined) throw new Error(`--${m[1]} does not take a value`); o[m[1]] = true; }
    else { const value = m[2] ?? argv[++i]; if (value === undefined || value.startsWith('--')) throw new Error(`--${m[1]} requires a value`); o[m[1]] = value; }
  }
  return o;
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  try {
    const a = argsOf(process.argv.slice(2));
    if (!a.input || !a.output || !a.adapter) throw new Error('usage: datasets.mjs --adapter case2code|xlam|tiny-codes|codesearchnet|magicoder|mceval --input rows.jsonl --output tasks.jsonl [--limit N] [--replace] [--license LICENSE] [--revision REV]');
    const result = await importDataset({ input: a.input, output: a.output, adapter: a.adapter, limit: a.limit === undefined ? 100 : Number(a.limit), replace: Boolean(a.replace), source: { revision: a.revision, license: a.license } });
    process.stdout.write(`${JSON.stringify(result)}\n`);
  } catch (error) { process.stderr.write(`${error.message}\n`); process.exitCode = 1; }
}
