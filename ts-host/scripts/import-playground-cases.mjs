#!/usr/bin/env node
/** Convert reviewed, admitted playground cases into program IR: the project, its inputs, and the checked result. */
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { basename, dirname, extname, isAbsolute, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { PROGRAM_VERSION, programDefinition } from '../dist/teacher/program.js';

function checkSourcePath(value) {
  if (typeof value !== 'string' || !value || value.includes('\\') || value.includes('\0'))
    throw new Error(`unsafe or unsupported source path: ${value}`);
  const parts = value.split('/');
  if (isAbsolute(value) || parts.some(part => !part || part === '.' || part === '..') || !['.nl', '.ts'].includes(extname(value)))
    throw new Error(`unsafe or unsupported source path: ${value}`);
}

export async function convertCase(caseRecord) {
  if (caseRecord?.schema !== 'natlang.playground.case/1' || caseRecord.reviewStatus !== 'accepted' ||
      caseRecord.admission?.admitted !== true) throw new Error('case is not reviewed and exactly admitted');
  const source = caseRecord.source ?? {}, files = source.files ?? {};
  for (const [path, text] of Object.entries(files)) {
    checkSourcePath(path);
    if (typeof text !== 'string') throw new Error('source file must contain text');
  }
  if (extname(source.root ?? '') !== '.nl' || typeof files[source.root] !== 'string')
    throw new Error('only natural-language root functions can become program IR');
  const events = caseRecord.trace ?? [];
  if (!events.length || events[0].kind !== 'manifest' || events.some((event, index) =>
    event.seq !== index || event.version !== 'reduction-trace/1'))
    throw new Error('missing or discontinuous reduction trace');
  if (events.some(event => event.kind === 'effect')) throw new Error('host effects need an effect contract');
  const expected = caseRecord.expected ?? {};
  if (!['done', 'blocked'].includes(expected.kind)) throw new Error(`unsupported expected outcome: ${expected.kind}`);
  const id = String(caseRecord.id);
  const record = { version: PROGRAM_VERSION, id, kind: 'lambda_source', source: 'browser_playground',
    split: caseRecord.split ?? 'train', source_ids: [id],
    source_groups: [String(caseRecord.groupId ?? caseRecord.projectId ?? id)],
    source_revisions: [String(caseRecord.revision)], license: 'user-authored', gold_sources: ['reviewed-browser-case'],
    semantics: { root: source.root, files: structuredClone(files), inputs: structuredClone(caseRecord.inputs ?? {}),
      expected: expected.kind === 'done' ? expected.value : null, operation: expected.kind === 'blocked' ? 'blocked' : 'exact',
      contract: { required_actions: events.filter(event => event.kind === 'action').map(event => ({ tool: event.name, arguments: event.arguments })),
        instruction_obligations: caseRecord.requiredActions ?? [] } } };
  // The project must load: its root, callable folder, and declared inputs.
  const definition = programDefinition(record);
  for (const name of Object.keys(record.semantics.inputs))
    if (!definition.params.some(param => param.name === name)) throw new Error(`${name} is not a parameter of ${definition.name}`);
  return record;
}

export async function importCases(sourcePath, outputPath, rejectsPath = null) {
  const source = resolve(sourcePath), output = resolve(outputPath);
  if (source === output) throw new Error('input and output must differ');
  const rejectFile = resolve(rejectsPath ?? join(dirname(output), `${basename(output, extname(output))}-rejects.jsonl`));
  await mkdir(dirname(output), { recursive: true });
  const accepted = [], rejected = [];
  const lines = (await readFile(source, 'utf8')).split(/\r?\n/);
  for (let index = 0; index < lines.length; index++) {
    if (!lines[index].trim()) continue;
    let record = null;
    try {
      record = JSON.parse(lines[index]);
      accepted.push(await convertCase(record));
    } catch (error) {
      rejected.push({ line: index + 1, id: record && typeof record === 'object' ? record.id ?? null : null,
        reason: error instanceof Error ? error.message : String(error) });
    }
  }
  await writeFile(output, accepted.map(row => JSON.stringify(row)).join('\n') + (accepted.length ? '\n' : ''));
  await writeFile(rejectFile, rejected.map(row => JSON.stringify(row)).join('\n') + (rejected.length ? '\n' : ''));
  console.log(JSON.stringify({ accepted: accepted.length, rejected: rejected.length,
    program_ir: output, rejects: rejectFile }));
  return { accepted: accepted.length, rejected: rejected.length, output, rejects: rejectFile };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [, , source, output, ...rest] = process.argv;
  if (!source || !output || (rest.length && (rest.length !== 2 || rest[0] !== '--rejects'))) {
    console.error('usage: node scripts/import-playground-cases.mjs INPUT.jsonl OUTPUT.jsonl [--rejects FILE]');
    process.exitCode = 2;
  } else {
    const rejectAt = rest[0] === '--rejects' ? process.argv.indexOf('--rejects') + 1 : -1;
    const rejects = rejectAt > 0 ? process.argv[rejectAt] : null;
    if (rest.length && !rejects) { console.error('unknown option'); process.exitCode = 2; }
    else importCases(source, output, rejects).catch(error => {
      console.error(error instanceof Error ? error.stack : String(error)); process.exitCode = 1;
    });
  }
}
