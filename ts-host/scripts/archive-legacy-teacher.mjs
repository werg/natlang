#!/usr/bin/env node
/** Collapse superseded teacher shards into one template-neutral decision archive. */
import { createHash, randomUUID } from 'node:crypto';
import { mkdir, readFile, readdir, rename, stat, writeFile } from 'node:fs/promises';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gzipSync } from 'node:zlib';

const LEGACY = 'natlang.teacher_trajectory/1';
const ARCHIVE = 'natlang.teacher_decision_archive/1';
const digest = value => createHash('sha256').update(typeof value === 'string' || Buffer.isBuffer(value) ?
  value : JSON.stringify(value)).digest('hex');

async function filesBelow(root) {
  const out = [];
  for (const entry of await readdir(root, { withFileTypes: true })) {
    const path = resolve(root, entry.name);
    if (entry.isDirectory()) out.push(...await filesBelow(path));
    else if (entry.isFile() && entry.name.endsWith('.jsonl')) out.push(path);
  }
  return out.sort();
}

function publicProvenance(value = {}) {
  const out = structuredClone(value);
  for (const key of ['teacher_system_prompt', 'system_prompt']) if (typeof out[key] === 'string') {
    out[`${key}_sha256`] = digest(out[key]); delete out[key];
  }
  return out;
}

export function archiveLegacyRow(row, source) {
  if (row.version !== LEGACY || !Array.isArray(row.trajectory)) return null;
  const task = structuredClone(row.task ?? {});
  if (task.program_ir) {
    task.program_ir_id = task.program_ir.id ?? null;
    task.program_ir_sha256 = digest(JSON.stringify(task.program_ir));
    delete task.program_ir;
  }
  const decisions = row.trajectory.map((turn, index) => ({ index,
    function: turn.function ?? null, call_id: turn.call_id ?? null,
    phase: turn.phase ?? (turn.assistant?.calls?.length ? 'action' : 'reply'),
    segment_turns: turn.segment_turns ?? null, segment_messages: turn.segment_messages ?? null,
    note: turn.note ?? null, assistant: structuredClone(turn.assistant ?? { content: '', reasoning: null, calls: [] }),
    reviews: structuredClone(turn.reviews ?? []), executions: structuredClone(turn.executions ?? []),
    offered_tool_names: (turn.tools_offered ?? []).map(tool => tool.function?.name ?? tool.name).filter(Boolean),
    request_context_sha256: digest(turn.context ?? []), raw_response_sha256: turn.raw_response_sha256 ?? null }));
  const sourceProgramIds = [...new Set([...(row.task?.source_program_ids ?? []),
    ...(row.task?.program_ir?.id ? [row.task.program_ir.id] : [])])];
  return { version: ARCHIVE, id: `${row.id}:${source.row_sha256.slice(0, 12)}`,
    teacher_trajectory_id: row.id, source, task,
    provenance: publicProvenance(row.provenance), outcome: structuredClone(row.outcome ?? {}),
    decisions, regeneration: { policy: row.task?.program_ir ? 'native-program-ir' :
      row.task?.reference_key ? 'native-reference-case' : sourceProgramIds.length ? 'native-linked-programs' : 'unlinked-review-only',
      source_program_ids: sourceProgramIds, reference_key: row.task?.reference_key ?? null },
    training_admission: { approved: false, reason: 'obsolete model-facing tool surface; regenerate on scope-eval-v1' },
    capture_limits: structuredClone(row.capture_limits ?? []) };
}

async function main() {
  const args = process.argv.slice(2), rootAt = args.indexOf('--root'), outAt = args.indexOf('--out');
  if (outAt < 0) throw new Error('usage: archive-legacy-teacher.mjs --out ARCHIVE.jsonl [--root runs]');
  const root = resolve(rootAt < 0 ? 'runs' : args[rootAt + 1]), output = resolve(args[outAt + 1]);
  const rows = [], exact = new Set(), programs = new Map(), sourceFiles = [], counts = { source_rows: 0, duplicate_rows: 0,
    archived_rows: 0, decisions: 0 };
  for (const path of await filesBelow(root)) {
    const info = await stat(path);
    if (!info.size) continue;
    const text = await readFile(path, 'utf8'), sourceSha256 = digest(text); let lineNumber = 0, used = 0;
    for (const line of text.split(/\r?\n/)) {
      lineNumber++; if (!line.trim()) continue;
      let row; try { row = JSON.parse(line); } catch { continue; }
      if (row.version !== LEGACY || !Array.isArray(row.trajectory)) continue;
      counts.source_rows++; const rowSha256 = digest(JSON.stringify(row));
      if (exact.has(rowSha256)) { counts.duplicate_rows++; continue; }
      exact.add(rowSha256);
      if (row.task?.program_ir?.id) {
        const encoded = JSON.stringify(row.task.program_ir);
        programs.set(digest(encoded), encoded);
      }
      const archived = archiveLegacyRow(row, { file: relative(process.cwd(), path), line: lineNumber,
        file_sha256: sourceSha256, row_sha256: rowSha256 });
      rows.push(archived); used++; counts.archived_rows++; counts.decisions += archived.decisions.length;
    }
    if (used) sourceFiles.push({ file: relative(process.cwd(), path), sha256: sourceSha256, unique_rows: used });
  }
  await mkdir(dirname(output), { recursive: true });
  const staged = `${output}.building-${process.pid}-${randomUUID()}`;
  const archiveText = rows.map(row => JSON.stringify(row)).join('\n') + (rows.length ? '\n' : '');
  await writeFile(staged, output.endsWith('.gz') ? gzipSync(archiveText, { level: 9, mtime: 0 }) : archiveText,
    { flag: 'wx' });
  await rename(staged, output);
  const programsOutput = output.endsWith('.jsonl.gz') ? output.slice(0, -'.jsonl.gz'.length) + '.programs.jsonl' :
    `${output}.programs.jsonl`;
  const programsStaged = `${programsOutput}.building-${process.pid}-${randomUUID()}`;
  await writeFile(programsStaged, [...programs.values()].join('\n') + (programs.size ? '\n' : ''), { flag: 'wx' });
  await rename(programsStaged, programsOutput);
  await writeFile(`${output}.manifest.json`, JSON.stringify({ version: 'natlang.teacher_decision_archive_manifest/1',
    archive: relative(process.cwd(), output), archive_sha256: digest(await readFile(output)), root: relative(process.cwd(), root),
    recovered_programs: relative(process.cwd(), programsOutput),
    recovered_programs_sha256: digest(await readFile(programsOutput)),
    counts: { ...counts, recovered_program_revisions: programs.size }, source_files: sourceFiles }, null, 2) + '\n');
  console.log(JSON.stringify({ output, ...counts }));
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url))
  main().catch(error => { console.error(error instanceof Error ? error.stack : String(error)); process.exitCode = 1; });
