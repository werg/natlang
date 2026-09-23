#!/usr/bin/env node
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { syntheticRecords, NATIVE_SYNTHETIC_FAMILIES } from '../../dist/teacher/synthetic-generator.js';
import { digest, readJsonl, writeJsonl } from './common.mjs';

export async function teacherSeeds(input, output, count = 1000, seed = 42, failureInput = null) {
  const captured = await readJsonl(input);
  const failures = failureInput ? await readJsonl(failureInput) : [];
  const generated = syntheticRecords(seed, 0, count, [...NATIVE_SYNTHETIC_FAMILIES]).map(row => {
    const name = row.semantics?.root?.$lambda?.function ?? row.family;
    const group = `natlang-synthetic-code:${name}`;
    return { ...row, split:parseInt(digest(group).slice(0,8),16)%100 < 5 ? 'test' : 'train',
      source_groups:[group], curriculum_evidence:'generated_reference_requires_teacher_execution' };
  });
  // Interleave code tasks with composed workflows so a bounded teacher run sees both.
  const rows = [], ids = new Set();
  for (const row of failures) {
    if (ids.has(row.id)) throw new Error(`duplicate failure program id: ${row.id}`);
    ids.add(row.id); rows.push(row);
  }
  for (let i=0; i<Math.max(captured.length, generated.length); i++) for (const row of [captured[i], generated[i]]) {
    if (!row || ids.has(row.id)) continue;
    ids.add(row.id); rows.push(row);
  }
  const identity = {version:'natlang.teacher_seeds/1', input_sha256:digest(captured),
    failure_sha256:digest(failures), count, seed,
    generator_sha256:digest(await readFile(new URL('../../dist/teacher/synthetic-generator.js', import.meta.url), 'utf8'))};
  try {
    const [old] = await readJsonl(`${output}.manifest.jsonl`);
    if (digest(old.identity)!==digest(identity) || old.output_sha256!==digest(await readJsonl(output))) throw new Error('Teacher seed content/config changed');
    return old;
  } catch(error) { if(error.code!=='ENOENT') throw error; }
  // Recover the data/manifest commit window only when bytes describe the same rows.
  try { if(digest(await readJsonl(output))!==digest(rows)) throw new Error('Refusing to overwrite different teacher seeds'); }
  catch(error) { if(error.code!=='ENOENT') throw error; await writeJsonl(output, rows); }
  const manifest = {identity, output_sha256:digest(rows), programs:rows.length};
  await writeJsonl(`${output}.manifest.jsonl`, [manifest]);
  return manifest;
}
if(process.argv[1] && import.meta.url===pathToFileURL(resolve(process.argv[1])).href) {
  const [input, output, count='1000', seed='42', failureInput] = process.argv.slice(2);
  if(!input||!output) throw new Error('Usage: teacher-seeds.mjs PROJECTED_PROGRAMS OUTPUT [COUNT] [SEED] [FAILURE_CASES]');
  if(!Number.isSafeInteger(Number(count))||Number(count)<0) throw new Error('COUNT must be nonnegative');
  console.log(JSON.stringify(await teacherSeeds(input, output, Number(count), Number(seed), failureInput)));
}
