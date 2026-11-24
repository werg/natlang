#!/usr/bin/env node
// Direct controls of authored verifier tasks (`oracle: "verifier"`, scripts/rl_tasks/sources/authored.py), without a
// model or a program run: each task's verifier module is loaded through the runtime's own trusted-verifier binding
// (allow-listed root, sha256, verifyCandidate contract: ts-host/src/teacher/trusted-exact-verifier.ts) and judged on
//   reference  `expected` (with `expected_folders` as the files the run leaves)    must be accepted
//   aliases    every `alias_answers` entry (other valid outcomes)                   must be accepted
//   wrong      every `wrong_answers` entry                                          must be rejected
//   keyword    `keyword_baseline.code` run as an async function of the inputs      must be rejected
//              (a baseline that reads the world folder runs only in verify.mjs's replay: recorded as skipped)
//   degenerate null, {}, [], "" and 0                                               must be rejected
//   repeat     the reference judged twice                                           same verdict and score
// usage: node verifier_controls.mjs RUNTIME_TS_HOST TASKS.jsonl [ANSWERS.json]
// Prints {task id: {errors: [...], verdicts: {...}}}; with ANSWERS ({id: {answer, folders}}) it judges those answers
// instead and prints {task id: {accepted, score, issues}}. Repository root: NATLANG_REPO_ROOT, else the working directory.
import fs from 'node:fs';
import crypto from 'node:crypto';
import path from 'node:path';

const [runtime, tasksPath, answersPath] = process.argv.slice(2);
if (!runtime || !tasksPath) { console.error('usage: verifier_controls.mjs RUNTIME TASKS.jsonl [ANSWERS.json]'); process.exit(2); }
const { checkTrustedExactVerifier } = await import(path.resolve(runtime, 'dist/teacher/trusted-exact-verifier.js'));
const root = path.resolve(process.env.NATLANG_REPO_ROOT ?? process.cwd());
const tasks = fs.readFileSync(tasksPath, 'utf8').split('\n').filter(line => line.trim()).map(line => JSON.parse(line))
  .filter(task => task.oracle === 'verifier');

function spec(task) {
  const file = path.resolve(root, task.verifier.path);
  return { path: task.verifier.path, sha256: crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex'),
    oracle: task.verifier.oracle };
}
async function judge(task, value, files) {
  try {
    const verdict = await checkTrustedExactVerifier(spec(task), { value, oracle: task.verifier.oracle,
      inputs: task.inputs ?? {}, fixture: task.folders ?? {}, ...(files ? { files } : {}) });
    return { accepted: verdict.accepted, score: verdict.score ?? null, issues: verdict.verdict ?? '' };
  } catch (error) { return { accepted: null, error: String(error?.message ?? error).slice(0, 300) }; }
}
// Read-only lookup over `task.files` matching the runtime's `read_code(name)` contract for static keyword baselines.
function makeReadCode(task) {
  const files = task.files ?? {};
  const rootPath = typeof task.root === 'string' ? task.root : '';
  const rootDir = rootPath.endsWith('.nl') ? rootPath.slice(0, -3) : rootPath.replace(/\.[^/.]+$/, '');
  return function read_code(input) {
    const raw = typeof input === 'string' ? input
      : (input && typeof input === 'object' && !Array.isArray(input) && typeof input.name === 'string' ? input.name : null);
    if (raw === null) throw new Error('read_code expects a name string or { name: string }');
    const clean = raw.replace(/^\.\//, '');
    if (Object.hasOwn(files, clean)) return files[clean];
    const candidates = [];
    if (clean.startsWith('skills.') || clean.startsWith('skills/')) {
      const skill = clean.slice('skills.'.length);
      if (rootDir) candidates.push(`${rootDir}/skills/${skill}/SKILL.md`, `${rootDir}/skills/${skill}`);
      candidates.push(`skills/${skill}/SKILL.md`, `skills/${skill}`);
    }
    const slashed = clean.includes('/') ? clean : clean.replace(/\./g, '/');
    const exts = ['', '.nl', '.ts', '.js', '.mjs', '.md'];
    for (const base of new Set([clean, slashed])) {
      if (rootDir) {
        for (const ext of exts) candidates.push(`${rootDir}/${base}${ext}`);
        candidates.push(`${rootDir}/skills/${base}/SKILL.md`);
      }
      for (const ext of exts) candidates.push(`${base}${ext}`);
      candidates.push(`skills/${base}/SKILL.md`);
    }
    for (const candidate of candidates) {
      if (Object.hasOwn(files, candidate)) return files[candidate];
    }
    // Unique suffix/stem match over .nl/.ts/skill files in task.files.
    const matches = Object.keys(files).filter(filePath => {
      if (filePath.endsWith(`/skills/${clean}/SKILL.md`) || filePath === `skills/${clean}/SKILL.md`) return true;
      const base = filePath.split('/').pop() ?? '';
      return base === clean || base.replace(/\.(?:nl|ts|js|mjs)$/, '') === clean;
    });
    if (matches.length === 1) return files[matches[0]];
    throw new Error(`read_code: no such function or file: ${raw}`);
  };
}

// The baseline sees the root's inputs and a read-only `read_code` over `task.files`, as a root eval does; any helper,
// nl or decide call is a ReferenceError, so a baseline that is secretly semantic cannot pass as a keyword rule.
async function runBaseline(code, inputs, task) {
  const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
  const names = Object.keys(inputs);
  const args = names.map(name => structuredClone(inputs[name]));
  if (!names.includes('read_code')) {
    names.push('read_code');
    args.push(makeReadCode(task));
  }
  return await new AsyncFunction(...names, code)(...args);
}

const out = {};
if (answersPath) {
  const answers = JSON.parse(fs.readFileSync(answersPath, 'utf8'));
  for (const task of tasks) {
    const given = answers[task.id];
    out[task.id] = given && 'answer' in given ? await judge(task, given.answer, given.folders)
      : { accepted: null, error: 'no answer' };
  }
  console.log(JSON.stringify(out));
  process.exit(0);
}
for (const task of tasks) {
  const errors = [], verdicts = {};
  const files = task.expected_folders ?? undefined;
  const reference = verdicts.reference = await judge(task, task.expected, files);
  if (reference.accepted !== true)
    errors.push('verifier rejects the reference answer' + (reference.error ? ` (${reference.error})` : `: ${reference.issues}`));
  const again = await judge(task, task.expected, files);
  if (again.accepted !== reference.accepted || again.score !== reference.score)
    errors.push('verifier is not deterministic on the reference answer');
  verdicts.aliases = [];
  for (const [index, alias] of (task.alias_answers ?? []).entries()) {
    const verdict = await judge(task, alias, files);
    verdicts.aliases.push(verdict);
    if (verdict.accepted !== true) errors.push(`verifier rejects alias answer ${index}: ${verdict.issues ?? verdict.error}`);
  }
  verdicts.wrong = [];
  for (const [index, wrong] of (task.wrong_answers ?? []).entries()) {
    const verdict = await judge(task, wrong, files);
    verdicts.wrong.push(verdict);
    if (verdict.accepted !== false)
      errors.push(`non-discriminating verifier: wrong answer ${index} is ${verdict.accepted === null ? 'an error' : 'accepted'}`);
  }
  verdicts.degenerate = [];
  for (const value of [null, {}, [], '', 0]) {
    const verdict = await judge(task, value, files);
    verdicts.degenerate.push(verdict);
    if (verdict.accepted === true) errors.push(`non-discriminating verifier: accepts ${JSON.stringify(value)}`);
  }
  const code = task.keyword_baseline?.code;
  if (typeof code !== 'string') errors.push('missing keyword_baseline.code');
  else if (/\bfolder\b/.test(code) || task.folders) verdicts.keyword_baseline = { skipped: 'reads the world folder; judged by verify.mjs replay' };
  else {
    let value;
    try { value = await runBaseline(code, task.inputs ?? {}, task); }
    catch (error) {
      errors.push(`keyword baseline does not run on the inputs alone: ${String(error?.message ?? error).slice(0, 200)}`);
    }
    if (!errors.some(error => error.startsWith('keyword baseline'))) {
      const verdict = verdicts.keyword_baseline = { value, ...(await judge(task, value, files)) };
      if (verdict.accepted !== false)
        errors.push(`keyword baseline is ${verdict.accepted === null ? 'an error' : 'accepted'}: the verifier rewards surface cues`);
    }
  }
  out[task.id] = { errors, verdicts };
}
console.log(JSON.stringify(out));
