#!/usr/bin/env node
import { spawnSync } from 'node:child_process';
import { mkdir, writeFile, lstat } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const git = (args, cwd) => {
  const result = spawnSync('git', args, { cwd, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
  if (result.status !== 0) throw new Error((result.stderr || result.stdout || `git ${args[0]} failed`).trim());
  return result.stdout.trim();
};

const gitSource = (id, url, license, patterns, readiness, adapter = 'source inventory only') => ({
  id, type: 'git', url, license, expected: patterns, adapterReadiness: readiness, adapter,
});
const hfSource = (id, url, dataset, license, patterns, readiness, adapter, { config = 'default', split = 'train' } = {}) => ({
  id, type: 'huggingface', url, dataset, license, expected: patterns, adapterReadiness: readiness, adapter,
  defaultConfig: config, defaultSplit: split,
  acquisition: 'bounded datasets-server rows endpoint; requires accessible dataset-server config',
});

const huggingFaceOrigins = new Set(['https://huggingface.co', 'https://datasets-server.huggingface.co']);
export function huggingFaceRequestHeaders(input, env = process.env) {
  const url = input instanceof URL ? input : new URL(input);
  const token = env.HF_TOKEN || env.HUGGINGFACE_HUB_TOKEN;
  return huggingFaceOrigins.has(url.origin) && token
    ? { Authorization: `Bearer ${token}` }
    : {};
}

export const SOURCES = Object.freeze([
  gitSource('es-toolkit', 'https://github.com/toss/es-toolkit.git', 'MIT', ['src/**/*.ts', 'tests/**/*', 'docs/**/*.md'], 'generic', 'Generic TypeScript extraction available; upstream Vitest runner integration deferred'),
  gitSource('radashi', 'https://github.com/radashi-org/radashi.git', 'MIT', ['src/**/*.ts', 'tests/**/*', 'docs/**/*.md'], 'generic', 'Generic TypeScript extraction available; upstream test runner integration deferred'),
  gitSource('simple-statistics', 'https://github.com/simple-statistics/simple-statistics.git', 'ISC', ['src/**/*.js', 'src/**/*.md', 'test/**/*.js'], 'generic', 'Generic JavaScript extraction available; upstream test runner integration deferred'),
  gitSource('exercism-typescript', 'https://github.com/exercism/typescript.git', 'MIT', ['exercises/{concept,practice}/**/.docs/instructions.md', 'exercises/**/.meta/config.json', 'exercises/**/*.ts', 'exercises/**/*.test.ts'], 'generic', 'Exercism inventory and generic extraction available; track test runner integration deferred'),
  gitSource('exercism-javascript', 'https://github.com/exercism/javascript.git', 'MIT', ['exercises/{concept,practice}/**/.docs/instructions.md', 'exercises/**/.meta/config.json', 'exercises/**/*.js', 'exercises/**/*.spec.js'], 'generic', 'Exercism inventory and generic extraction available; track test runner integration deferred'),
  gitSource('problem-specifications', 'https://github.com/exercism/problem-specifications.git', 'MIT', ['exercises/*/canonical-data.json', 'exercises/*/description.md'], 'inventory', 'Canonical problem data inventory only; no runtime test integration'),
  gitSource('remeda', 'https://github.com/remeda/remeda.git', 'MIT', ['packages/remeda/src/**/*.ts', 'packages/remeda/**/*.test.ts', 'website/docs/**/*.md'], 'generic', 'Generic TypeScript extraction available; upstream runner integration and curry-wrapper normalization deferred'),
  gitSource('ramda', 'https://github.com/ramda/ramda.git', 'MIT', ['source/**/*.js', 'test/**/*.js', 'docs/**/*.md'], 'generic', 'Generic JavaScript extraction available; upstream runner integration and curry-wrapper normalization deferred'),
  hfSource('codesearchnet', 'https://huggingface.co/datasets/code-search-net/code_search_net', 'code-search-net/code_search_net', 'Apache-2.0', ['code', 'docstring', 'repository'], 'ready', 'Bounded JavaScript split importer; preserves original code and docstring, no execution claim', { config: 'javascript' }),
  hfSource('magicoder', 'https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K', 'ise-uiuc/Magicoder-OSS-Instruct-75K', 'MIT', ['lang', 'problem', 'solution'], 'ready', 'Bounded JS/TS importer; preserves original response, no execution claim'),
  hfSource('mceval', 'https://huggingface.co/datasets/Multilingual-Multimodal-NLP/McEval-Instruct', 'Multilingual-Multimodal-NLP/McEval-Instruct', 'CC-BY-SA-4.0', ['language', 'instruction', 'output'], 'ready', 'Bounded JS/TS instruction importer; evaluation dataset remains separate'),
  gitSource('deno-std', 'https://github.com/denoland/std.git', 'MIT', ['**/*.ts', '**/*.md'], 'generic', 'Generic TypeScript extraction available; nearest README context used for undocumented functions; upstream test integration deferred'),
  gitSource('javascript-algorithms', 'https://github.com/trekhleb/javascript-algorithms.git', 'MIT', ['src/**/*.js', 'README.md', 'src/**/README.md'], 'generic', 'Generic JavaScript extraction available; nearest README context used for undocumented functions; upstream test integration deferred'),
  gitSource('d3-array', 'https://github.com/d3/d3-array.git', 'ISC', ['src/**/*.js', 'README.md'], 'generic', 'Generic JavaScript export inventory includes named and default exports; upstream test integration deferred'),
  gitSource('30-seconds-of-code', 'https://github.com/Chalarangelo/30-seconds-of-code.git', 'CC-BY-4.0', ['content/**/*.md', 'README.md'], 'generic', 'Markdown frontmatter and JavaScript fences become instruction-backed inventory tasks; snippets are never executed'),
  hfSource('case2code', 'https://huggingface.co/datasets/OpenMOSS-Team/case2code-data', 'OpenMOSS-Team/case2code-data', 'CC-BY-NC-4.0', ['prompt', 'Python code', 'inputs', 'outputs'], 'ready', 'Local bounded importer adapter; Python preservation/translation only, no execution claim'),
  hfSource('xlam', 'https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k', 'Salesforce/xlam-function-calling-60k', 'CC-BY-4.0', ['tools', 'query', 'answers'], 'ready', 'Local bounded importer and schema-ordered call-view adapter; calls are not executed', { config: 'dataset' }),
  hfSource('tiny-codes', 'https://huggingface.co/datasets/nampdn-ai/tiny-codes', 'nampdn-ai/tiny-codes', 'MIT', ['instruction', 'code'], 'ready', 'Local bounded importer adapter; language must be explicit, code is not executed'),
  hfSource('stack-edu', 'https://huggingface.co/datasets/HuggingFaceTB/stack-edu', 'HuggingFaceTB/stack-edu', 'see dataset metadata', ['raw source code', 'Software Heritage identifiers'], 'deferred', 'Bounded sample only; SWH resolution and extraction deferred'),
  gitSource('spoc', 'https://github.com/sumith1896/spoc.git', 'see dataset/repository terms', ['**/*.tsv', '**/*.cpp'], 'deferred', 'SPoC pseudocode/C++ alignment adapter deferred'),
  hfSource('codeact', 'https://huggingface.co/datasets/xingyaoww/code-act', 'xingyaoww/code-act', 'Apache-2.0', ['id', 'conversations'], 'deferred', 'Bounded sample only; multi-turn CodeAct interaction normalization deferred', { split: 'codeact' }),
  hfSource('code-feedback', 'https://huggingface.co/datasets/m-a-p/Code-Feedback', 'm-a-p/Code-Feedback', 'Apache-2.0', ['id', 'messages'], 'deferred', 'Bounded sample only; multi-turn feedback extraction deferred'),
  hfSource('commitpackft', 'https://huggingface.co/datasets/bigcode/commitpackft', 'bigcode/commitpackft', 'see dataset metadata', ['commit diffs', 'source snapshots'], 'deferred', 'Bounded sample only; edit-task extraction deferred'),
  gitSource('bugsjs', 'https://github.com/BugsJS/bug-dataset.git', 'see repository terms', ['Projects/**/*.csv', 'Projects/**/*.bin'], 'deferred', 'BugsJS benchmark inventory only; buggy/fixed extraction deferred'),
  hfSource('menvdata', 'https://huggingface.co/datasets/ernie-research/MEnvData-SWE', 'ernie-research/MEnvData-SWE', 'see dataset metadata', ['repository tasks', 'patches', 'tests'], 'deferred', 'Bounded sample only; trajectory extraction deferred'),
]);

export function resolveGitCommit(source, revision, runGit = git) {
  if (/^[0-9a-f]{40}$/i.test(revision)) return revision.toLowerCase();
  const ref = runGit(['ls-remote', source.url, revision === 'HEAD' ? 'HEAD' : revision]);
  const commit = ref.split(/\s+/)[0];
  if (!/^[0-9a-f]{40}$/i.test(commit)) throw new Error(`Could not resolve revision ${revision}`);
  return commit;
}

export async function acquireGitSource(id, destination, { revision = 'HEAD' } = {}) {
  const source = SOURCES.find((item) => item.id === id);
  if (!source || source.type !== 'git') throw new Error(`Unknown git source: ${id}`);
  const target = resolve(destination);
  try { await lstat(target); throw new Error(`Destination already exists: ${target}`); }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
  const commit = resolveGitCommit(source, revision);
  await mkdir(dirname(target), { recursive: true });
  try {
    git(['clone', '--no-checkout', '--filter=blob:none', source.url, target]);
    git(['fetch', '--filter=blob:none', 'origin', commit], target);
    git(['checkout', '--detach', commit], target);
    const manifest = { source: id, url: source.url, requestedRevision: revision, commit, license: source.license, adapterReadiness: source.adapterReadiness, fetchedAt: new Date().toISOString(), scriptsExecuted: false };
    await writeFile(resolve(target, '.code-corpus-source.json'), `${JSON.stringify(manifest, null, 2)}\n`, { flag: 'wx' });
    return manifest;
  } catch (error) {
    // Leave an incomplete destination visible for recovery/inspection; never recursively erase user data.
    throw new Error(`Acquisition failed at ${target}: ${error.message}`);
  }
}

export async function fetchHuggingFaceRows(id, { config = 'default', split = 'train', offset = 0, length = 100, signal } = {}) {
  const source = SOURCES.find((item) => item.id === id);
  if (!source || source.type !== 'huggingface') throw new Error(`Unknown Hugging Face source: ${id}`);
  if (!Number.isInteger(offset) || offset < 0 || !Number.isInteger(length) || length < 1 || length > 1000) throw new Error('Rows request must use offset >= 0 and length between 1 and 1000');
  const url = new URL(`https://datasets-server.huggingface.co/rows?dataset=${encodeURIComponent(source.dataset)}&config=${encodeURIComponent(config)}&split=${encodeURIComponent(split)}&offset=${offset}&length=${length}`);
  const response = await fetch(url, { signal, headers: huggingFaceRequestHeaders(url) });
  if (!response.ok) throw new Error(`Hugging Face rows unavailable (${response.status}); dataset may be gated or lack a datasets-server config`);
  const data = await response.json();
  if (!Array.isArray(data.rows)) throw new Error('Unexpected datasets-server rows response');
  return data.rows;
}

export async function main(argv = process.argv.slice(2)) {
  const [command, ...args] = argv;
  if (command === 'list') {
    for (const s of SOURCES) console.log(`${s.id}\t${s.type}\t${s.adapterReadiness}\t${s.url}`);
    return;
  }
  if (command === 'fetch') {
    const [id, destination, ...opts] = args;
    if (!id || !destination) throw new Error('Usage: sources.mjs fetch <source-id> <new-destination> [--revision <git-ref>]');
    const pos = opts.indexOf('--revision');
    if (pos >= 0 && !opts[pos + 1]) throw new Error('--revision requires a git ref');
    const manifest = await acquireGitSource(id, destination, { revision: pos >= 0 ? opts[pos + 1] : 'HEAD' });
    console.log(JSON.stringify(manifest, null, 2));
    return;
  }
  if (command === 'hf-rows') {
    const [id, ...opts] = args;
    const option = (name, fallback) => { const i = opts.indexOf(name); return i < 0 ? fallback : opts[i + 1]; };
    const source = SOURCES.find((item) => item.id === id);
    const rows = await fetchHuggingFaceRows(id, { config: option('--config', source?.defaultConfig ?? 'default'), split: option('--split', source?.defaultSplit ?? 'train'), offset: Number(option('--offset', 0)), length: Number(option('--length', 100)) });
    for (const row of rows) console.log(JSON.stringify(row));
    return;
  }
  throw new Error('Usage: sources.mjs list | fetch <id> <destination> [--revision ref] | hf-rows <id> [--config name] [--split name] [--offset n] [--length n]');
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => { console.error(error.message); process.exitCode = 1; });
}
