#!/usr/bin/env node
/** Execute authored crisp-skill searches and sealed paired evaluations. Keep every attempt. */
import { readFile, writeFile, mkdir, open, appendFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { scoreSkillObjective } from '../../dist/skills/objective.js';
import { authorSkillEpisode } from '../../dist/improvement/skill-authoring.js';
import { openAICompatibleModelTurn } from '../../dist/model/openai-compatible.js';
import { createPiModelBackend } from '../../dist/model/pi-provider.js';

const options = { limit: 4, experiments: 2, endpoint: 'http://127.0.0.1:8082',
  model: 'nvidia/Qwen3.6-35B-A3B-NVFP4' };
for (let i = 2; i < process.argv.length; i += 2) {
  const key = process.argv[i].replace(/^--/, ''), value = process.argv[i + 1];
  if (!['episodes', 'out', 'endpoint', 'model', 'limit', 'experiments', 'executor-endpoint', 'executor-model'].includes(key) || value === undefined)
    throw Error('Usage: collect-episodes.mjs --episodes FILE --out DIR [--endpoint URL|pi:PROVIDER --model ID --limit N --experiments N]');
  options[key] = ['limit', 'experiments'].includes(key) ? Number(value) : value;
}
if (!options.episodes || !options.out || !Number.isSafeInteger(options.limit) || options.limit < 1 ||
  !Number.isSafeInteger(options.experiments) || options.experiments < 1) throw Error('episodes/out and positive integer limits required');
const out = resolve(options.out), bytes = await readFile(options.episodes);
const sha = value => createHash('sha256').update(value).digest('hex');
const episodes = bytes.toString().split('\n').filter(line => line.trim()).map(JSON.parse)
  .filter(row => row.split === 'train').slice(0, options.limit);
if (!episodes.length || new Set(episodes.map(row => row.id)).size !== episodes.length) throw Error('nonempty unique train episodes required');
await mkdir(out, { recursive: true });
const runtimePath = fileURLToPath(new URL('../../', import.meta.url));
const sealBytes = await readFile(join(runtimePath,'frozen-runtime.json'));
const seal = JSON.parse(sealBytes);
if (seal.schema !== 'natlang.skill-authoring-runtime/1') throw Error('Use a separately frozen skill authoring runtime');
const runtime = {path:runtimePath,manifest_sha256:sha(sealBytes)};
const codePins = {};
for (const file of ['improvement/skill-authoring.js','improvement/program.js','improvement/host.js',
  'improvement/source-worker.js','improvement/authored-source.js','skills/objective.js','skills/registry.js',
  'runtime/kernel.js','native/agent.js','native/prompt.js']) {
  codePins[file] = sha(await readFile(new URL('../../dist/'+file, import.meta.url)));
}
codePins.collector = sha(await readFile(new URL(import.meta.url)));
for (const [relative,pin] of Object.entries(seal.files)) {
  if (sha(await readFile(join(runtimePath,relative))) !== pin) throw Error('Frozen runtime file changed: '+relative);
}
const identity = { runtime, codePins, node: process.version, version: 'natlang.skill-collection/1', input_sha256: sha(bytes),
  episode_ids: episodes.map(row => row.id), options: { ...options, episodes: resolve(options.episodes), out } };
const manifestPath = join(out, 'collection.json');
try {
  const old = JSON.parse(await readFile(manifestPath, 'utf8'));
  if (JSON.stringify(old) !== JSON.stringify(identity)) throw Error('collection resume requires identical inputs/options');
} catch (error) {
  if (error.code !== 'ENOENT') throw error;
  await writeFile(manifestPath, JSON.stringify(identity, null, 2) + '\n', { flag: 'wx' });
}
const lock = await open(join(out, 'active.lock'), 'wx');
const controller = new AbortController();
for (const signal of ['SIGINT', 'SIGTERM']) process.once(signal, () => controller.abort(new Error(signal)));
let positive = 0, evaluated = 0;
const backends = [];
function driver(endpoint, model, exchanges, file) {
  if (endpoint.startsWith('pi:')) {
    const backend = createPiModelBackend(endpoint.slice(3), model); backends.push(backend);
    return async (request, signal) => {
      const response = await backend.turn(request, signal);
      const exchange = {request,wireRequest:{messages:request.messages,tools:request.tools},wireResponse:{choices:[{message:{role:"assistant",content:response.text??null,reasoning_content:response.reasoning??null,tool_calls:response.raw_calls??[]}}]}};
      exchanges.push(exchange); await appendFile(file, JSON.stringify(exchange)+"\n"); return response;
    };
  }
  if (endpoint.includes('openrouter.ai')) throw Error('Use a reviewed free-provider launcher; this collector does not bypass its routing policy');
  return openAICompatibleModelTurn({ endpoint, model, apiKey: process.env.NATLANG_IMPROVEMENT_API_KEY,
    request: { temperature: 0.2 }, onExchange: async exchange => {exchanges.push(exchange); await appendFile(file,JSON.stringify(exchange)+"\n");} });
}
try {
  for (const episode of episodes) {
    if (controller.signal.aborted) break;
    const directory = join(out, sha(episode.id).slice(0, 20)); await mkdir(directory, { recursive: true });
    const resultPath = join(directory, 'result.json');
    try {
      const saved = JSON.parse(await readFile(resultPath, 'utf8'));
      if (saved.episode !== episode.id) throw Error('saved episode identity mismatch');
      if (saved.positive) positive++; if (saved.disposition === 'evaluated') evaluated++; continue;
    } catch (error) { if (error.code !== 'ENOENT') throw error; }
    const authorExchanges = [], executorExchanges = [];
    const author = driver(options.endpoint, options.model, authorExchanges, join(directory,"author-exchanges.jsonl"));
    const executorEndpoint = options['executor-endpoint'] ?? options.endpoint;
    const executorModel = options['executor-model'] ?? options.model;
    const executor = driver(executorEndpoint, executorModel, executorExchanges, join(directory,"executor-exchanges.jsonl"));
    let result;
    const metric = episode.provenance?.metric;
    if (metric && (metric.schema !== 'natlang.skill-objective/1' ||
        !['knapsack','bin-packing','weighted-tardiness'].includes(metric.kind))) throw Error('unsupported objective metric');
    const scoring = metric ? {identity: 'natlang.skill-objective/1:'+metric.kind+':'+codePins['skills/objective.js'],
      score: (row, output) => output.error ? {quality:0,gates:{completed:false}} :
        scoreSkillObjective(metric.kind,row.args[0],output.value,row.expected)} : undefined;
    const transferMetric = episode.provenance?.transfer_metric;
    if (metric && episode.transfer && (!transferMetric || transferMetric.schema !== 'natlang.skill-objective/1' ||
      !['knapsack','bin-packing','weighted-tardiness'].includes(transferMetric.kind))) throw Error('transfer requires its own objective metric');
    const transferScoring = transferMetric ? {identity:'natlang.skill-objective/1:'+transferMetric.kind+':'+codePins['skills/objective.js'],
      score:(row,output)=>output.error ? {quality:0,gates:{completed:false}} :
        scoreSkillObjective(transferMetric.kind,row.args[0],output.value,row.expected)} : undefined;
    const traces = [];
    try {
      result = await authorSkillEpisode({ episode, directory, author, executor,
        executorId: `${executorEndpoint}:${executorModel}`, signal: controller.signal,
        maxExperiments: options.experiments, scoring, transferScoring,
        scoringDescriptor: metric ?? {kind:"exact-return-and-files"}, trace: trace => traces.push(trace),
        onSearch: async search => {await writeFile(join(directory,"search-result.json"),JSON.stringify(search,null,2)+"\n");},
        searchBudget: { maxModelCalls: 400, maxRollouts: 160, maxProposals: options.experiments * 2 },
        evaluationBudget: { maxModelCalls: 400, maxRollouts: 160, maxProposals: 0 } });
    } catch (error) {
      result = { version: 'natlang.skill-authoring-trajectory/1', episode: episode.id, split: episode.split,
        disposition: controller.signal.aborted ? 'interrupted' : 'failed', positive: false, error: String(error) };
    }
    const artifact = { ...result, runtime, codePins, collection_sha256: sha(JSON.stringify(identity)),
      author_identity: `${options.endpoint}:${options.model}`, executor_identity: `${executorEndpoint}:${executorModel}`,
      authorExchanges, executorExchanges, traces: result.traces ?? traces };
    await writeFile(resultPath, JSON.stringify(artifact, null, 2) + '\n', { flag: 'wx' });
    if (result.positive) positive++; if (result.disposition === 'evaluated') evaluated++;
    console.log(JSON.stringify({ episode: episode.id, disposition: result.disposition, positive: result.positive,
      query_gain: result.query?.effect, transfer_gain: result.transfer?.effect, author_requests: authorExchanges.length,
      executor_requests: executorExchanges.length }));
  }
  await writeFile(join(out, 'summary.json'), JSON.stringify({ episodes: episodes.length, evaluated, positive,
    interrupted: controller.signal.aborted, publication: 'Unpublished candidates; all failed and neutral attempts retained' }, null, 2) + '\n');
} finally {
  for (const backend of backends) backend.close();
  await lock.close();
  const { unlink } = await import('node:fs/promises'); await unlink(join(out, 'active.lock'));
}
