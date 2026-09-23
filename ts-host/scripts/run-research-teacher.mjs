#!/usr/bin/env node
/** Execute one Inquiry Lab scenario through an OpenAI-compatible teacher.
 * This runner never admits training data: it saves the full trajectory,
 * executable workspace, receipts and an explicitly unreviewed semantic rubric.
 */
import { mkdir, readFile, readdir, rename, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { MemoryResearchAdapter } from '../studio/shared/research-workspace.mjs';
import { ResearchHost } from '../studio/research/host.mjs';
import { scenarios, scenarioById } from '../studio/research/scenarios.mjs';
import { auditResearchBundle } from '../studio/research/evaluation.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const studio = resolve(here, '../studio/research');

async function api() {
    const processValue = globalThis.process;
    try { globalThis.process = undefined; return await import('../dist/browser/natlang.js'); }
    finally { globalThis.process = processValue; }
}
function decode(raw) { try { return JSON.parse(raw || '{}'); } catch { return { __unparsed__: raw }; } }
function teacherDriver(server, exchanges) {
    return async request => {
        const tools = structuredClone(request.tools);
        for (const tool of tools)
            for (const key of Object.keys(tool.function.parameters ?? {})) if (key.startsWith('x-')) delete tool.function.parameters[key];
        const messages = structuredClone(request.messages);
        const payload = { messages, tools, tool_choice: 'auto', parallel_tool_calls: true,
            temperature: request.temperature, seed: request.seed, top_p: .95, top_k: 20,
            thinking_budget_tokens: 512, chat_template_kwargs: { reasoning_effort: 'low' } };
        if (request.max_tokens !== null) payload.max_tokens = request.max_tokens;
        const response = await fetch(`${server}/v1/chat/completions`, { method: 'POST',
            headers: { 'content-type': 'application/json' }, body: JSON.stringify(payload) });
        const body = await response.json();
        if (!response.ok) throw new Error(`teacher HTTP ${response.status}: ${JSON.stringify(body).slice(0, 2000)}`);
        const message = body.choices?.[0]?.message ?? {};
        const calls = (message.tool_calls ?? []).map(call => {
            const name = call.function?.name;
            const args = decode(call.function?.arguments);
            return [name, args];
        });
        exchanges.push({ request, response: body, assistant: { content: String(message.content ?? ''),
            reasoning: message.reasoning_content ?? message.reasoning ?? message.thinking ?? null,
            calls: calls.map(([tool, arguments_]) => ({ tool, arguments: arguments_ })) } });
        return { calls, text: String(message.content ?? ''), completion_tokens: body.usage?.completion_tokens,
            prompt_tokens: body.usage?.prompt_tokens, raw_response: body };
    };
}

class ResearchMemoryStore extends MemoryResearchAdapter {
    constructor() { super(); this.values = new Map(); }
    async get(store, id) { return store === 'native_values' ? structuredClone(this.values.get(id)) : undefined; }
    async put(store, id, value) { if (store !== 'native_values') throw new Error(`Unsupported memory store ${store}`); this.values.set(id, structuredClone(value)); }
}

async function programFiles() {
    const names = ['types.ts', 'reduce.nl', 'view.nl', 'view.ts', 'learn.nl', 'revise_schema.nl',
        'invent_interaction.nl', 'preserve_intent.nl', 'investigate_beliefs.nl'];
    for (const name of await readdir(resolve(studio, 'programs/reduce'))) names.push(`reduce/${name}`);
    return Object.fromEntries(await Promise.all(names.sort().map(async name => [name,
        await readFile(resolve(studio, 'programs', name), 'utf8')])));
}

async function runScenario({ scenario, server, model, seed }) {
    const bindings = await api(), exchanges = [], driver = teacherDriver(server, exchanges);
    const store = new ResearchMemoryStore(); let childIndex = 0, state;
    const research = new ResearchHost({ store, runContext: () => ({ model, seed: { mode: 'derived', root: seed }, evaluator: 'typescript-host/browser' }),
        runSource: async (files, root, inputs, options) => {
            const child = new bindings.BrowserNatlangClient({ host: { research: { readNative: async id => (await store.get('native_values', id))?.value } } });
            try {
                const result = await child.run({ source: { kind: 'files', root, files: Object.fromEntries(files.map(row => [row.id, row.source])) },
                    inputs, modelTurn: driver, validationFeedback: 'local', options: { seed: options.provenance.seed } });
                if (result.outcome.kind !== 'done') throw new Error(result.outcome.detail);
                return { value: result.value, trace_id: `child-${++childIndex}` };
            } finally { await child.close(); }
        } });
    const sources = await programFiles();
    const observations = JSON.parse(await readFile(resolve(studio, 'sample/observations.json'), 'utf8'));
    const methods = await readFile(resolve(studio, 'sample/methods.txt'), 'utf8');
    const head = await research.runtime.commit('', {
        ...Object.fromEntries(Object.entries(sources).map(([path, content]) => [path, { kind: path === 'types.ts' ? 'schema' : 'source', content }])),
        'evidence/reliability-observations.json': { kind: 'evidence', content: observations },
        'evidence/reliability-methods.txt': { kind: 'evidence', content: methods },
    }, { message: 'Frozen Inquiry Lab teacher scenario' });
    state = { revision: 0, head: head.id, question: '', notice: 'Ready.', active_view: '', selected: '', receipts: [] };
    const client = new bindings.BrowserNatlangClient({ host: { research: research.api() }, mode: 'retained' });
    const app = new bindings.BrowserNatlangApplication({ client,
        source: { files: sources, reducer: 'reduce.nl', view: 'view.ts' }, initialState: state, seedRoot: seed, modelTurn: driver,
        validationFeedback: 'local',
        onCommit: async commit => { await research.verifyCommit(state, commit.state); state = commit.state; },
    });
    let transition, failure = '';
    const event = { id: `scenario-${scenario.id}-${seed}`, kind: 'question', value: scenario.question };
    try {
        await app.start(); research.begin(event); transition = await app.dispatch(event);
    } catch (error) {
        failure = String(error);
    } finally { research.end(); await app.close(); await client.close(); }
    const workspace = await research.runtime.workspace.export();
    const bundle = { format: 1, workspace, state };
    const audit = await auditResearchBundle(bundle, scenario.id);
    return { schema: 'natlang.research_teacher_trajectory/1', scenario, provenance: { model, seed, server,
        source_manifest: head.id, evaluator: 'typescript-host/browser' },
        outcome: { completed: Boolean(transition), error: failure,
            reduction: transition?.reducerRun?.outcome ?? null, structural: audit.structural,
            semantic_review: audit.semantic_review, training_admission: audit.training_admission },
        runs: transition ? { reducer: transition.reducerRun, view: transition.viewRun } : null, exchanges, bundle };
}

async function atomic(path, value) {
    await mkdir(dirname(path), { recursive: true }); const temporary = `${path}.building`;
    await writeFile(temporary, JSON.stringify(value) + '\n'); await rename(temporary, path);
}
async function main() {
    const args = process.argv.slice(2), take = flag => { const at = args.indexOf(flag); return at < 0 ? null : args[at + 1]; };
    if (args.includes('--list')) { for (const scenario of scenarios) console.log(`${scenario.id}\t${scenario.capability}\t${scenario.question}`); return; }
    const id = take('--scenario'), model = take('--model'), server = take('--server') ?? 'http://127.0.0.1:8081';
    const seed = Number(take('--seed')), out = take('--out');
    const scenario = scenarioById.get(id);
    if (!scenario || !model || !Number.isSafeInteger(seed) || !out)
        throw new Error('usage: run-research-teacher.mjs --scenario ID --model ID --seed N --out FILE [--server URL]');
    const result = await runScenario({ scenario, server, model, seed });
    await atomic(resolve(out), result);
    console.log(`${scenario.id}: completed=${result.outcome.completed}; structural=${result.outcome.structural.passed}; semantic=unreviewed -> ${resolve(out)}`);
    if (!result.outcome.completed) process.exitCode = 2;
}
await main();
