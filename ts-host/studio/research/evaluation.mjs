import { MemoryResearchAdapter, ResearchWorkspace } from '../shared/research-workspace.mjs';
import { ResearchRuntime } from '../shared/research-runtime.mjs';
import { scenarioById } from './scenarios.mjs';

/** Structural evidence gate. Semantic judgments remain explicit review items. */
export async function auditResearchBundle(bundle, scenarioId) {
    const scenario = scenarioById.get(scenarioId);
    if (!scenario) throw new Error(`Unknown research scenario ${scenarioId}`);
    if (bundle?.format !== 1 || bundle.state?.head !== bundle.workspace?.head)
        throw new Error('State and workspace bundle disagree');
    const adapter = new MemoryResearchAdapter(), workspace = new ResearchWorkspace(adapter);
    await workspace.import(bundle.workspace);
    const runtime = new ResearchRuntime({ adapter, runSource: async () => { throw new Error('Audit does not execute source'); } });
    const head = bundle.state.head, entries = await runtime.list(head), graph = await runtime.beliefGraph(head);
    const kinds = Object.groupBy(entries, row => row.kind);
    const receipts = Object.values(bundle.workspace.receipts ?? {});
    const completed = receipts.filter(row => row.status === 'complete');
    const checks = [
        { name: 'versioned state', pass: Boolean(head && bundle.workspace.manifests[head]) },
        { name: 'raw evidence retained', pass: (kinds.evidence?.length ?? 0) > 0 },
    ];
    if (!['irrelevant_update', 'incompatible_merges'].includes(scenario.id))
        checks.push({ name: 'actual execution receipts', pass: completed.length > 0 });
    if (scenario.capability === 'learn') checks.push(
        { name: 'new executable method', pass: entries.some(row => row.kind === 'source' && !row.path.startsWith('reduce/') && !['reduce.nl','view.nl'].includes(row.path) && !['learn.nl','revise_schema.nl','invent_interaction.nl','preserve_intent.nl','investigate_beliefs.nl'].includes(row.path)) },
        { name: 'method described', pass: (kinds.method?.length ?? 0) > 0 });
    if (scenario.capability === 'schema') checks.push(
        { name: 'migration source or record', pass: (kinds.migration?.length ?? 0) > 0 },
        { name: 'revised schema', pass: (kinds.schema?.length ?? 0) > 1 });
    if (scenario.capability === 'interaction') {
        const views = entries.filter(row => row.kind === 'view');
        let valid = false;
        for (const view of views) try { await runtime.interaction(head, view.path); valid = true; } catch { /* reported by failed gate */ }
        checks.push({ name: 'bound generated view', pass: valid });
    }
    if (scenario.capability === 'intent') checks.push(
        { name: 'intent artifacts', pass: (kinds.intent?.length ?? 0) > 0 },
        { name: 'candidate branches', pass: (await runtime.branches()).some(row => row.kind === 'candidate') });
    if (scenario.capability === 'beliefs') checks.push(
        { name: 'evidence-linked claims', pass: graph.nodes.some(row => row.kind === 'claim') && graph.links.length > 0 },
        { name: 'missing links identified', pass: graph.missing.length === 0 });
    return { scenario: scenario.id, structural: { passed: checks.every(check => check.pass), checks },
        observations: { artifacts: entries.length, claims: graph.nodes.filter(row => row.kind === 'claim').length,
            links: graph.links.length, receipts: receipts.length, completed: completed.length,
            imported_receipts: receipts.filter(row => row.imported).length },
        semantic_review: scenario.review.map(question => ({ question, result: 'unreviewed' })),
        training_admission: 'requires independent semantic review and provenance inspection' };
}
