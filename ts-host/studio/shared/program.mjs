import { controlDecision } from './contracts.mjs';
/** Bundle real natlang sources with an app's portable types and host services. */
export const sourceNames = ['reduce.nl', 'reduce/choose.nl', 'reduce/perform.ts', 'reduce/finish.ts', 'view.nl', 'types.ts'];
export async function loadProgram(spec, read = path => fetch(path).then(response => {
    if (!response.ok)
        throw new Error(`Missing programme: ${path}`);
    return response.text();
})) {
    const files = Object.fromEntries(await Promise.all(sourceNames.map(async (name) => [name, await read(`./programs/${spec.id}/${name}`)])));
    return { files, reducer: 'reduce.nl', view: 'view.nl' };
}
/** Explicit controls only: this exercises the interpreter, never imitates semantic inference. */
export function fixtureTurn(spec, getState, getEvent) {
    return turn => {
        const prompt = String(turn.messages.find(row => row.role === 'user')?.content ?? '');
        if (turn.messages.length > 2)
            return { text: 'done' };
        const lines = [...prompt.matchAll(/^\s*(\d+) \[ \]/gm)].map(match => Number(match[1]));
        const mark = ['mark_lines', { start: 1, end: Math.max(1, ...lines) }];
        const evaluate = code => ({ calls: [['eval', { code }], mark], completion_tokens: 1 });
        if (prompt.includes(`Drive the ${spec.title} interaction to completion.`))
            return evaluate('const decision = await choose(state, event); const step = await perform(state, event, decision); result = await finish(step)');
        if (prompt.includes(spec.instructions)) {
            const event = getEvent();
            if (event.kind === 'command')
                throw new Error('Free-form commands require a loaded model. Use the explicit controls in fixture mode.');
            return evaluate(`result = ${JSON.stringify(controlDecision(spec, event))}`);
        }
        const state = getState();
        return evaluate(`result = ${JSON.stringify({ heading: spec.title, summary: state.notice, focus: spec.panelIds, suggestions: [] })}`);
    };
}
