import { controlDecision } from './contracts.mjs';
/** Bundle real natlang sources with an app's portable types and native host binding. */
export const sourceNames = ['reduce.nl', 'reduce/choose.nl', 'reduce/apply.ts', 'reduce/finish.ts', 'view.nl', 'types.ts'];
export async function loadProgram(spec, read = path => fetch(path).then(response => {
    if (!response.ok)
        throw new Error(`Missing programme: ${path}`);
    return response.text();
})) {
    const files = Object.fromEntries(await Promise.all(sourceNames.map(async (name) => [`${spec.id}/${name}`, await read(`./programs/${spec.id}/${name}`)])));
    return { files, reducer: `${spec.id}/reduce.nl`, view: `${spec.id}/view.nl` };
}
/** Explicit controls only: this exercises the interpreter, never imitates semantic inference. */
export function fixtureTurn(spec, getState, getEvent) {
    return turn => {
        if (turn.messages.filter(row => row.role === 'assistant').length > 1)
            return { calls: [], text: 'done', completion_tokens: 1 };
        const prompt = String(turn.messages.find(row => row.role === 'user')?.content ?? '');
        if (prompt.includes(`Drive the ${spec.title} interaction to completion.`))
            return { calls: [
                    ['call', { function: 'choose', to: 'let/decision', inputs: { state: 'args/state', event: 'args/event' } }],
                    ['call', { function: 'apply', to: 'let/step', inputs: { state: 'args/state', event: 'args/event', decision: 'let/decision' } }],
                    ['call', { function: 'finish', to: 'return', inputs: { step: 'let/step' } }],
                ], completion_tokens: 1 };
        if (prompt.includes(spec.instructions)) {
            const event = getEvent();
            if (event.kind === 'command')
                throw new Error('Free-form commands require a loaded model. Use the explicit controls in fixture mode.');
            return { calls: [['write', { path: 'return', value: controlDecision(spec,event) }]], completion_tokens: 1 };
        }
        const state = getState();
        return { calls: [['write', { path: 'return', value: { heading: spec.title,
                            summary: state.notice, focus: spec.panelIds, suggestions: [] } }]], completion_tokens: 1 };
    };
}
