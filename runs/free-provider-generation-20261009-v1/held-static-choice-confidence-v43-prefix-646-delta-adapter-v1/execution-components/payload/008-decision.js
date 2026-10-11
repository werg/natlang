export const DECISION_READOUT_MODES = ['decision'];
/** Every value of a finite type, in declaration order; null when the type is not finite. */
export function finiteValues(type, env) {
    const out = [];
    const visit = (item, depth) => {
        if (depth > 32)
            return false;
        const resolved = item.kind === 'name' ? env.resolve(item) : item;
        switch (resolved.kind) {
            case 'refined':
            case 'untrusted': return visit(resolved.base, depth + 1);
            case 'lit':
                out.push(resolved.value);
                return true;
            case 'union': return resolved.members.every(member => visit(member, depth + 1));
            case 'prim':
                if (resolved.name === 'boolean') {
                    out.push(true, false);
                    return true;
                }
                if (resolved.name === 'null') {
                    out.push(null);
                    return true;
                }
                return false;
            default: return false;
        }
    };
    if (!visit(type, 0))
        return null;
    return [...new Map(out.map(value => [JSON.stringify(value), value])).values()];
}
/** The readout's system prompt: the call's opening stays, the tool manual does not (no tools are offered). */
export const DECISION_SYSTEM_PROMPT = 'You are running one call of a natural-language function. Carry out its ' +
    "instructions yourself for this call's argument values, judging the visible inputs. Your reply is the function's " +
    'return value: one JSON value of the declared type, with nothing else.';
export const decisionPrompt = (replies) => 'Answer this call now, without tools. Reply with exactly one of these JSON values and nothing else: ' +
    replies.join(', ') + '.';
/** The driver's decision scorer, if its transport has one. */
export function decisionScorer(driver) {
    const decide = driver?.decide;
    return typeof decide === 'function' ? decide : undefined;
}
/**
 * Score several decision requests at once. A scorer with its own `scoreMany` (batched on the server, or coalescing
 * the readouts of concurrent calls) is used; otherwise the items are issued concurrently, so the model scheduler and
 * the server see them together. One entry per item; an item that fails does so alone.
 */
export function scoreMany(scorer, items, signal) {
    return scorer.scoreMany ? scorer.scoreMany(items, signal) : Promise.allSettled(items.map(item => scorer(item, signal)));
}
/** Normalised probabilities from log-probabilities (stable softmax). */
export function softmax(logProbs) {
    const top = Math.max(...logProbs);
    const weights = logProbs.map(value => Math.exp(value - top));
    const total = weights.reduce((sum, value) => sum + value, 0);
    return weights.map(value => value / total);
}
