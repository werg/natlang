/** A console writer for one eval; the runtime pages long output for the model. */
export function consoleWriter(logs, show) {
    return (...values) => {
        logs.push(values.map(value => typeof value === 'string' ? value : value === undefined ? 'undefined' : show(value)).join(' '));
    };
}
/** Settle with `work`, or fail once `timeoutMs` passes; work already started is not stopped. */
export async function withinTimeout(work, timeoutMs) {
    if (timeoutMs === undefined)
        return work;
    let timer;
    const expired = new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error(`eval timed out after ${timeoutMs} ms; the natural-language calls it started were stopped, ` +
            'and service calls it already made may still complete')), timeoutMs);
    });
    try {
        return await Promise.race([work, expired]);
    }
    finally {
        clearTimeout(timer);
    }
}
export class EvalFailure extends Error {
    events;
    debug;
    constructor(message, events, debug = {}) {
        super(message);
        this.events = events;
        this.debug = debug;
        this.name = 'EvalFailure';
    }
}
