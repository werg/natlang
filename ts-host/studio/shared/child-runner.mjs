/** One cancellable child execution; the parent interpreter yields while it runs. */
export function runChild(payload, { model, signal, onProgress = () => { } } = {}) {
    if (signal?.aborted)
        return Promise.reject(new Error('Child run cancelled'));
    const worker = new Worker(new URL('./child-worker.mjs', import.meta.url), { type: 'module' });
    return new Promise((resolve, reject) => {
        let settled = false;
        const finish = (error, result) => { if (settled)
            return; settled = true; signal?.removeEventListener('abort', cancel); worker.terminate(); error ? reject(error) : resolve(result); };
        const cancel = () => finish(new Error('Child run cancelled'));
        signal?.addEventListener('abort', cancel, { once: true });
        worker.onerror = event => finish(new Error(event.message));
        worker.onmessage = async ({ data }) => {
            if (data.kind === 'result')
                return finish(null, data.result);
            if (data.kind === 'error')
                return finish(new Error(data.error));
            if (data.kind === 'model-turn') {
                onProgress('Interpreting child program…');
                try {
                    if (!model)
                        throw new Error('This child program needs a loaded interpreter model');
                    const result = await model.turn(data.request, signal);
                    if (!settled)
                        worker.postMessage({ kind: 'model-result', id: data.id, result });
                }
                catch (error) {
                    if (!settled)
                        worker.postMessage({ kind: 'model-result', id: data.id, error: String(error) });
                }
            }
        };
        onProgress('Executing child program…');
        worker.postMessage({ kind: 'run', ...payload });
    });
}
