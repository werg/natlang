/** The timer that interrupts a Python cell (native/folder-python.ts), by platform. */

/**
 * Starts a timer outside the interpreter's thread that sets the interrupt flag (`buffer`, an Int32Array's) to 2 after
 * `ms` (Pyodide then raises KeyboardInterrupt). A platform supplies one (`Platform.pythonWatchdog`): browsers use a
 * Web Worker (`webWorkerWatchdog`), Node worker_threads (runtime/node.ts).
 */
export type PythonWatchdog = (buffer: SharedArrayBuffer, ms: number) => { terminate(): Promise<unknown> | unknown } | undefined;
// Pyodide consumes and clears the signal slot. Keep requesting interruption
// after the deadline until the caller terminates this watchdog; a consumed
// signal must not leave a still-running C iterator without further requests.
export const WATCHDOG_SCRIPT = 'const flag = new Int32Array(buffer); setTimeout(() => { const interrupt = () => Atomics.store(flag, 0, 2); interrupt(); setInterval(interrupt, Math.max(1, Math.min(25, ms))); }, ms)';
export const webWorkerWatchdog: PythonWatchdog = (buffer, ms) => {
  if (typeof Worker === 'undefined') return undefined;
  const workerUrl = URL.createObjectURL(new Blob([`onmessage = ({ data }) => { const { buffer, ms } = data; ${WATCHDOG_SCRIPT}; };`],
    { type: 'text/javascript' }));
  const worker = new Worker(workerUrl);
  worker.postMessage({ buffer, ms });
  return { terminate: () => { worker.terminate(); URL.revokeObjectURL(workerUrl); } };
};
